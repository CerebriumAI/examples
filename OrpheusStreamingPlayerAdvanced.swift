import Foundation
import AVFoundation

public class OrpheusStreamingPlayerAdvanced: NSObject {
    private let baseURL = URL(string: "http://34.125.197.177:5005")!
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var expectedFormat: AVAudioFormat?
    private var session: URLSession?
    private var dataTask: URLSessionDataTask?
    private var completionHandler: (() -> Void)?
    // No header processing for raw PCM
    // Moved from extension:
    private var wavDataBuffer = Data()
    private var wavHeaderParsed = false
    private var playbackStarted = false
    private var scheduledBufferCount = 0
    private let minBuffersBeforePlayback = 4

    public override init() {
        super.init()
    }

    public func streamAudio(text: String, voice: String = "tara", completion: (() -> Void)? = nil) {
        print("[Orpheus] Starting streamAudio with text=\(text.prefix(30))..., voice=\(voice)")
        let url = baseURL.appendingPathComponent("/v1/audio/speech/stream")
        print("[Orpheus] Streaming URL: \(url)")

        // Prepare JSON body
        let body: [String: Any] = [
            "input": text,
            "voice": voice,
            "model": "orpheus",
            "response_format": "wav",
            "speed": 1.0
        ]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: body, options: []) else {
            print("[Orpheus] Failed to encode JSON body")
            return
        }

        // Configure audio session for playback
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            // Removed setPreferredSampleRate(24000) to avoid forcing unsupported sample rate
            try audioSession.setActive(true)
            print("[Orpheus] Audio session configured")
        } catch {
            print("[Orpheus] Warning: Failed to configure audio session: \(error)")
        }
        
        // Set up engine and player node
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        
        guard let engine = engine, let playerNode = playerNode else {
            print("[Orpheus] Failed to create audio engine or player node")
            return
        }
        
        // Determine the hardware output format (typically 44.1kHz/48kHz stereo)
        let hardwareFormat = engine.outputNode.inputFormat(forBus: 0)
        print("[Orpheus] Hardware output format: sampleRate=\(hardwareFormat.sampleRate), channels=\(hardwareFormat.channelCount)")
        
        // Use hardware format for expectedFormat to avoid format mismatch
        expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: hardwareFormat.sampleRate, channels: hardwareFormat.channelCount, interleaved: false)
        guard let inputFormat = expectedFormat else {
            print("[Orpheus] Failed to create expected AVAudioFormat")
            return
        }
        
        // Attach player and connect through main mixer for automatic format conversion
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: inputFormat)
        
        do {
            try engine.start()
            playerNode.play()
            print("[Orpheus] Audio engine started")
        } catch {
            print("[Orpheus] Error starting audio engine: \(error)")
            return
        }

        self.completionHandler = completion

        let config = URLSessionConfiguration.default
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData
        print("[Orpheus] Sending POST request with JSON body: \(body)")
        dataTask = session?.dataTask(with: request)
        dataTask?.resume()
    }

    deinit {
        session?.invalidateAndCancel()
    }
}

extension OrpheusStreamingPlayerAdvanced: URLSessionDataDelegate {
    public func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        print("[Orpheus] Received data chunk of size: \(data.count)")
        guard let expectedFormat = expectedFormat, let playerNode = playerNode else {
            print("[Orpheus] Missing format or playerNode")
            return
        }
        
        // Buffer WAV data and decode into int16 PCM chunks
        bufferWavAndSchedule(with: data)
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[Orpheus] Streaming error: \(error)")
        } else {
            print("[Orpheus] Streaming completed successfully.")
        }
        playerNode?.stop()
        engine?.stop()
        print("[Orpheus] Audio engine and player node stopped.")
        completionHandler?()
    }

    // Buffer for incoming streamed WAV data
    private let chunkFrameCount = 2048  // frames per buffer
    private func bufferWavAndSchedule(with data: Data) {
        wavDataBuffer.append(data)

        // Parse WAV header if not yet done
        if !wavHeaderParsed {
            guard wavDataBuffer.count >= 44 else { return }
            // Parse WAV header for format info
            let header = wavDataBuffer.prefix(44)
            let sampleRate = header.withUnsafeBytes { ptr -> UInt32 in
                ptr.load(fromByteOffset: 24, as: UInt32.self).littleEndian
            }
            let channels = header.withUnsafeBytes { ptr -> UInt16 in
                ptr.load(fromByteOffset: 22, as: UInt16.self).littleEndian
            }
            print("[Orpheus] WAV header parsed. sampleRate=\(sampleRate), channels=\(channels)")
            wavHeaderParsed = true
            // Update expectedFormat to match WAV header
            expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                           sampleRate: Double(sampleRate),
                                           channels: AVAudioChannelCount(channels),
                                           interleaved: false)
            // Remove header bytes
            wavDataBuffer.removeFirst(44)
        }

        guard let format = expectedFormat, let playerNode = playerNode else {
            print("[Orpheus] Missing format or playerNode")
            return
        }

        let bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
        let chunkByteSize = chunkFrameCount * bytesPerFrame

        // Schedule in fixed-size chunks
        while wavDataBuffer.count >= chunkByteSize {
            guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format,
                                                  frameCapacity: AVAudioFrameCount(chunkFrameCount)) else { break }
            pcmBuffer.frameLength = AVAudioFrameCount(chunkFrameCount)
            
            wavDataBuffer.withUnsafeBytes { raw in
                let intPtr = raw.baseAddress!.assumingMemoryBound(to: Int16.self)
                let floatPtr = pcmBuffer.floatChannelData![0]
                for i in 0..<chunkFrameCount {
                    floatPtr[i] = Float(intPtr[i]) / 32768.0
                }
            }
            
            playerNode.scheduleBuffer(pcmBuffer, at: nil, options: [], completionHandler: nil)
            scheduledBufferCount += 1
            print("[Orpheus] Scheduled chunk buffer \(scheduledBufferCount)")

            if !playbackStarted && scheduledBufferCount >= minBuffersBeforePlayback {
                playerNode.play()
                playbackStarted = true
                print("[Orpheus] Playback started after \(minBuffersBeforePlayback) buffers")
            }

            // Remove data for scheduled chunk
            wavDataBuffer.removeFirst(chunkByteSize)
        }
    }
}
