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
            try audioSession.setPreferredSampleRate(24000)
            try audioSession.setActive(true)
            print("[Orpheus] Audio session configured")
        } catch {
            print("[Orpheus] Warning: Failed to configure audio session: \(error)")
        }
        
        // Set up engine with the known format - float32 PCM at 24kHz mono
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        expectedFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24000, channels: 1, interleaved: true)
        
        guard let engine = engine, let playerNode = playerNode, let inputFormat = expectedFormat else {
            print("[Orpheus] Failed to create audio engine, player node, or format")
            return
        }
        
        // Determine the hardware output format (typically 44.1kHz/48kHz stereo)
        let hardwareFormat = engine.outputNode.inputFormat(forBus: 0)
        print("[Orpheus] Hardware output format: sampleRate=\(hardwareFormat.sampleRate), channels=\(hardwareFormat.channelCount)")
        
        // Connect and start the engine - use the expected format (float32 PCM) directly
        // This ensures no format conversion happens which could introduce artifacts
        engine.attach(playerNode)
        // Directly connect to hardware output to bypass mixer and avoid resampling
        engine.connect(playerNode, to: engine.outputNode, format: inputFormat)
        
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

    // Buffer for incomplete WAV header/data
    private var wavDataBuffer = Data()
    private var wavHeaderParsed = false
    private var wavDataStartIndex: Int = 0

    // Handles streaming WAV data, parses header, and schedules int16 PCM buffers
    private func bufferWavAndSchedule(with data: Data) {
        wavDataBuffer.append(data)

        // Parse WAV header if not yet done
        if !wavHeaderParsed {
            if wavDataBuffer.count >= 44 {
                // Standard WAV header is 44 bytes
                wavHeaderParsed = true
                wavDataStartIndex = 44
                print("[Orpheus] WAV header parsed. Data starts at byte 44.")
            } else {
                // Wait for more data
                return
            }
        }

        guard let format = expectedFormat, let playerNode = playerNode, let engine = engine else {
            print("[Orpheus] bufferWavAndSchedule: Missing format or playerNode or engine")
            return
        }

        // Schedule all available PCM frames (after header)
        let pcmData = wavDataBuffer.subdata(in: wavDataStartIndex..<wavDataBuffer.count)
        let bytesPerSample = 2 // int16
        let frameCapacity = UInt32(pcmData.count) / UInt32(bytesPerSample)
        guard frameCapacity > 0 else { return }

        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else {
            print("[Orpheus] Failed to create AVAudioPCMBuffer with frameCapacity=\(frameCapacity)")
            return
        }
        inputBuffer.frameLength = frameCapacity

        // Fill buffer with int16 PCM data
        pcmData.withUnsafeBytes { rawBuffer in
            if let sourcePtr = rawBuffer.baseAddress?.assumingMemoryBound(to: Int16.self) {
                guard let int16ChannelData = inputBuffer.int16ChannelData else {
                    print("[Orpheus] Error: Could not get int16 channel data")
                    return
                }
                let destPtr = int16ChannelData[0]
                for i in 0..<Int(frameCapacity) {
                    destPtr[i] = sourcePtr[i]
                }
            }
        }
        playerNode.scheduleBuffer(inputBuffer, at: nil, options: [], completionHandler: nil)
        print("[Orpheus] Scheduled buffer: frameLength=\(inputBuffer.frameLength)")

        // Move buffer start index forward
        wavDataStartIndex = wavDataBuffer.count
    }
}
