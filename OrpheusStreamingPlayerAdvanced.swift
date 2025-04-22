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
            "response_format": "pcm_f32le",
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
            try audioSession.setActive(true)
            print("[Orpheus] Audio session configured")
        } catch {
            print("[Orpheus] Warning: Failed to configure audio session: \(error)")
        }
        
        // Set up engine with the known format - float32 PCM at 24kHz mono
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        expectedFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: true)
        
        guard let engine = engine, let playerNode = playerNode, let inputFormat = expectedFormat else {
            print("[Orpheus] Failed to create audio engine, player node, or format")
            return
        }
        
        // Determine the hardware output format (typically 44.1kHz/48kHz stereo)
        let hardwareFormat = engine.outputNode.inputFormat(forBus: 0)
        print("[Orpheus] Hardware output format: sampleRate=\(hardwareFormat.sampleRate), channels=\(hardwareFormat.channelCount)")
        
        // Connect and start the engine – use the hardware format so that scheduled buffers
        // must match this format (we will convert from inputFormat -> hardwareFormat later).
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
        
        // Directly schedule the buffer - we're receiving raw pcm_f32le data
        scheduleBuffer(with: data)
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

    private func scheduleBuffer(with data: Data) {
        guard let format = expectedFormat, let playerNode = playerNode, let engine = engine else {
            print("[Orpheus] scheduleBuffer: Missing format or playerNode or engine")
            return
        }
        
        // For pcm_f32le, each sample is 4 bytes
        let bytesPerSample = 4
        let frameCapacity = UInt32(data.count) / UInt32(bytesPerSample)
        
        guard frameCapacity > 0 else {
            print("[Orpheus] Not enough data to create audio buffer")
            return
        }
        
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity) else {
            print("[Orpheus] Failed to create AVAudioPCMBuffer with frameCapacity=\(frameCapacity)")
            return
        }
        inputBuffer.frameLength = frameCapacity
        
        // Fill the buffer with float32 PCM data
        data.withUnsafeBytes { rawBuffer in
            if let sourcePtr = rawBuffer.baseAddress?.bindMemory(to: Float32.self, capacity: Int(frameCapacity)) {
                guard let floatChannelData = inputBuffer.floatChannelData else {
                    print("[Orpheus] Error: Could not get float channel data")
                    return
                }
                let destPtr = floatChannelData[0]
                for i in 0..<Int(frameCapacity) {
                    var sample = sourcePtr[i]
                    var rawValue: UInt32 = 0
                    memcpy(&rawValue, &sample, 4)
                    if abs(sample) > 10 {
                        rawValue = CFSwapInt32LittleToHost(rawValue)
                        memcpy(&sample, &rawValue, 4)
                    }
                    // Clamp extreme values to valid audio range
                    if sample > 1.0 { sample = 1.0 }
                    if sample < -1.0 { sample = -1.0 }
                    
                    // Store in buffer
                    destPtr[i] = sample
                }
                
                print("[Orpheus] Processed \(frameCapacity) float32 frames with conversion")
            } else {
                print("[Orpheus] Failed to bind memory to Float32")
                return
            }
        }
        
        // Schedule buffer directly and let AVAudioEngine handle conversion
        playerNode.scheduleBuffer(inputBuffer, at: nil, options: [], completionHandler: nil)
        print("[Orpheus] Scheduled buffer: frameLength=\(inputBuffer.frameLength)")
    }
}
