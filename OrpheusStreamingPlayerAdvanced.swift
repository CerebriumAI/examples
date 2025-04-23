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
    private var headerData = Data()
    private var headerParsed = false
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
        expectedFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: hardwareFormat.sampleRate, channels: hardwareFormat.channelCount, interleaved: false)
        guard let inputFormat = expectedFormat else {
            print("[Orpheus] Failed to create expected AVAudioFormat")
            return
        }
        
        // Attach player and connect through main mixer for automatic format conversion
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: inputFormat)
        
        do {
            try engine.start()
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
        bufferWavAndSchedule(with: data)
    }

    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[Orpheus] Streaming error: \(error)")
        } else {
            print("[Orpheus] Streaming completed.")
        }
        completionHandler?()
    }

    // Parse WAV header once, then schedule each chunk as its own PCM buffer
    private func bufferWavAndSchedule(with data: Data) {
        var pcmData = data
        // On first chunk, accumulate header bytes
        if !headerParsed {
            headerData.append(data)
            guard headerData.count >= 44 else { return }
            let header = headerData.prefix(44)
            let sampleRate = header.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt32.self).littleEndian }
            let channels = header.withUnsafeBytes { $0.load(fromByteOffset: 22, as: UInt16.self).littleEndian }
            expectedFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                           sampleRate: Double(sampleRate),
                                           channels: AVAudioChannelCount(channels),
                                           interleaved: false)
            if let engine = engine, let node = playerNode, let fmt = expectedFormat {
                engine.disconnectNodeInput(node)
                engine.connect(node, to: engine.mainMixerNode, format: fmt)
            }
            headerParsed = true
            // Extract PCM data after header
            pcmData = headerData.dropFirst(44)
            headerData.removeAll()
        }
        // Schedule this PCM data chunk
        guard let fmt = expectedFormat, let node = playerNode else { return }
        let bytesPerFrame = Int(fmt.streamDescription.pointee.mBytesPerFrame)
        let frameCount = UInt32(pcmData.count) / UInt32(bytesPerFrame)
        guard frameCount > 0, let buffer = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frameCount) else { return }
        buffer.frameLength = frameCount
        pcmData.withUnsafeBytes { raw in
            let ptr = raw.baseAddress!.assumingMemoryBound(to: Int16.self)
            for ch in 0..<Int(fmt.channelCount) {
                let channelData = buffer.int16ChannelData![ch]
                for frame in 0..<Int(frameCount) {
                    channelData[frame] = ptr[frame * Int(fmt.channelCount) + ch]
                }
            }
        }
        node.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        scheduledBufferCount += 1
        if !playbackStarted && scheduledBufferCount >= minBuffersBeforePlayback {
            node.play()
            playbackStarted = true
        }
    }
}
