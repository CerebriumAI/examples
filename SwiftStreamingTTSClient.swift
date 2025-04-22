import Accelerate
import AVFoundation

/// Advanced streaming TTS player that streams WAV audio from Orpheus API and plays it continuously.
class OrpheusStreamingPlayerAdvanced: NSObject {
    // MARK: - Audio Engine Components
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let mainMixer: AVAudioMixerNode
    private var audioFormat: AVAudioFormat?
    private var sourceFormat: AVAudioFormat?
    private var outputFormat: AVAudioFormat?
    private var framesPerBuffer: AVAudioFrameCount = 0
    private var bufferDuration: TimeInterval = 0.1
    private var headerParsed = false

    // MARK: - Processing Queue
    private let processingQueue = DispatchQueue(label: "OrpheusStreamingProcessingQueue")

    // MARK: - Networking
    private var session: URLSession!
    private var dataTask: URLSessionDataTask?

    // MARK: - Callbacks
    private var playbackCompletion: (() -> Void)?

    // MARK: - Initialization
    override init() {
        self.mainMixer = engine.mainMixerNode
        super.init()

        // Attach and connect player node
        engine.attach(playerNode)
        engine.connect(playerNode, to: mainMixer, format: nil)

        // Install tap on mixer to log audio levels
        installLevelTap(on: mainMixer)


        // Configure URLSession
        let config = URLSessionConfiguration.default
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    deinit {
        dataTask?.cancel()
    }

    // MARK: - Public API

    /// Starts streaming TTS audio for given text and optional voice.
    /// - Parameters:
    ///   - text: Text to synthesize and stream
    ///   - voice: Voice identifier (default: "tara")
    ///   - completion: Called when playback finishes
    func streamAudio(text: String,
                     voice: String = "tara",
                     completion: (() -> Void)? = nil) {
        print("[streamAudio] Called with text: \(text.prefix(100))... voice: \(voice)")
        playbackCompletion = completion

        // Prepare URL and Request
        guard let url = URL(string: "http://34.125.197.177:5005/v1/audio/speech/stream") else {
            print("[streamAudio] Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "input": text,
            "voice": voice,
            "model": "orpheus",
            "response_format": "pcm_f32le",
            "speed": 1.0
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        print("[streamAudio] Request body: \(body)")

        // Start audio engine
        do {
            print("[streamAudio] Starting audio engine...")
            try engine.start()
            try AVAudioSession.sharedInstance().setActive(true)
            try AVAudioSession.sharedInstance().overrideOutputAudioPort(.speaker)
            playerNode.play()
            print("[streamAudio] Audio engine started and player node playing.")
        } catch {
            print("[streamAudio] Failed to start engine: \(error)")
            return
        }

        // Begin streaming
        print("[streamAudio] Starting data task for streaming...")
        dataTask = session.dataTask(with: request)
        dataTask?.resume()
    }

    // MARK: - Audio Level Metering
    private func installLevelTap(on node: AVAudioNode) {
        node.installTap(onBus: 0, bufferSize: 1024, format: node.outputFormat(forBus: 0)) { buffer, when in
            guard let channelData = buffer.floatChannelData?[0] else { return }
            let channelDataValueArray = stride(from: 0,
                                               to: Int(buffer.frameLength),
                                               by: buffer.stride).map { channelData[$0] }
            let rms = sqrt(channelDataValueArray.map { $0 * $0 }.reduce(0, +) / Float(buffer.frameLength))
            let avgPower = 20 * log10(rms)
            print(String(format: "Audio Level: %.2f dBFS", avgPower))
        }
    }


    private func resetBuffers() {
        sourceFormat = nil
        outputFormat = nil
        framesPerBuffer = 0
        headerParsed = false
    }

    private func setupAudioEngine(with format: AVAudioFormat) {
        print("[setupAudioEngine] Setting up with format: \(format)")
        if engine.isRunning {
            print("[setupAudioEngine] Engine already running, skipping reset.")
            return
        }
        engine.stop()
        engine.reset()
        playerNode.stop()
        playerNode.reset()
        engine.attach(playerNode)
        engine.connect(playerNode, to: mainMixer, format: format)
        do {
            try engine.start()
            try AVAudioSession.sharedInstance().setActive(true)
            try AVAudioSession.sharedInstance().overrideOutputAudioPort(.speaker)
            playerNode.play()
            print("[setupAudioEngine] Engine started and player node playing.")
        } catch {
            print("[setupAudioEngine] Failed to start engine: \(error)")
        }
    }

    private func scheduleAudioChunk(_ chunk: Data, format: AVAudioFormat) {
        print("[scheduleAudioChunk] Received chunk of size: \(chunk.count) bytes")
        let bytesPerFrame = MemoryLayout<Float>.size * Int(format.channelCount)
        let frameCount = chunk.count / bytesPerFrame
        let leftover = chunk.count % bytesPerFrame
        if leftover != 0 {
            print("[scheduleAudioChunk] Warning: chunk has leftover bytes (\(leftover)), truncating to nearest frame.")
        }
        guard frameCount > 0 else {
            print("[scheduleAudioChunk] Chunk too small, skipping.")
            return
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            print("[scheduleAudioChunk] Buffer creation failed")
            return
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        chunk.withUnsafeBytes { (ptr: UnsafePointer<Float>) in
            let data = buffer.floatChannelData!
            for ch in 0..<Int(format.channelCount) {
                let dst = data[ch]
                for i in 0..<frameCount {
                    dst[i] = ptr[i * Int(format.channelCount) + ch]
                }
            }
            let firstSamples = (0..<min(10, frameCount)).map { data[0][$0] }
            print("[scheduleAudioChunk] First 10 float samples: \(firstSamples)")
        }
        playerNode.scheduleBuffer(buffer, completionHandler: nil)
        print("[scheduleAudioChunk] Buffer scheduled for playback (\(frameCount) frames).")
    }
}

// MARK: - URLSession Delegate
extension OrpheusStreamingPlayerAdvanced: URLSessionDataDelegate {
    func urlSession(_ session: URLSession, dataTask: URLSessionTask, didReceive data: Data) {
        print("[urlSession:didReceive] Received data chunk of size: \(data.count) bytes")
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            if self.outputFormat == nil {
                let sampleRate: Double = 24000
                let channelCount = AVAudioChannelCount(1)
                let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: channelCount, interleaved: false)!
                self.outputFormat = fmt
                self.framesPerBuffer = AVAudioFrameCount(sampleRate * self.bufferDuration)
                print("[urlSession:didReceive] Setting up audio engine for first chunk...")
                self.setupAudioEngine(with: fmt)
            }
            guard let fmt = self.outputFormat else {
                print("[urlSession:didReceive] No output format available!")
                return
            }
            self.scheduleAudioChunk(data, format: fmt)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("Streaming error: \(error)")
        }
        if let format = self.audioFormat {
            let emptyBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1)!
            emptyBuffer.frameLength = 1
            self.playerNode.scheduleBuffer(emptyBuffer, at: nil) {
                self.engine.stop()
                DispatchQueue.main.async {
                    self.playbackCompletion?()
                }
            }
        }
    }
}
