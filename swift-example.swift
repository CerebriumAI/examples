/// A client capable of streaming WAV audio from an Orpheus FastAPI server and playing it as it arrives.
class OrpheusStreamingPlayerAdvanced: NSObject, URLSessionDataDelegate {
    // MARK: - Properties
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let converterNode = AVAudioMixerNode()
    private var sourceFormat: AVAudioFormat?
    private var outputFormat: AVAudioFormat?
    private var session: URLSession!
    private var dataTask: URLSessionDataTask?
    private var headerBuffer = Data()
    private var headerParsed = false
    private var completionHandler: (() -> Void)?
    private var streamingEnded = false
    private var pendingData = Data()

    // Enhanced buffering system
    private var audioProcessingQueue = DispatchQueue(label: "com.orpheus.audioProcessing", qos: .userInteractive)
    private var audioChunks = [Data]()
    private var audioChunksLock = NSLock()
    private var isSchedulingBuffers = false
    private var bufferSemaphore = DispatchSemaphore(value: 1)

    // Buffer settings for smooth playback
    private let prefillBufferCount = 6      // Number of buffers to fill before starting playback (increased for underrun protection)
    private let bufferDuration = 0.1        // Duration of each audio buffer in seconds
    private let maximumBufferedDuration = 2.0  // Maximum audio to buffer ahead in seconds
    private var initialBufferThreshold: Int {
        Int(ceil(maximumBufferedDuration / bufferDuration))
    }
    private var lastScheduledTime: AVAudioTime?
    private var converter: AVAudioConverter?
    private var needsScheduling = false
    private var isPlaying = false
    private var framesPerBuffer: AVAudioFrameCount = 4410 // Will be recalculated based on format

    // Stats for debugging
    private var totalScheduledFrames: UInt64 = 0
    private var bufferUnderrunCount = 0
    private var lastDebugPrintTime = Date()

    /// Initializes the audio session, engine, and URLSession for streaming.
    override init() {
        super.init()
        setupAudioSession()
        setupAudioEngine()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    // MARK: - Setup

    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("AVAudioSession setup error: \(error)")
        }
    }

    private func setupAudioEngine() {
        // Output format is fixed to system rate for maximum compatibility
        outputFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)

        // Attach and connect nodes with the output format
        engine.attach(playerNode)
        engine.attach(converterNode)

        // Connect with output format (will be properly configured later)
        engine.connect(playerNode, to: engine.mainMixerNode, format: outputFormat)

        // Set up buffer observer to detect buffer underruns
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleEngineConfigurationChange),
            name: .AVAudioEngineConfigurationChange,
            object: engine
        )
    }

    @objc private func handleEngineConfigurationChange(_ notification: Notification) {
        print("Audio engine configuration changed")

        // Restart the engine if needed
        if !engine.isRunning {
            startEngine()
        }
    }

    private func startEngine() {
        if engine.isRunning { return }

        do {
            try engine.start()
            print("Audio engine started")
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    // MARK: - Public API

    /// Streams TTS audio from the server and plays it.
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: The voice identifier (default: "tara").
    ///   - completion: Optional callback when streaming completes or errors.
    func streamAudio(text: String,
                     voice: String = "tara",
                     completion: (() -> Void)? = nil) {
        completionHandler = completion
        guard let url = URL(string: "http://34.125.197.177:5005/v1/audio/speech/stream") else {
            print("Invalid server URL")
            return
        }

        // Build JSON payload
        let payload: [String: Any] = [
            "input": text,
            "voice": voice,
            "response_format": "wav"
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            print("Failed to encode JSON payload")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        // Reset state
        reset()

        // Begin streaming
        dataTask = session.dataTask(with: request)
        dataTask?.resume()
    }

    func stop() {
        isPlaying = false
        playerNode.stop()
        engine.stop()
        dataTask?.cancel()
        reset()
    }

    private func reset() {
        audioChunksLock.lock()
        audioChunks.removeAll()
        audioChunksLock.unlock()

        headerBuffer.removeAll()
        headerParsed = false
        isSchedulingBuffers = false
        lastScheduledTime = nil
        totalScheduledFrames = 0
        bufferUnderrunCount = 0
        isPlaying = false
        lastDebugPrintTime = Date()
        streamingEnded = false
        pendingData.removeAll()
    }

    // MARK: - Audio Processing

    /// Starts the scheduling process if not already running
    private func ensureSchedulingActive() {
        let wasScheduling = isSchedulingBuffers
        audioChunksLock.lock()
        let pendingChunks = audioChunks.count
        audioChunksLock.unlock()
        // Only start scheduling if we have enough prefill buffers
        if !wasScheduling && isPlaying && pendingChunks >= prefillBufferCount {
            isSchedulingBuffers = true
            audioProcessingQueue.async { [weak self] in
                self?.processAudioChunks()
            }
        }
    }

    /// Main audio processing loop - runs on audioProcessingQueue
    private func processAudioChunks() {
        // If stream ended and no pending chunks, finish playback
        audioChunksLock.lock()
        let pending = audioChunks.count
        audioChunksLock.unlock()
        if streamingEnded && pending == 0 {
            isSchedulingBuffers = false
            DispatchQueue.main.async { [weak self] in
                self?.completionHandler?()
            }
            return
        }

        guard isSchedulingBuffers else { return }

        // Get next chunk if available
        var chunk: Data?
        audioChunksLock.lock()
        if !audioChunks.isEmpty {
            chunk = audioChunks.removeFirst()
        }
        let pendingChunksCount = audioChunks.count
        audioChunksLock.unlock()

        // Process chunk if we have one
        if let chunk = chunk, let sourceFormat = sourceFormat, let converter = converter {
            processAndScheduleAudioChunk(chunk, sourceFormat: sourceFormat, converter: converter)

            // Debug output every few seconds
            let now = Date()
            if now.timeIntervalSince(lastDebugPrintTime) > 5.0 {
                print("Audio stats: \(pendingChunksCount) chunks buffered, \(bufferUnderrunCount) underruns")
                lastDebugPrintTime = now
            }
        } else {
            // No chunk available, wait briefly and check player status
            if playerNode.isPlaying {
                // Player is still going - check if we need more buffers scheduled
                if needsScheduling {
                    bufferUnderrunCount += 1
                    print("⚠️ Buffer underrun detected!")
                    needsScheduling = false
                }
            }
        }

        // Continue processing if we're still scheduling
        if isSchedulingBuffers {
            // Use short delay to prevent tight loop while allowing frequent scheduling
            DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: .now() + 0.01) { [weak self] in
                self?.audioProcessingQueue.async {
                    self?.processAudioChunks()
                }
            }
        }
    }

    /// Processes and schedules a chunk of audio data
    private func processAndScheduleAudioChunk(_ chunk: Data, sourceFormat: AVAudioFormat, converter: AVAudioConverter) {
        bufferSemaphore.wait()
        defer { bufferSemaphore.signal() }

        let bytesPerFrame = Int(sourceFormat.streamDescription.pointee.mBytesPerFrame)
        let frameCount = AVAudioFrameCount(chunk.count / bytesPerFrame)

        // Skip empty chunks
        if frameCount == 0 {
            return
        }

        // Create buffer with the original format
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            print("Failed to create source buffer")
            return
        }

        sourceBuffer.frameLength = frameCount

        // Copy chunk data to source buffer
        chunk.withUnsafeBytes { rawBufferPointer in
            let audioBuffer = sourceBuffer.audioBufferList.pointee.mBuffers
            if let destPtr = audioBuffer.mData,
               let srcPtr = rawBufferPointer.baseAddress {
                memcpy(destPtr, srcPtr, Int(audioBuffer.mDataByteSize))
            }
        }

        // Calculate output frame capacity based on sample rate ratio
        let sampleRateRatio = outputFormat!.sampleRate / sourceFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * sampleRateRatio)

        // Create output buffer
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat!, frameCapacity: outputFrameCount) else {
            print("Failed to create output buffer")
            return
        }

        // Perform the conversion
        var error: NSError?
        let conversionResult = converter.convert(to: outputBuffer, error: &error) { packetCount, status in
            status.pointee = .haveData
            return sourceBuffer
        }

        if let error = error {
            print("Conversion error: \(error)")
            return
        }

        if conversionResult == .error || outputBuffer.frameLength == 0 {
            print("Conversion failed or produced no output")
            return
        }

        // Schedule the buffer at the appropriate time
        scheduleBuffer(outputBuffer)
    }

    /// Schedules a buffer with proper timing
    private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) {
        // Calculate the time when this buffer should start
        var schedulingTime: AVAudioTime?

        if let lastTime = lastScheduledTime {
            // Calculate next buffer start time from previous end time
            let sampleTime = lastTime.sampleTime + AVAudioFramePosition(totalScheduledFrames)
            schedulingTime = AVAudioTime(sampleTime: sampleTime, atRate: outputFormat!.sampleRate)
        } else {
            // First buffer; schedule immediately (in 100ms to allow for more buffering)
            let hostTime = mach_absolute_time() + UInt64(100_000_000) // 100ms
            schedulingTime = AVAudioTime(hostTime: hostTime)
        }

        // If player isn't playing, start it
        if !playerNode.isPlaying {
            playerNode.play(at: schedulingTime)
            isPlaying = true
        }

        // Schedule buffer with completion handler to track when it finishes
        playerNode.scheduleBuffer(buffer, at: schedulingTime) { [weak self] in
            DispatchQueue.main.async {
                self?.needsScheduling = true
            }
        }

        // Update tracking variables
        lastScheduledTime = schedulingTime
        totalScheduledFrames = UInt64(buffer.frameLength)
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        if !headerParsed {
            headerBuffer.append(data)
            if headerBuffer.count >= 44 {
                parseWAVHeader(headerBuffer)
                headerParsed = true
                if headerBuffer.count > 44 {
                    let audioData = headerBuffer.suffix(from: 44)
                    pendingData.append(audioData)
                    headerBuffer.removeAll()
                }
            }
        } else {
            pendingData.append(data)
        }
        // Slice pendingData into fixed-size buffers
        guard headerParsed, let sourceFormat = sourceFormat else { return }
        let bytesPerFrame = Int(sourceFormat.streamDescription.pointee.mBytesPerFrame)
        let bytesPerBuffer = bytesPerFrame * Int(framesPerBuffer)
        while pendingData.count >= bytesPerBuffer {
            let chunk = pendingData.prefix(bytesPerBuffer)
            pendingData.removeFirst(bytesPerBuffer)
            enqueueAudioData(chunk)
        }
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        if let error = error {
            print("Streaming error: \(error)")
        } else {
            print("Streaming completed successfully")
        }
        // Flush leftover pendingData
        if headerParsed && pendingData.count > 0 {
            enqueueAudioData(pendingData)
            pendingData.removeAll()
        }
        // Start playback if buffered
        audioChunksLock.lock()
        let pendingChunks = audioChunks.count
        audioChunksLock.unlock()
        if pendingChunks > 0 {
            if !isPlaying {
                isPlaying = true
                startEngine()
            }
            ensureSchedulingActive()
        }
        streamingEnded = true
    }

    // MARK: - Helpers

    /// Enqueues audio data for processing
    private func enqueueAudioData(_ data: Data) {
        // Skip empty data
        if data.isEmpty { return }

        // Add to queue
        audioChunksLock.lock()
        audioChunks.append(data)
        let pendingChunks = audioChunks.count
        audioChunksLock.unlock()

        // Only start playback once we have enough buffers
        if !isPlaying && pendingChunks >= prefillBufferCount {
            isPlaying = true
            startEngine()
            ensureSchedulingActive()
        } else if isPlaying {
            ensureSchedulingActive()
        }
    }

    /// Parses the WAV header to extract format info and configure the audio pipeline
    private func parseWAVHeader(_ header: Data) {
        guard header.count >= 44 else { return }

        let bytes = [UInt8](header)
        let channels = UInt16(bytes[22]) | (UInt16(bytes[23]) << 8)
        let rate = UInt32(bytes[24]) |
        (UInt32(bytes[25]) << 8) |
        (UInt32(bytes[26]) << 16) |
        (UInt32(bytes[27]) << 24)
        let bits = UInt16(bytes[34]) | (UInt16(bytes[35]) << 8)

        print("WAV format: \(rate)Hz, \(channels) channels, \(bits) bits")

        // Create the input format
        sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(rate),
            channels: AVAudioChannelCount(channels),
            interleaved: true)

        // Create/update the output format
        outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 44100, // Fixed rate for consistency
            channels: AVAudioChannelCount(channels),
            interleaved: false)

        // Calculate frames per buffer based on desired buffer duration
        framesPerBuffer = AVAudioFrameCount(outputFormat!.sampleRate * bufferDuration)

        // Create the converter
        converter = AVAudioConverter(from: sourceFormat!, to: outputFormat!)
        if converter == nil {
            print("Failed to create audio converter")
            return
        }

        // Reset audio engine, ensure connections are correct
        resetAudioEngine()
    }

    /// Reconfigures the audio engine with the current formats
    private func resetAudioEngine() {
        // Stop everything
        if engine.isRunning {
            engine.stop()
        }

        // Reset connections
        engine.disconnectNodeInput(playerNode)
        engine.disconnectNodeInput(engine.mainMixerNode)

        // Reconnect with proper format
        engine.connect(playerNode, to: engine.mainMixerNode, format: outputFormat)

        // Don't start yet - we'll start when we have enough buffered data
    }
}
