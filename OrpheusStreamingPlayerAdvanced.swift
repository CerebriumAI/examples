class OrpheusStreamingPlayerAdvanced: NSObject, URLSessionDataDelegate {
    // MARK: - Properties
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var sourceFormat: AVAudioFormat?
    private var outputFormat: AVAudioFormat?
    private var session: URLSession!
    private var dataTask: URLSessionDataTask?
    private var headerBuffer = Data()
    private var headerParsed = false
    private var completionHandler: (() -> Void)?

    // Enhanced buffering system
    private var audioProcessingQueue = DispatchQueue(label: "com.orpheus.audioProcessing", qos: .userInteractive)
    private var audioChunks = [Data]()
    private var audioChunksLock = NSLock()
    private var isSchedulingBuffers = false
    private var bufferSemaphore = DispatchSemaphore(value: 1)

    // Buffer settings for smooth playback
    private var prefillBufferCount = 5      // Increased to reduce underruns
    private let bufferDuration = 0.05       // Smaller chunks for more responsive playback
    private let maximumBufferedDuration = 2.0  // Reduced to prevent excessive buffering
    private var lastScheduledTime: AVAudioTime?
    private var converter: AVAudioConverter?
    private var needsScheduling = false
    private var isPlaying = false
    private var framesPerBuffer: AVAudioFrameCount = 3528 // Will be recalculated based on format

    // Advanced jitter buffer
    private var jitterBufferEnabled = true
    private var targetBufferLevel = 4       // Adjusted target to match prefill
    private var minBufferLevel = 1          // Allow more aggressive underrun protection
    private var maxBufferLevel = 6          // Prevent excessive buffering
    private var bufferingStrategy = BufferingStrategy.adaptive

    // Stats for debugging and adaptive playback
    private var totalScheduledFrames: UInt64 = 0
    private var bufferUnderrunCount = 0
    private var lastDebugPrintTime = Date()
    private var playbackRate: Float = 1.0
    private var bufferHealthHistory = [Int]()
    private var networkJitterMs = 0.0       // Estimated network jitter in milliseconds

    // MARK: - Enums

    enum BufferingStrategy {
        case fixed      // Fixed buffer size
        case adaptive   // Adapt buffer size based on network conditions
        case aggressive // More aggressive buffering for difficult networks
    }

    /// Initializes the audio session, engine, and URLSession for streaming.
    override init() {
        super.init()
        setupAudioSession()
        setupAudioEngine()

        // Create custom URLSession configuration for audio streaming
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = true
        config.httpMaximumConnectionsPerHost = 1
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        print("OrpheusStreamingPlayer initialized with prefill count: \(prefillBufferCount)")
    }

    // MARK: - Setup

    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)

            // Use smaller IO buffer for lower latency
            let preferredIOBufferDuration = 0.005  // 5ms buffer for lower latency
            try audioSession.setPreferredIOBufferDuration(preferredIOBufferDuration)

            print("Audio session configured with \(preferredIOBufferDuration * 1000)ms IO buffer")
        } catch {
            print("AVAudioSession setup error: \(error)")
        }
    }

    private func setupAudioEngine() {
        // Use hardware sample rate for best quality
        let hwSampleRate = AVAudioSession.sharedInstance().sampleRate
        outputFormat = AVAudioFormat(standardFormatWithSampleRate: hwSampleRate, channels: 1)

        // Attach and connect nodes with the output format
        engine.attach(playerNode)

        // Set larger buffer size on the main mixer for more stability
        engine.mainMixerNode.volume = 1.0

        // Connect with output format (will be properly configured later)
        engine.connect(playerNode, to: engine.mainMixerNode, format: outputFormat)

        // Enable manual rendering mode to prevent audio dropouts
        engine.prepare()

        // Set up buffer observer to detect buffer underruns
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleEngineConfigurationChange),
            name: .AVAudioEngineConfigurationChange,
            object: engine
        )

        // Also monitor for audio interruptions
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAudioInterruption),
            name: AVAudioSession.interruptionNotification,
            object: nil
        )
    }

    @objc private func handleEngineConfigurationChange(_ notification: Notification) {
        print("Audio engine configuration changed")

        // Restart the engine if needed
        if !engine.isRunning {
            startEngine()
        }
    }

    @objc private func handleAudioInterruption(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
            return
        }

        if type == .began {
            // Interruption began, audio stopped
            print("Audio interrupted - playback paused")
        } else if type == .ended {
            // Interruption ended, resume if needed
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                if options.contains(.shouldResume) {
                    print("Audio interruption ended - resuming playback")
                    startEngine()

                    if !playerNode.isPlaying && isPlaying {
                        playerNode.play()
                    }
                }
            }
        }
    }

    private func startEngine() {
        if engine.isRunning { return }

        do {
            try engine.start()
            print("Audio engine started")
        } catch {
            print("Failed to start audio engine: \(error)")

            // Try to recover
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.recoverFromEngineFailure()
            }
        }
    }

    // Recovery method for engine failures
    private func recoverFromEngineFailure() {
        print("Attempting to recover from engine failure...")

        do {
            // Reset audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setActive(false)
            try audioSession.setActive(true)

            // Restart engine
            try engine.start()
            print("Engine recovered successfully")
        } catch {
            print("Recovery failed: \(error)")
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

        // Prepare buffering strategy based on text length
        if text.count > 500 {
            // For longer texts, use more aggressive buffering
            bufferingStrategy = .aggressive
            targetBufferLevel = 5
            prefillBufferCount = 6
            print("Using aggressive buffering for long text (\(text.count) chars)")
        } else if text.count > 200 {
            // Medium length texts
            bufferingStrategy = .adaptive
            targetBufferLevel = 4
            prefillBufferCount = 5
            print("Using adaptive buffering for medium text (\(text.count) chars)")
        } else {
            // Short texts, prioritize latency
            bufferingStrategy = .fixed
            targetBufferLevel = 2
            prefillBufferCount = 3
            print("Using fixed buffering for short text (\(text.count) chars)")
        }

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
        playbackRate = 1.0
        bufferHealthHistory.removeAll()
        networkJitterMs = 0.0
    }

    // MARK: - Audio Processing

    /// Starts the scheduling process if not already running
    private func ensureSchedulingActive() {
        let wasScheduling = isSchedulingBuffers
        if !wasScheduling && isPlaying {
            isSchedulingBuffers = true
            audioProcessingQueue.async { [weak self] in
                self?.processAudioChunks()
            }
        }
    }

    /// Main audio processing loop - runs on audioProcessingQueue
    private func processAudioChunks() {
        guard isSchedulingBuffers else { return }

        // Get current buffer health
        audioChunksLock.lock()
        let pendingChunksCount = audioChunks.count
        let hasData = !audioChunks.isEmpty
        audioChunksLock.unlock()

        // Update buffer health history
        bufferHealthHistory.append(pendingChunksCount)
        if bufferHealthHistory.count > 20 {
            bufferHealthHistory.removeFirst()
        }

        // Adaptive rate adjustment based on buffer health
        if bufferingStrategy == .adaptive || bufferingStrategy == .aggressive {
            updatePlaybackRate(bufferCount: pendingChunksCount)
        }

        // Process chunk if we have one
        if hasData {
            audioChunksLock.lock()
            let chunk = audioChunks.removeFirst()
            audioChunksLock.unlock()

            if let sourceFormat = sourceFormat, let converter = converter {
                processAndScheduleAudioChunk(chunk, sourceFormat: sourceFormat, converter: converter)

                // Debug output every few seconds
                let now = Date()
                if now.timeIntervalSince(lastDebugPrintTime) > 5.0 {
                    // Calculate network jitter
                    if bufferHealthHistory.count >= 10 {
                        let bufferVariance = calculateBufferVariance()
                        networkJitterMs = bufferVariance * bufferDuration * 1000 // convert to ms
                    }

                    print("Buffer health: \(pendingChunksCount)/\(targetBufferLevel) chunks (\(bufferUnderrunCount) underruns, jitter: \(networkJitterMs)ms)")
                    lastDebugPrintTime = now
                }
            }
        } else {
            // No chunk available, check if we need more buffers scheduled
            if playerNode.isPlaying && needsScheduling {
                bufferUnderrunCount += 1
                print("⚠️ Buffer underrun detected! (\(bufferUnderrunCount) total)")
                needsScheduling = false

                // Stop engine to refill minimum buffer cache
                playerNode.stop()
                engine.stop()
                isSchedulingBuffers = false
                isPlaying = false

                return
            }
        }

        // Continue processing if we're still scheduling
        if isSchedulingBuffers {
            // Use short delay to prevent tight loop while allowing frequent scheduling
            let delayTime: TimeInterval
            if pendingChunksCount > targetBufferLevel {
                delayTime = 0.005 // Process quickly if we have lots of data
            } else if pendingChunksCount > 0 {
                delayTime = 0.01 // Normal processing
            } else {
                delayTime = 0.02 // Wait longer if no data to process
            }

            DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: .now() + delayTime) { [weak self] in
                self?.audioProcessingQueue.async {
                    self?.processAudioChunks()
                }
            }
        }
    }

    /// Calculate variance in buffer level (used to estimate jitter)
    private func calculateBufferVariance() -> Double {
        guard bufferHealthHistory.count > 1 else { return 0 }

        let average = Double(bufferHealthHistory.reduce(0, +)) / Double(bufferHealthHistory.count)
        let sumOfSquaredDifferences = bufferHealthHistory.reduce(0.0) { result, value in
            let difference = Double(value) - average
            return result + (difference * difference)
        }

        return sqrt(sumOfSquaredDifferences / Double(bufferHealthHistory.count))
    }

    /// Update playback rate based on buffer health
    private func updatePlaybackRate(bufferCount: Int) {
        if bufferCount < minBufferLevel {
            // Buffer getting too low - slow down playback slightly to let buffer refill
            let newRate = max(0.95, playbackRate - 0.01)
            if newRate != playbackRate {
                playbackRate = newRate
                // Apply rate change if significant
                if abs(playbackRate - 1.0) > 0.03 {
                    print("Adjusting playback rate to \(playbackRate) (buffer low: \(bufferCount))")
                }
            }
        } else if bufferCount > maxBufferLevel && bufferCount > targetBufferLevel * 2 {
            // Buffer growing too large - speed up playback slightly to catch up
            let newRate = min(1.05, playbackRate + 0.01)
            if newRate != playbackRate {
                playbackRate = newRate
                // Apply rate change if significant
                if abs(playbackRate - 1.0) > 0.03 {
                    print("Adjusting playback rate to \(playbackRate) (buffer high: \(bufferCount))")
                }
            }
        } else if bufferCount >= targetBufferLevel && bufferCount <= targetBufferLevel + 2 {
            // Buffer at healthy level - gradually return to normal rate
            if playbackRate != 1.0 {
                playbackRate = playbackRate + (1.0 - playbackRate) * 0.2
                if abs(playbackRate - 1.0) < 0.01 {
                    playbackRate = 1.0
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

        // Calculate output frame capacity based on sample rate ratio and playback rate
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
            let sampleTime = lastTime.sampleTime + AVAudioFramePosition(buffer.frameLength)
            schedulingTime = AVAudioTime(sampleTime: sampleTime, atRate: outputFormat!.sampleRate)
        } else {
            // First buffer; schedule with minimal delay to reduce latency
            let hostTime = mach_absolute_time() + UInt64(50_000_000) // 50ms initial delay
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
        totalScheduledFrames += UInt64(buffer.frameLength)
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        if !headerParsed {
            // Collect header bytes
            headerBuffer.append(data)

            if headerBuffer.count >= 44 {
                // We have enough bytes for the WAV header
                parseWAVHeader(headerBuffer)
                headerParsed = true

                // Start processing any remaining audio data after the header
                if headerBuffer.count > 44 {
                    let audioData = headerBuffer.suffix(from: 44)
                    enqueueAudioData(Data(audioData))
                }
            }
        } else {
            // Process incoming audio data
            enqueueAudioData(data)
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

        // Allow any remaining audio to play, but don't wait too long
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.isSchedulingBuffers = false
            self?.completionHandler?()
        }
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

        // Start scheduling if we have enough initial data
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

        // Create/update the output format - use hardware sample rate
        let hwSampleRate = AVAudioSession.sharedInstance().sampleRate
        outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: hwSampleRate, // Use hardware rate for best quality
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
