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
    private var prefillBufferCount = 0    // Significantly increased for smoother playback
    private let bufferDuration = 0.15      // Larger chunks for more stability
    private let maximumBufferedDuration = 12.0  // Increased to ensure enough buffer
    private var lastScheduledTime: AVAudioTime?
    private var converter: AVAudioConverter?
    private var needsScheduling = false
    private var isPlaying = false
    private var framesPerBuffer: AVAudioFrameCount = 3528 // Will be recalculated based on format
    private var isPaused = false           // Track if playback is paused due to buffer underrun

    // Advanced jitter buffer
    private var jitterBufferEnabled = false  // Disabled jitter buffer
    private var targetBufferLevel = 0
    private var minBufferLevel = 0
    private var maxBufferLevel = 0
    private var bufferingStrategy = BufferingStrategy.aggressive
    private var criticalBufferLevel = 5
    private var refillTargetLevel = 5
    private var bufferConsumptionRate = 3   // Default higher consumption rate

    // Chunk aggregation to reduce processing overhead
    private var chunkAggregationEnabled = true
    private var aggregatedChunks = [Data]()
    private var maxAggregatedChunks = 3
    private var minChunkSize = 2048

    // Underrun protection
    private var silencePaddingEnabled = false   // Enable silence padding to avoid underruns
    private var silencePaddingFrames: AVAudioFrameCount = 8820  // About 200ms of silence at 44.1kHz
    private var lastUnderrunTime = Date.distantPast
    private var underrunProtectionActive = false
    private var silenceBuffer: AVAudioPCMBuffer?
    private var bufferStarvationCheckEnabled = true
    private var bufferStarvationCheckInterval: TimeInterval = 0.2  // Less frequent checks
    private var bufferHealthCheckTimer: Timer?

    // Stats for debugging and adaptive playback
    private var totalScheduledFrames: UInt64 = 0
    private var bufferUnderrunCount = 0
    private var lastDebugPrintTime = Date()
    private var playbackRate: Float = 1.0
    private var bufferHealthHistory = [Int]()
    private var networkJitterMs = 0.0       // Estimated network jitter in milliseconds
    private var isNetworkUnstable = false
    private var consecutiveLowBufferEvents = 0

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

        // Increase network buffer sizes
        config.httpShouldUsePipelining = true

        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        print("OrpheusStreamingPlayer initialized with prefill count: \(prefillBufferCount)")

        // Start buffer health check timer
        startBufferHealthMonitoring()
    }

    deinit {
        stopBufferHealthMonitoring()
    }

    // MARK: - Setup

    private func startBufferHealthMonitoring() {
        stopBufferHealthMonitoring() // Stop any existing timer

        bufferHealthCheckTimer = Timer.scheduledTimer(withTimeInterval: bufferStarvationCheckInterval, repeats: true) { [weak self] _ in
            self?.checkBufferHealth()
        }
    }

    private func stopBufferHealthMonitoring() {
        bufferHealthCheckTimer?.invalidate()
        bufferHealthCheckTimer = nil
    }

    private func checkBufferHealth() {
        guard bufferStarvationCheckEnabled && isPlaying else { return }

        audioChunksLock.lock()
        let currentBufferCount = audioChunks.count
        audioChunksLock.unlock()

        // Check if we need to pause due to low buffer
        if currentBufferCount < criticalBufferLevel && !isPaused {
            print("⚠️ Critical buffer level detected (\(currentBufferCount)). Pausing playback to refill buffers.")
            pausePlaybackForBuffering()
        }

        // Check if we can resume after pausing
        if isPaused && currentBufferCount >= refillTargetLevel {
            print("✅ Buffer refilled to \(currentBufferCount)/\(refillTargetLevel). Resuming playback.")
            resumePlaybackAfterBuffering()
        }

        // Adaptive buffer strategy adjustments
      if isNetworkUnstable && currentBufferCount < Int(Double(targetBufferLevel) * 0.5) && !isPaused {
            // Network is unstable and buffer is getting low
            increaseTempBufferTargets()
        }
    }

    private func increaseTempBufferTargets() {
        let originalTarget = targetBufferLevel

        targetBufferLevel = min(targetBufferLevel + 2, maxBufferLevel)
        criticalBufferLevel = min(criticalBufferLevel + 1, targetBufferLevel / 2)
        refillTargetLevel = min(refillTargetLevel + 2, targetBufferLevel - 2)

        if targetBufferLevel != originalTarget {
            print("🔄 Preemptively increasing buffer targets due to unstable network")
        }
    }

    private func pausePlaybackForBuffering() {
        // No-op: continue playback silently on buffer underrun
    }

    private func resumePlaybackAfterBuffering() {
        isPaused = false

        // Resume the player node if it was paused
        if !playerNode.isPlaying {
            playerNode.play()
        }

        // Resume the scheduling process
        if !isSchedulingBuffers {
            isSchedulingBuffers = true
            audioProcessingQueue.async { [weak self] in
                self?.processAudioChunks()
            }
        }

        // Reset the consecutive low buffer events counter since we recovered
        if consecutiveLowBufferEvents > 0 {
            consecutiveLowBufferEvents = 0
        }
    }

    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)

            // Increased buffer size for more stability
            let preferredIOBufferDuration = 0.02  // 20ms buffer for better stability
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

        // Create silence buffer for underrun protection
        createSilenceBuffer()
    }

    private func createSilenceBuffer() {
        guard let format = outputFormat else { return }

        silenceBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: silencePaddingFrames)
        guard let buffer = silenceBuffer else { return }

        buffer.frameLength = silencePaddingFrames

        // Fill with silence (zeros)
        for channel in 0..<Int(format.channelCount) {
            if let data = buffer.floatChannelData?[channel] {
                memset(data, 0, Int(silencePaddingFrames) * MemoryLayout<Float>.size)
            }
        }
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
        guard let url = URL(string: "http://34.71.2.239:5005/v1/audio/speech/stream") else {
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

        // Start buffer health monitoring
        startBufferHealthMonitoring()

        // Use fixed buffering strategy for all text lengths
        bufferingStrategy = .aggressive


        print("📊 Using fixed buffering for text (\(text.count) chars)")

        // Begin streaming
        dataTask = session.dataTask(with: request)
        dataTask?.resume()
    }

    func stop() {
        isPlaying = false
        isPaused = false
        playerNode.stop()
        engine.stop()
        dataTask?.cancel()
        stopBufferHealthMonitoring()
        reset()
    }

    private func reset() {
        audioChunksLock.lock()
        audioChunks.removeAll()
        aggregatedChunks.removeAll()
        audioChunksLock.unlock()

        headerBuffer.removeAll()
        headerParsed = false
        isSchedulingBuffers = false
        lastScheduledTime = nil
        totalScheduledFrames = 0
        bufferUnderrunCount = 0
        isPlaying = false
        isPaused = false
        lastDebugPrintTime = Date()
        playbackRate = 1.0
        bufferHealthHistory.removeAll() // Clear buffer history on reset
        isNetworkUnstable = false
        consecutiveLowBufferEvents = 0
        bufferConsumptionRate = 2
    }

    // MARK: - Audio Processing

    /// Starts the scheduling process if not already running
    private func ensureSchedulingActive() {
        let wasScheduling = isSchedulingBuffers
        if !wasScheduling && isPlaying && !isPaused {
            isSchedulingBuffers = true
            audioProcessingQueue.async { [weak self] in
                self?.processAudioChunks()
            }
        }
    }

    /// Main audio processing loop - runs on audioProcessingQueue
    private func processAudioChunks() {
        guard isSchedulingBuffers else { return }

        // Set initial chunk consumption rate based on buffer health
        adjustBufferConsumptionRate()

        // Enhanced scheduling with refill wait
        while isSchedulingBuffers {
            // First check if we're paused due to buffer starvation
            if isPaused {
                sleep(1) // Sleep and wait for buffer health checker to resume
                continue
            }

            // Process multiple chunks at once if enabled (adjusted by consumption rate)
            let chunksToProcess = processBufferChunks()

            // If we didn't get any chunks, schedule silence padding to avoid underrun
            if chunksToProcess.isEmpty {
                // No buffers; playback remains silent while waiting for buffers
                let sleepMicroseconds = UInt32(bufferDuration * 500_000) // half buffer duration
                usleep(sleepMicroseconds)
                continue
            }

            // Process the aggregated chunks (or single chunk)
            if let mergedData = mergeChunks(chunksToProcess),
               let srcFmt = sourceFormat,
               let conv = converter {

                // Process and schedule the audio chunk
                processAndScheduleAudioChunk(mergedData, sourceFormat: srcFmt, converter: conv)

                // Sleep briefly to pace buffer consumption
                let adjustmentFactor = 0.2 / Double(bufferConsumptionRate)
                let sleepDurationSeconds = bufferDuration * adjustmentFactor
                let sleepMicroseconds = UInt32(sleepDurationSeconds * 1000000)
                if sleepMicroseconds > 0 {
                    usleep(sleepMicroseconds)
                }

                // Re-adjust consumption rate based on current buffer health
                if bufferingStrategy == .adaptive {
                    adjustBufferConsumptionRate()
                }
            }
        }
        // Done scheduling
        isSchedulingBuffers = false
    }

    /// Process and retrieve buffer chunks based on current consumption rate
    private func processBufferChunks() -> [Data] {
        var chunks = [Data]()
        var currentBufferCount = 0

        audioChunksLock.lock()

        // Get consumption rate chunks, but don't exceed available chunks
        let numChunksToGet = min(bufferConsumptionRate, audioChunks.count)

        if numChunksToGet > 0 {
            for _ in 0..<numChunksToGet {
                if !audioChunks.isEmpty {
                    chunks.append(audioChunks.removeFirst())
                } else {
                    break
                }
            }
        }

        currentBufferCount = audioChunks.count
        audioChunksLock.unlock()

        // Monitor buffer health
        monitorBufferHealth(currentBufferCount)

        // If buffer is critically low, pause processing immediately
        if currentBufferCount < criticalBufferLevel && isPlaying && !chunks.isEmpty {
            print("🛑 Buffer level critically low during processing (\(currentBufferCount)/\(criticalBufferLevel)). Pausing processing.")
            pausePlaybackForBuffering()

            // Put chunks back at the front of the queue
            audioChunksLock.lock()
            audioChunks.insert(contentsOf: chunks, at: 0)
            audioChunksLock.unlock()

            return []
        }

        return chunks
    }

    /// Merge multiple chunks for more efficient processing
    private func mergeChunks(_ chunks: [Data]) -> Data? {
        guard !chunks.isEmpty else { return nil }

        // If only one chunk or aggregation disabled, return the first chunk
        if chunks.count == 1 || !chunkAggregationEnabled {
            return chunks[0]
        }

        // Merge chunks for more efficient processing
        let totalSize = chunks.reduce(0) { $0 + $1.count }
        var mergedData = Data(capacity: totalSize)

        for chunk in chunks {
            mergedData.append(chunk)
        }

        return mergedData
    }

    /// Adjust buffer consumption rate based on buffer health
    private func adjustBufferConsumptionRate() {
        audioChunksLock.lock()
        let currentBufferCount = audioChunks.count
        audioChunksLock.unlock()

        // Higher consumption rates to prevent buffer buildup
        if currentBufferCount > 100 {
            // Buffer is extremely large, consume much faster
            bufferConsumptionRate = 8
        } else if currentBufferCount > 50 {
            // Buffer is very large, consume faster
            bufferConsumptionRate = 6
        } else if currentBufferCount > 30 {
            // Buffer is large, consume faster
            bufferConsumptionRate = 4
        } else if currentBufferCount > targetBufferLevel {
            // Buffer above target, consume slightly faster
            bufferConsumptionRate = 3
        } else if currentBufferCount < minBufferLevel {
            // Buffer is low, consume slower
            bufferConsumptionRate = 1
        } else {
            // Buffer is at normal levels
            bufferConsumptionRate = 2
        }
    }

    private func monitorBufferHealth(_ currentBufferCount: Int) {
        // Calculate network stability metrics every second
        let now = Date()
        if now.timeIntervalSince(lastDebugPrintTime) >= 1.0 {
            lastDebugPrintTime = now

            // Debug output - simplified without jitter info
            print("📊 Buffer health: \(currentBufferCount)/\(targetBufferLevel) [min:\(minBufferLevel), crit:\(criticalBufferLevel), refill:\(refillTargetLevel)], Underruns: \(bufferUnderrunCount)")
        }

        // Add buffer trend analysis for predictive buffering
        bufferHealthHistory.append(currentBufferCount)
        if bufferHealthHistory.count > 20 {
            bufferHealthHistory.removeFirst()
        }
        if bufferHealthHistory.count >= 5 {
            let trend = bufferHealthHistory.last! - bufferHealthHistory.first!
            if trend < -5 {
                isNetworkUnstable = true
                print("🌐 Network instability detected: buffer trending down by \(-trend) chunks")
            } else {
                isNetworkUnstable = false
            }
        }
    }

    private func handlePotentialBufferUnderrun(_ currentBufferCount: Int) {
        bufferUnderrunCount += 1
        lastUnderrunTime = Date()
        print("⚠️ Buffer underrun detected! (count: \(bufferUnderrunCount))")

        // No longer pausing on buffer underrun
        underrunProtectionActive = false
    }

    private func calculateBufferVariance() -> Double {
        // Disabled jitter calculation
        return 0
    }

    /// Update playback rate based on buffer health
    private func updatePlaybackRate(bufferCount: Int) {
        // Fix playback rate to 1.0 for consistent playback
        playbackRate = 1.0
    }

    /// Processes and schedules a chunk of audio data
    private func processAndScheduleAudioChunk(_ chunk: Data, sourceFormat: AVAudioFormat, converter: AVAudioConverter) {
        bufferSemaphore.wait()
        defer { bufferSemaphore.signal() }

        // Skip processing if the player is paused
        if isPaused {
            return
        }

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
        let adjustedRatio = sampleRateRatio * Double(playbackRate)
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * adjustedRatio)

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

        // Apply volume ramping to prevent clicks/pops at buffer boundaries
        applyVolumeRamping(outputBuffer)

        // Schedule the buffer at the appropriate time
        scheduleBuffer(outputBuffer)
    }

    /// Schedule extra silence padding if needed during underrun conditions
    private func scheduleExtraPaddingIfNeeded() {
        // Disabled - don't add silence padding during underruns
        return
    }

    /// Apply volume ramping to prevent clicks/pops at buffer boundaries
    private func applyVolumeRamping(_ buffer: AVAudioPCMBuffer) {
        guard let floatData = buffer.floatChannelData else { return }

        let frameCount = Int(buffer.frameLength)
        let rampSamples = min(Int(buffer.frameLength) / 20, 50) // Much shorter, gentler ramp

        // Only apply if we have enough samples
        if frameCount < rampSamples * 2 { return }

        // For each channel
        for channel in 0..<Int(buffer.format.channelCount) {
            let channelData = floatData[channel]

            // Apply fade-in ramp at start (cubic curve for smoother transition)
            for i in 0..<rampSamples {
                let factor = Float(i) / Float(rampSamples)
                // Linear ramping for more natural sound
                channelData[i] *= factor
            }

            // Apply fade-out ramp at end (cubic curve for smoother transition)
            for i in 0..<rampSamples {
                let position = frameCount - rampSamples + i
                let factor = Float(rampSamples - i) / Float(rampSamples)
                // Linear ramping for more natural sound
                channelData[position] *= factor
            }
        }
    }

    /// Schedules a buffer with proper timing
    private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) {
        // Ensure engine and player are running
        if !engine.isRunning {
            startEngine()
        }

        if !playerNode.isPlaying && !isPaused {
            playerNode.play()
            isPlaying = true
        }

        // Schedule buffer immediately after last buffered audio
        playerNode.scheduleBuffer(buffer) { [weak self] in
            self?.needsScheduling = true
        }
    }

    // MARK: - Buffer Prefetching

    private let bufferManagementQueue = DispatchQueue(label: "com.orpheus.buffer-management", qos: .userInitiated)
    private var prefetchWorkItem: DispatchWorkItem?

    private func startPrefetchLoop() {
        prefetchWorkItem?.cancel()

        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self, self.isPlaying else { return }

            self.audioChunksLock.lock()
            let availableBuffers = self.audioChunks.count
            self.audioChunksLock.unlock()

            if availableBuffers < self.prefillBufferCount {
                self.fetchNextAudioSegment()
            }

            // Schedule next check based on buffer consumption rate
            let delay = self.calculateDynamicPrefetchDelay()
            self.bufferManagementQueue.asyncAfter(deadline: .now() + delay, execute: self.prefetchWorkItem!)
        }

        prefetchWorkItem = workItem
        bufferManagementQueue.async(execute: workItem)
    }

    private func calculateDynamicPrefetchDelay() -> TimeInterval {
        // Base delay on current buffer health and network conditions
        let baseDelay = max(0.1, min(2.0, bufferStarvationCheckInterval * 0.75))
        return TimeInterval(baseDelay)
    }

    private func fetchNextAudioSegment() {
        guard let url = URL(string: "http://34.71.2.239:5005/v1/audio/speech/stream") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Send async request
        let task = session.dataTask(with: request)
        task.resume()
    }

    // MARK: - Playback Control

    func play() {
        isPlaying = true
        startPrefetchLoop()  // Start async prefetching
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
            self?.stopBufferHealthMonitoring()
            self?.completionHandler?()
        }
    }

    // MARK: - Helpers

    /// Enqueues audio data for processing
    private func enqueueAudioData(_ data: Data) {
        // Skip empty data
        if data.isEmpty { return }

        // We'll aggregate very small chunks before adding to the queue if enabled
        if chunkAggregationEnabled && data.count < minChunkSize {
            aggregatedChunks.append(data)

            // Only process when we have enough aggregated or if this is first data
            if aggregatedChunks.count >= maxAggregatedChunks {
                let totalSize = aggregatedChunks.reduce(0) { $0 + $1.count }
                var mergedData = Data(capacity: totalSize)

                for chunk in aggregatedChunks {
                    mergedData.append(chunk)
                }

                // Add merged chunk to queue
                addChunkToQueue(mergedData)
                aggregatedChunks.removeAll()
            }
        } else {
            // Process any aggregated chunks first
            if !aggregatedChunks.isEmpty && chunkAggregationEnabled {
                let totalSize = aggregatedChunks.reduce(0) { $0 + $1.count }
                var mergedData = Data(capacity: totalSize)

                for chunk in aggregatedChunks {
                    mergedData.append(chunk)
                }

                // Add merged chunk to queue
                addChunkToQueue(mergedData)
                aggregatedChunks.removeAll()
            }

            // Add this chunk to queue
            addChunkToQueue(data)
        }
    }

    private func addChunkToQueue(_ data: Data) {
        // Add to queue
        audioChunksLock.lock()
        audioChunks.append(data)
        let pendingChunks = audioChunks.count
        audioChunksLock.unlock()

        // Start scheduling if we have enough initial data
        if !isPlaying && pendingChunks >= prefillBufferCount {
            isPlaying = true
            isPaused = false
            startEngine()
            ensureSchedulingActive()
            print("▶️ Starting playback with \(pendingChunks) chunks in buffer")
        } else if isPlaying && isPaused && pendingChunks >= refillTargetLevel {
            // Resume playback if we were paused and now have enough buffer
            resumePlaybackAfterBuffering()
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

        // Recalculate silence padding frames based on output sample rate
        silencePaddingFrames = AVAudioFrameCount(outputFormat!.sampleRate * 0.1) // 100ms of silence

        // Create the silence buffer with the current format
        createSilenceBuffer()

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
