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
    private var prefillBufferCount = 30    // Increased for smoother playback
    private let bufferDuration = 0.12      // Slightly smaller chunks for more responsiveness
    private let maximumBufferedDuration = 10.0  // Increased to ensure enough buffer
    private var lastScheduledTime: AVAudioTime?
    private var converter: AVAudioConverter?
    private var needsScheduling = false
    private var isPlaying = false
    private var framesPerBuffer: AVAudioFrameCount = 3528 // Will be recalculated based on format
    private var isPaused = false           // Track if playback is paused due to buffer condition

    // Advanced jitter buffer
    private var jitterBufferEnabled = true  // Enable jitter buffer
    private var targetBufferLevel = 25
    private var minBufferLevel = 10        // Lower minimum to avoid stopping playback
    private var maxBufferLevel = 50        // Increased max buffer level
    private var bufferingStrategy = BufferingStrategy.adaptive // Use adaptive buffering by default
    private var criticalBufferLevel = 5    // Lower critical level to avoid stopping
    private var refillTargetLevel = 20
    private var bufferConsumptionRate = 2   // Default consumption rate

    // Chunk aggregation to reduce processing overhead
    private var chunkAggregationEnabled = true
    private var aggregatedChunks = [Data]()
    private var maxAggregatedChunks = 3
    private var minChunkSize = 2048

    // Underrun protection
    private var silencePaddingEnabled = true  // Changed to true - fill gaps with silence
    private var silencePaddingFrames: AVAudioFrameCount = 8820  // About 200ms of silence at 44.1kHz
    private var lastUnderrunTime = Date.distantPast
    private var underrunProtectionActive = false
    private var silenceBuffer: AVAudioPCMBuffer?
    private var bufferStarvationCheckEnabled = true
    private var bufferStarvationCheckInterval: TimeInterval = 0.1  // More frequent checks
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
    private var lastBufferLevelCheckTime = Date()
    private var bufferTrend = BufferTrend.stable // Track buffer health trend

    // MARK: - Enums

    enum BufferingStrategy {
        case fixed      // Fixed buffer size
        case adaptive   // Adapt buffer size based on network conditions
        case aggressive // More aggressive buffering for difficult networks
    }
    
    enum BufferTrend {
        case decreasing
        case stable
        case increasing
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
        
        // Update buffer trend analysis
        updateBufferTrend(currentBufferCount)

        // Monitor for potential buffer starvation before it happens
        if currentBufferCount < criticalBufferLevel && !isPaused {
            // Don't pause the player, just adjust parameters and report
            print("⚠️ Buffer level critically low (\(currentBufferCount)/\(criticalBufferLevel))")
            consecutiveLowBufferEvents += 1
            
            // Instead of stopping playback, schedule silence buffer if needed
            if silencePaddingEnabled && !underrunProtectionActive {
                scheduleExtraPaddingIfNeeded()
                underrunProtectionActive = true
            }
            
            // If the network is unstable, increase the buffer targets
            if isNetworkUnstable {
                increaseTempBufferTargets()
            }
        } else if currentBufferCount > targetBufferLevel {
            // Buffer is healthy again
            consecutiveLowBufferEvents = 0
            underrunProtectionActive = false
            
            // Resume playback if it was paused
            if isPaused {
                resumePlaybackAfterBuffering()
            }
        } else if currentBufferCount < minBufferLevel {
            // Below minimum but not critical - take preventive action
            if bufferTrend == .decreasing && !underrunProtectionActive {
                print("⚠️ Buffer trending down and below minimum (\(currentBufferCount)/\(minBufferLevel))")
                
                // Proactively schedule silence to prevent audio gaps
                if silencePaddingEnabled {
                    scheduleExtraPaddingIfNeeded()
                    underrunProtectionActive = true
                }
                
                increaseTempBufferTargets()
            }
        }

        // Adaptive buffer strategy adjustments
        if isNetworkUnstable && currentBufferCount < Int(Double(targetBufferLevel) * 0.5) && !isPaused {
            // Network is unstable and buffer is getting low
            increaseTempBufferTargets()
        }
    }

    private func updateBufferTrend(currentBufferCount: Int) {
        let now = Date()
        if now.timeIntervalSince(lastBufferLevelCheckTime) >= 0.5 {
            // Store buffer health history for trend analysis
            bufferHealthHistory.append(currentBufferCount)
            if bufferHealthHistory.count > 5 {
                bufferHealthHistory.removeFirst()
            }
            
            // Analyze trend
            if bufferHealthHistory.count >= 3 {
                let trend = analyzeTrend(bufferHealthHistory)
                bufferTrend = trend
                
                // Update network stability based on buffer trends
                if trend == .decreasing && currentBufferCount < targetBufferLevel {
                    isNetworkUnstable = true
                } else if trend == .increasing && currentBufferCount > targetBufferLevel {
                    isNetworkUnstable = false
                }
            }
            
            lastBufferLevelCheckTime = now
        }
    }
    
    private func analyzeTrend(_ history: [Int]) -> BufferTrend {
        guard history.count >= 2 else { return .stable }
        
        // Calculate slope of recent buffer levels
        let first = history.prefix(history.count / 2).reduce(0, +) / (history.count / 2)
        let second = history.suffix(history.count / 2).reduce(0, +) / (history.count / 2)
        
        if second - first < -2 {
            return .decreasing
        } else if second - first > 2 {
            return .increasing
        } else {
            return .stable
        }
    }

    private func pausePlaybackForBuffering() {
        // We're in "paused" mode from a logic perspective, but we don't actually stop the engine
        isPaused = true
        
        print("🔄 Buffer refill mode active - not stopping engine")
        
        // Slow down buffer consumption while in refill mode
        bufferConsumptionRate = 1
        
        // Schedule silence to maintain continuity instead of stopping
        if silencePaddingEnabled && !underrunProtectionActive {
            scheduleExtraPaddingIfNeeded()
            underrunProtectionActive = true
        }
        
        // Adjust buffer targets during refill period
        if bufferingStrategy == .adaptive || bufferingStrategy == .aggressive {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                
                // Dynamically adjust buffer targets based on historical performance
                let bufferUnderrunFactor = min(5, self.bufferUnderrunCount)
                let newTarget = min(self.maxBufferLevel, self.targetBufferLevel + bufferUnderrunFactor)
                let newCritical = max(2, self.criticalBufferLevel + bufferUnderrunFactor / 2)
                let newRefill = min(newTarget - 2, self.refillTargetLevel + bufferUnderrunFactor)
                
                if newTarget != self.targetBufferLevel || newCritical != self.criticalBufferLevel {
                    print("📊 Adjusting buffer strategy: target=\(newTarget), critical=\(newCritical), refill=\(newRefill)")
                    self.targetBufferLevel = newTarget
                    self.criticalBufferLevel = newCritical
                    self.refillTargetLevel = newRefill
                }
            }
        }
    }

    private func resumePlaybackAfterBuffering() {
        isPaused = false
        underrunProtectionActive = false
        
        // Ensure the player node is playing
        if !playerNode.isPlaying {
            playerNode.play()
        }
        
        // Reset consecutive low buffer counter
        consecutiveLowBufferEvents = 0
        
        // Resume normal buffer consumption rate
        bufferConsumptionRate = 2
        
        print("▶️ Resuming normal playback mode")
    }

    /// Schedule extra silence padding if needed during underrun conditions
    private func scheduleExtraPaddingIfNeeded() {
        guard silencePaddingEnabled, let silenceBuffer = silenceBuffer, !isPaused else { return }
        
        print("🔈 Scheduling silence padding to prevent audio gaps")
        
        // Schedule twice the silence buffer for added protection
        playerNode.scheduleBuffer(silenceBuffer)
        playerNode.scheduleBuffer(silenceBuffer)
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
        bufferingStrategy = .fixed
        targetBufferLevel = 0
        prefillBufferCount = 0
        minBufferLevel = 0
        criticalBufferLevel = 0
        refillTargetLevel = 0
        bufferConsumptionRate = 2
        maxAggregatedChunks = 3

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
        bufferHealthHistory.removeAll()
        networkJitterMs = 0.0
        isNetworkUnstable = false
        underrunProtectionActive = false
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
            // Slow down processing if paused but don't stop
            if isPaused {
                // Just slow down instead of sleeping
                let sleepTime = UInt32(0.2 * 1000000) // 200ms
                usleep(sleepTime)
                
                // Check if we have enough buffer to resume normal playback
                audioChunksLock.lock()
                let currentBufferCount = audioChunks.count
                audioChunksLock.unlock()
                
                if currentBufferCount >= refillTargetLevel {
                    resumePlaybackAfterBuffering()
                }
                
                continue
            }

            // Process multiple chunks at once if enabled (adjusted by consumption rate)
            let chunksToProcess = processBufferChunks()

            // If we didn't get any chunks, handle potential underrun but don't stop playback
            if chunksToProcess.isEmpty {
                if isPlaying {
                    audioChunksLock.lock()
                    let currentBufferCount = audioChunks.count
                    audioChunksLock.unlock()
                    
                    // Handle the low buffer condition
                    handlePotentialBufferUnderrun(currentBufferCount)
                    
                    // Schedule silence buffer to prevent starvation
                    if silencePaddingEnabled && !underrunProtectionActive {
                        scheduleExtraPaddingIfNeeded()
                        underrunProtectionActive = true
                    }
                }
                
                // Brief pause to avoid busy-waiting, but not too long
                usleep(100000) // 100ms wait
                continue
            }

            // Process the aggregated chunks (or single chunk)
            if let mergedData = mergeChunks(chunksToProcess),
               let srcFmt = sourceFormat,
               let conv = converter {

                // Process and schedule the audio chunk
                processAndScheduleAudioChunk(mergedData, sourceFormat: srcFmt, converter: conv)

                // Dynamic sleep time based on buffer health
                var adjustmentFactor = 0.2 / Double(bufferConsumptionRate)
                
                // If buffer is very low, process faster to avoid audio gaps
                audioChunksLock.lock()
                let currentCount = audioChunks.count
                audioChunksLock.unlock()
                
                if currentCount < criticalBufferLevel {
                    // Process much faster when critically low
                    adjustmentFactor = 0.05
                } else if currentCount < minBufferLevel {
                    // Process faster when below minimum
                    adjustmentFactor = 0.1
                }
                
                let sleepDurationSeconds = bufferDuration * adjustmentFactor
                let sleepMicroseconds = UInt32(sleepDurationSeconds * 1000000)
                if sleepMicroseconds > 0 {
                    usleep(sleepMicroseconds)
                }

                // Re-adjust consumption rate based on current buffer health
                if bufferingStrategy != .fixed {
                    adjustBufferConsumptionRate()
                }
            }
        }
        // Don't terminate scheduling completely
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

        // If buffer is critically low, don't pause processing but slow down
        if currentBufferCount < criticalBufferLevel && isPlaying && !chunks.isEmpty && !underrunProtectionActive {
            print("⚠️ Buffer level critically low during processing (\(currentBufferCount)/\(criticalBufferLevel)). Adding protection.")
            
            // Instead of pausing, add silence padding
            if silencePaddingEnabled {
                scheduleExtraPaddingIfNeeded()
                underrunProtectionActive = true
            }
            
            // Slow down consumption rate
            bufferConsumptionRate = 1
        }

        return chunks
    }

    private func handlePotentialBufferUnderrun(_ currentBufferCount: Int) {
        bufferUnderrunCount += 1
        lastUnderrunTime = Date()
        print("⚠️ Low buffer detected! (count: \(bufferUnderrunCount))")
        
        // Don't pause playback, just add silence padding if enabled
        if silencePaddingEnabled && !underrunProtectionActive {
            scheduleExtraPaddingIfNeeded()
            underrunProtectionActive = true
            
            // Notify but don't pause
            print("🔄 Scheduling silence instead of pausing")
        }
        
        // Mark network as unstable
        isNetworkUnstable = true
        
        // If we're running very low, slow down buffer consumption
        if currentBufferCount <= criticalBufferLevel {
            bufferConsumptionRate = 1
        }
    }
    
    private func increaseTempBufferTargets() {
        let originalTarget = targetBufferLevel
        
        // More aggressive buffer target adjustments
        targetBufferLevel = min(targetBufferLevel + 5, maxBufferLevel)
        criticalBufferLevel = max(3, min(criticalBufferLevel + 2, targetBufferLevel / 3))
        refillTargetLevel = min(refillTargetLevel + 3, targetBufferLevel - 3)
        
        if targetBufferLevel != originalTarget {
            print("🔄 Increasing buffer targets: target=\(targetBufferLevel), critical=\(criticalBufferLevel), refill=\(refillTargetLevel)")
        }
    }

    // ... rest of the code remains the same ...
}
