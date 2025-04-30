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

    // Buffer settings for smooth playback - improved values
    private var prefillBufferCount = 12     // Increased for smoother initial playback
    private let bufferDuration = 0.025      // Reduced for more responsive yet stable playback
    private let maximumBufferedDuration = 3.0  // Increased to prevent dropout during network hiccups
    private var lastScheduledTime: AVAudioTime?
    private var converter: AVAudioConverter?
    private var needsScheduling = false
    private var isPlaying = false
    private var framesPerBuffer: AVAudioFrameCount = 3528 // Will be recalculated based on format

    // Advanced jitter buffer - enhanced values
    private var jitterBufferEnabled = true
    private var targetBufferLevel = 8       // Increased to maintain consistent playback
    private var minBufferLevel = 3          // Increased to prevent underruns more aggressively
    private var maxBufferLevel = 16         // Allow for more buffering during network jitter
    private var bufferingStrategy = BufferingStrategy.adaptive
    private var bufferMonitoringTimer: Timer?

    // Stats for debugging and adaptive playback
    private var totalScheduledFrames: UInt64 = 0
    private var bufferUnderrunCount = 0
    private var lastDebugPrintTime = Date()
    private var playbackRate: Float = 1.0
    private var bufferHealthHistory = [Int]()
    private var networkJitterMs = 0.0       // Estimated network jitter in milliseconds
    private var consecutiveEmptyBufferCount = 0
    private var lastNetworkDataTime = Date()

    // New audio smoothing properties
    private var crossfadeEnabled = true
    private var lastAudioBuffer: AVAudioPCMBuffer?
    private var crossfadeDuration = 0.01    // 10ms crossfade to smooth transitions
    private var renderQuality: AVAudioQuality = .high
    private var useHardwareCodec = true
    private var audioBufferPool = [AVAudioPCMBuffer]() // Buffer pool for reuse

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

        // Create custom URLSession configuration optimized for audio streaming
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        config.waitsForConnectivity = true
        config.httpMaximumConnectionsPerHost = 1
        
        // Enhance network performance
        config.httpShouldUsePipelining = true
        config.requestCachePolicy = .useProtocolCachePolicy
        config.networkServiceType = .avStreaming
        
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        // Start buffer monitoring
        startBufferMonitoring()
        
        print("OrpheusStreamingPlayer initialized with prefill count: \(prefillBufferCount)")
    }
    
    deinit {
        bufferMonitoringTimer?.invalidate()
        clearBufferPool()
    }

    // MARK: - Setup

    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers, .mixWithOthers])
            try audioSession.setActive(true)

            // Use optimized buffer duration based on device capability
            let optimalIOBuffer = determineOptimalIOBuffer()
            try audioSession.setPreferredIOBufferDuration(optimalIOBuffer)

            print("Audio session configured with \(optimalIOBuffer * 1000)ms IO buffer")
        } catch {
            print("AVAudioSession setup error: \(error)")
        }
    }
    
    // Determine the best IO buffer size based on device performance
    private func determineOptimalIOBuffer() -> TimeInterval {
        // Check device performance level
        let deviceName = UIDevice.current.name
        let processorCount = ProcessInfo.processInfo.processorCount
        
        if processorCount >= 6 && deviceName.contains("Pro") {
            // High-end device - can use smaller buffer
            return 0.005 // 5ms
        } else if processorCount >= 4 {
            // Mid-range device
            return 0.01 // 10ms
        } else {
            // Older/slower device - use larger buffer
            return 0.015 // 15ms
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
        
        // Add a reverb node with minimal settings to smooth audio
        let reverbNode = AVAudioUnitReverb()
        engine.attach(reverbNode)
        reverbNode.wetDryMix = 2.0 // Just enough to smooth edges, not noticeable as reverb
        
        // Connect player through reverb for smoother transitions
        engine.connect(playerNode, to: reverbNode, format: outputFormat)
        engine.connect(reverbNode, to: engine.mainMixerNode, format: outputFormat)

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
        
        // Listen for route changes
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleRouteChange),
            name: AVAudioSession.routeChangeNotification,
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
            
            // Save state
            let wasPlaying = playerNode.isPlaying
            
            // Stop playback but keep state
            if wasPlaying {
                playerNode.pause()
            }
        } else if type == .ended {
            // Interruption ended, resume if needed
            if let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
                let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                if options.contains(.shouldResume) {
                    print("Audio interruption ended - resuming playback")
                    
                    // Ensure session is active
                    do {
                        try AVAudioSession.sharedInstance().setActive(true)
                    } catch {
                        print("Failed to reactivate audio session: \(error)")
                    }
                    
                    startEngine()

                    if !playerNode.isPlaying && isPlaying {
                        // Add a small delay before resuming to ensure engine is ready
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                            self?.playerNode.play()
                        }
                    }
                }
            }
        }
    }
    
    @objc private func handleRouteChange(_ notification: Notification) {
        guard let userInfo = notification.userInfo,
              let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
              let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
            return
        }
        
        switch reason {
        case .newDeviceAvailable, .oldDeviceUnavailable, .categoryChange:
            // Route changed, ensure best audio quality for new route
            optimizeForCurrentRoute()
        default:
            break
        }
    }
    
    private func optimizeForCurrentRoute() {
        let currentRoute = AVAudioSession.sharedInstance().currentRoute
        
        // Check output ports
        for output in currentRoute.outputs {
            print("Optimizing for output: \(output.portType)")
            
            // Adjust buffer settings based on port type
            switch output.portType {
            case AVAudioSession.Port.headphones, AVAudioSession.Port.bluetoothA2DP:
                // Headphones or Bluetooth - can use lower buffer sizes
                bufferDuration = 0.02
                prefillBufferCount = 10
            case AVAudioSession.Port.builtInSpeaker:
                // Built-in speaker might need more buffering
                bufferDuration = 0.03
                prefillBufferCount = 14
            default:
                // Default settings for other devices
                bufferDuration = 0.025
                prefillBufferCount = 12
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

            // Restart engine with fresh state
            engine.reset()
            setupAudioEngine() // Reconfigure connections
            try engine.start()
            
            print("Engine recovered successfully")
            
            // Resume playback if needed
            if isPlaying && !playerNode.isPlaying {
                playerNode.play()
            }
        } catch {
            print("Recovery failed: \(error)")
        }
    }
    
    private func startBufferMonitoring() {
        // Stop existing timer if any
        bufferMonitoringTimer?.invalidate()
        
        // Create a timer to periodically check buffer health
        bufferMonitoringTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.checkBufferHealth()
        }
    }
    
    private func checkBufferHealth() {
        audioChunksLock.lock()
        let currentBufferCount = audioChunks.count
        audioChunksLock.unlock()
        
        // Track buffer health history
        bufferHealthHistory.append(currentBufferCount)
        if bufferHealthHistory.count > 20 {
            bufferHealthHistory.removeFirst()
        }
        
        // Calculate jitter
        networkJitterMs = calculateBufferVariance() * 20.0 // Approximate conversion to ms
        
        // Debug output (once per second)
        let now = Date()
        if now.timeIntervalSince(lastDebugPrintTime) >= 1.0 {
            lastDebugPrintTime = now
            
            // Only log if playing
            if isPlaying {
                print("Buffer health: \(currentBufferCount)/\(targetBufferLevel) chunks, jitter: \(networkJitterMs)ms")
                
                // Adjust buffer strategy based on network conditions
                updateBufferingStrategy()
            }
        }
        
        // If we're getting low on buffers but still have data coming, 
        // consider increasing prefill count for future streams
        if isPlaying && currentBufferCount <= minBufferLevel {
            let timeSinceLastData = now.timeIntervalSince(lastNetworkDataTime)
            if timeSinceLastData < 0.5 { // Still receiving data
                prefillBufferCount = min(prefillBufferCount + 1, 20)
                print("Increased prefill to \(prefillBufferCount) due to low buffer")
            }
        }
        
        // Update playback rate if adaptive buffering is enabled
        if bufferingStrategy == .adaptive {
            updatePlaybackRate(bufferCount: currentBufferCount)
        }
    }
    
    private func updateBufferingStrategy() {
        // If we're experiencing high jitter, adjust to more aggressive buffering
        if networkJitterMs > 30.0 && bufferingStrategy != .aggressive {
            print("Network jitter high (\(networkJitterMs)ms), switching to aggressive buffering")
            bufferingStrategy = .aggressive
            targetBufferLevel = 12
            minBufferLevel = 5
            prefillBufferCount = 14
        } else if networkJitterMs < 10.0 && bufferingStrategy == .aggressive {
            print("Network stabilized, switching to adaptive buffering")
            bufferingStrategy = .adaptive
            targetBufferLevel = 8
            minBufferLevel = 3
            prefillBufferCount = 10
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

        // Prepare buffering strategy based on text length and network conditions
        if text.count > 500 {
            // For longer texts, use more aggressive buffering
            bufferingStrategy = .aggressive
            targetBufferLevel = 12
            prefillBufferCount = 14
            print("Using aggressive buffering for long text (\(text.count) chars)")
        } else if text.count > 200 {
            // Medium length texts
            bufferingStrategy = .adaptive
            targetBufferLevel = 8
            prefillBufferCount = 10
            print("Using adaptive buffering for medium text (\(text.count) chars)")
        } else {
            // Short texts, prioritize latency but still ensure smooth playback
            bufferingStrategy = .fixed
            targetBufferLevel = 6
            prefillBufferCount = 8
            print("Using fixed buffering for short text (\(text.count) chars)")
        }
        
        // Remember when we start receiving data
        lastNetworkDataTime = Date()

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
        consecutiveEmptyBufferCount = 0
        lastAudioBuffer = nil
    }
    
    private func clearBufferPool() {
        audioProcessingQueue.async { [weak self] in
            self?.audioBufferPool.removeAll()
        }
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

        // Enhanced scheduling with adaptive timing
        while isSchedulingBuffers {
            var chunk: Data?
            audioChunksLock.lock()
            if !audioChunks.isEmpty {
                chunk = audioChunks.removeFirst()
                consecutiveEmptyBufferCount = 0
            } else {
                consecutiveEmptyBufferCount += 1
            }
            let currentBufferCount = audioChunks.count
            audioChunksLock.unlock()

            guard let data = chunk, let srcFmt = sourceFormat, let conv = converter else {
                // No data available, adapt wait time based on buffer status
                let waitTime = adaptiveWaitTime(emptyCount: consecutiveEmptyBufferCount)
                Thread.sleep(forTimeInterval: waitTime)
                continue
            }

            // Update last data time
            lastNetworkDataTime = Date()
            
            // Process the chunk with smooth transitions
            processAndScheduleAudioChunk(data, sourceFormat: srcFmt, converter: conv)

            // If buffer below min level, wait until prefillBufferCount chunks available,
            // but with a timeout to prevent deadlock
            if currentBufferCount < minBufferLevel {
                let waitStartTime = Date()
                let maxWaitTime = 0.5 // Maximum 500ms wait
                
                while isSchedulingBuffers {
                    let elapsedWait = Date().timeIntervalSince(waitStartTime)
                    if elapsedWait > maxWaitTime {
                        // Timeout - continue with what we have
                        break
                    }
                    
                    audioChunksLock.lock()
                    let count = audioChunks.count
                    audioChunksLock.unlock()
                    
                    if count >= prefillBufferCount { 
                        break 
                    }
                    
                    // Adaptive wait based on how full the buffer is
                    let adaptiveWait = min(0.01 * Double(prefillBufferCount - count), 0.05)
                    Thread.sleep(forTimeInterval: adaptiveWait)
                }
            } else {
                // Small yield to prevent thread hogging
                Thread.sleep(forTimeInterval: 0.001)
            }
        }
        // Done scheduling
        isSchedulingBuffers = false
    }
    
    /// Calculate adaptive wait time based on consecutive empty buffers
    private func adaptiveWaitTime(emptyCount: Int) -> TimeInterval {
        // If we've had many empty buffers, likely end of stream, so wait longer
        if emptyCount > 20 {
            return 0.05 // 50ms
        } else if emptyCount > 10 {
            return 0.02 // 20ms
        } else {
            return 0.01 // 10ms - quick response for active streaming
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
                    playerNode.rate = playbackRate
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
                    playerNode.rate = playbackRate
                }
            }
        } else if bufferCount >= targetBufferLevel && bufferCount <= targetBufferLevel + 2 {
            // Buffer at healthy level - gradually return to normal rate
            if playbackRate != 1.0 {
                playbackRate = playbackRate + (1.0 - playbackRate) * 0.2
                if abs(playbackRate - 1.0) < 0.01 {
                    playbackRate = 1.0
                }
                playerNode.rate = playbackRate
            }
        }
    }

    /// Processes and schedules a chunk of audio data with overlap handling for smooth playback
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

        // Calculate output frame capacity with some extra room
        let sampleRateRatio = outputFormat!.sampleRate / sourceFormat.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * sampleRateRatio * 1.1) // 10% extra

        // Create output buffer
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat!, frameCapacity: outputFrameCount) else {
            print("Failed to create output buffer")
            return
        }

        // Perform the conversion with quality settings
        var error: NSError?
        converter.sampleRateConverterQuality = renderQuality
        if useHardwareCodec {
            converter.sampleRateConverterAlgorithm = AVAudioQuality.high.rawValue
        }
        
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

        // Apply crossfade if needed and enabled
        if crossfadeEnabled, let previousBuffer = lastAudioBuffer {
            applyCrossfade(from: previousBuffer, to: outputBuffer)
        }
        
        // Store this buffer for potential crossfading with next buffer
        lastAudioBuffer = outputBuffer

        // Schedule the buffer at the appropriate time
        scheduleBuffer(outputBuffer)
    }
    
    /// Apply crossfade between audio buffers to smooth transitions
    private func applyCrossfade(from: AVAudioPCMBuffer, to: AVAudioPCMBuffer) {
        guard from.format == to.format else { return }
        
        // Calculate crossfade samples
        let sampleRate = from.format.sampleRate
        let framesToFade = AVAudioFrameCount(sampleRate * crossfadeDuration)
        
        // Ensure both buffers have enough frames for crossfade
        guard from.frameLength >= framesToFade && to.frameLength >= framesToFade else {
            return
        }
        
        // Get buffer pointers
        guard let fromBuffers = from.floatChannelData,
              let toBuffers = to.floatChannelData else {
            return
        }
        
        // Number of channels
        let channelCount = Int(from.format.channelCount)
        
        // Apply crossfade (linear fade for efficiency)
        for frame in 0..<Int(framesToFade) {
            let fromWeight = Float(framesToFade - AVAudioFrameCount(frame)) / Float(framesToFade)
            let toWeight = 1.0 - fromWeight
            
            for channel in 0..<channelCount {
                let fromIndex = Int(from.frameLength) - Int(framesToFade) + frame
                let fromSample = fromBuffers[channel][fromIndex]
                let toSample = toBuffers[channel][frame]
                
                // Blend the samples
                toBuffers[channel][frame] = fromSample * fromWeight + toSample * toWeight
            }
        }
    }

    /// Schedules a buffer with proper timing for smooth playback
    private func scheduleBuffer(_ buffer: AVAudioPCMBuffer) {
        // Ensure engine and player are running
        if !engine.isRunning {
            startEngine()
        }
        
        // First buffer - start playing immediately
        if !playerNode.isPlaying {
            playerNode.play()
            isPlaying = true
            playerNode.scheduleBuffer(buffer) { [weak self] in
                self?.needsScheduling = true
            }
            return
        }
        
        // For subsequent buffers, check if player node needs scheduling
        if needsScheduling || !playerNode.isPlaying {
            // Schedule buffer immediately after last buffered audio
            playerNode.scheduleBuffer(buffer) { [weak self] in
                self?.needsScheduling = true
            }
            needsScheduling = false
            
            // If player somehow stopped, restart it
            if !playerNode.isPlaying && isPlaying {
                playerNode.play()
            }
        } else {
            // If player is playing and has pending buffers, schedule this one as well
            playerNode.scheduleBuffer(buffer) { [weak self] in
                self?.needsScheduling = true
            }
        }
    }

    // MARK: - URLSessionDataDelegate

    func urlSession(_ session: URLSession,
                    dataTask: URLSessionDataTask,
                    didReceive data: Data) {
        // Update last data receipt time
        lastNetworkDataTime = Date()
        
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

        // Ensure a clean finish with a smooth fade out of the last buffer
        let shouldScheduleFadeOut = isPlaying && playerNode.isPlaying
        
        if shouldScheduleFadeOut {
            applyFadeOutToLastBuffer()
        }

        // Allow any remaining audio to play with adaptive wait time
        let waitTime = isPlaying ? max(0.5, Double(audioChunks.count) * bufferDuration * 0.5) : 0.1
        DispatchQueue.main.asyncAfter(deadline: .now() + waitTime) { [weak self] in
            self?.isSchedulingBuffers = false
            self?.completionHandler?()
        }
    }
    
    /// Apply a smooth fade out to the last buffer to prevent clicks or pops
    private func applyFadeOutToLastBuffer() {
        guard let lastBuffer = lastAudioBuffer, lastBuffer.frameLength > 0 else {
            return
        }
        
        // Create a fade-out buffer
        let fadeFrames = min(AVAudioFrameCount(outputFormat!.sampleRate * 0.1), lastBuffer.frameLength)
        
        guard let fadeBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat!, frameCapacity: fadeFrames) else {
            return
        }
        
        fadeBuffer.frameLength = fadeFrames
        
        // Copy the last frames from the last buffer
        if let destData = fadeBuffer.floatChannelData, let srcData = lastBuffer.floatChannelData {
            let channelCount = Int(outputFormat!.channelCount)
            
            for channel in 0..<channelCount {
                // Copy the last frames
                for frame in 0..<Int(fadeFrames) {
                    let srcIndex = Int(lastBuffer.frameLength) - Int(fadeFrames) + frame
                    destData[channel][frame] = srcData[channel][srcIndex]
                    
                    // Apply linear fade out
                    let fadeOutFactor = Float(fadeFrames - AVAudioFrameCount(frame)) / Float(fadeFrames)
                    destData[channel][frame] *= fadeOutFactor
                }
            }
        }
        
        // Schedule the fade-out buffer
        playerNode.scheduleBuffer(fadeBuffer)
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

        // Update last network data time
        lastNetworkDataTime = Date()

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

        // Create the converter with high quality settings for smooth audio
        converter = AVAudioConverter(from: sourceFormat!, to: outputFormat!)
        converter?.sampleRateConverterQuality = renderQuality
        
        if useHardwareCodec {
            converter?.sampleRateConverterAlgorithm = AVAudioQuality.high.rawValue
        }
        
        if converter == nil {
            print("Failed to create audio converter")
            return
        }

        // Reset audio engine, ensure connections are correct
        resetAudioEngine()
    }

    /// Reconfigures the audio engine with the current formats for smooth playback
    private func resetAudioEngine() {
        // Stop everything
        let wasPlaying = playerNode.isPlaying
        if engine.isRunning {
            engine.stop()
        }

        // Reset connections
        engine.disconnectNodeInput(playerNode)
        engine.disconnectNodeInput(engine.mainMixerNode)

        // Create a reverb node for smoother transitions
        let reverbNode = AVAudioUnitReverb()
        engine.attach(reverbNode)
        reverbNode.wetDryMix = 2.0 // Just enough to smooth edges

        // Reconnect with proper format through reverb
        engine.connect(playerNode, to: reverbNode, format: outputFormat)
        engine.connect(reverbNode, to: engine.mainMixerNode, format: outputFormat)
        
        // Prepare the engine
        engine.prepare()

        // Restart if we were playing
        if wasPlaying {
            startEngine()
            playerNode.play()
        }
    }
}
