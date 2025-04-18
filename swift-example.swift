/// A client capable of streaming WAV audio from an Orpheus FastAPI server and playing it as it arrives.
class OrpheusStreamingPlayerAdvanced: NSObject, URLSessionDataDelegate {
    // MARK: - Properties
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var session: URLSession!
    private var dataTask: URLSessionDataTask?
    private var headerBuffer = Data()
    private var headerParsed = false
    private var completionHandler: (() -> Void)?
    
    // WAV format parameters
    private var wavChannels: UInt16 = 1
    private var wavSampleRate: UInt32 = 24000
    private var wavBitsPerSample: UInt16 = 16
    
    // PCM buffer for the incoming WAV data
    private var audioBuffer = Data()
    
    // Flag to track if we've started playing
    private var hasStartedPlaying = false
    
    // For tracking audio scheduling
    private var scheduledBufferCount = 0
    private var playedBufferCount = 0

    /// Initializes the audio session, engine, and URLSession for streaming.
    override init() {
        super.init()
        
        // Configure audio session
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("AudioSession setup error: \(error)")
        }
        
        // Setup audio engine with playerNode
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        
        // Prepare the engine but don't start yet
        do {
            try engine.prepare()
        } catch {
            print("Failed to prepare engine: \(error)")
        }
        
        session = URLSession(configuration: .default, delegate: self, delegateQueue: .main)
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
        // Reset state
        headerParsed = false
        headerBuffer = Data()
        audioBuffer = Data()
        hasStartedPlaying = false
        scheduledBufferCount = 0
        playedBufferCount = 0
        completionHandler = completion
        
        // Close any existing task
        dataTask?.cancel()
        
        guard let url = URL(string: "http://35.205.197.251:5005/v1/audio/speech/stream") else {
            print("Invalid URL")
            return
        }
        
        // Prepare request
        let payload = ["input": text, "voice": voice, "response_format": "wav"]
        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            print("Failed to encode request")
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        
        // Start the audio engine
        if !engine.isRunning {
            do {
                try engine.start()
            } catch {
                print("Failed to start engine: \(error)")
                return
            }
        }
        
        // Start the player node
        if !playerNode.isPlaying {
            playerNode.play()
        }
        
        // Begin streaming
        dataTask = session.dataTask(with: request)
        dataTask?.resume()
    }

    func stop() {
        playerNode.stop()
        engine.stop()
        dataTask?.cancel()
    }
}

// MARK: - URLSessionDataDelegate
extension OrpheusStreamingPlayerAdvanced {
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        if !headerParsed {
            // Collect and process header
            headerBuffer.append(data)
            
            if headerBuffer.count >= 44 {
                // Process WAV header
                parseWAVHeader(headerBuffer)
                
                // Extract remaining audio data
                let audioData = headerBuffer.suffix(from: 44)
                if !audioData.isEmpty {
                    processAudioData(audioData)
                }
                
                headerParsed = true
            }
        } else {
            // Process audio data
            processAudioData(data)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        // Handle any remaining data
        if !audioBuffer.isEmpty && headerParsed {
            processRemainingAudioData()
        }
        
        // Report status
        if let error = error {
            print("Stream failed: \(error.localizedDescription)")
        } else {
            print("Stream completed successfully")
        }
        
        // Call completion handler
        DispatchQueue.main.async { [weak self] in
            self?.completionHandler?()
        }
    }

    // MARK: - Audio Processing
    private func parseWAVHeader(_ data: Data) {
        let header = [UInt8](data.prefix(44))
        
        // Extract WAV format info
        wavChannels = UInt16(header[22]) | (UInt16(header[23]) << 8)
        wavSampleRate = UInt32(header[24]) | (UInt32(header[25]) << 8) | 
                        (UInt32(header[26]) << 16) | (UInt32(header[27]) << 24)
        wavBitsPerSample = UInt16(header[34]) | (UInt16(header[35]) << 8)
        
        print("WAV format: \(wavSampleRate)Hz, \(wavChannels) channels, \(wavBitsPerSample) bits")
    }
    
    private func processAudioData(_ data: Data) {
        // Add new data to buffer
        audioBuffer.append(data)
        
        // For streaming, we want to process smaller chunks more frequently
        let bytesPerSample = Int(wavBitsPerSample) / 8
        let bytesPerFrame = bytesPerSample * Int(wavChannels)
        
        // Process chunks of 1/10th second of audio for smoother playback
        let bytesPerChunk = Int(wavSampleRate) / 10 * bytesPerFrame
        
        // Process complete chunks
        while audioBuffer.count >= bytesPerChunk {
            // Get a complete chunk
            let chunk = audioBuffer.prefix(bytesPerChunk)
            audioBuffer = audioBuffer.suffix(from: bytesPerChunk)
            
            // Schedule this chunk for playback
            scheduleAudioChunk(chunk)
        }
    }
    
    private func processRemainingAudioData() {
        // Get bytesPerFrame to ensure we only process complete frames
        let bytesPerSample = Int(wavBitsPerSample) / 8
        let bytesPerFrame = bytesPerSample * Int(wavChannels)
        
        // Calculate complete frames
        let completeFrameCount = audioBuffer.count / bytesPerFrame
        let completeDataSize = completeFrameCount * bytesPerFrame
        
        if completeDataSize > 0 {
            let completeData = audioBuffer.prefix(completeDataSize)
            scheduleAudioChunk(completeData)
        }
    }
    
    private func scheduleAudioChunk(_ chunkData: Data) {
        // Create the audio format based on WAV header
        let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(wavSampleRate),
            channels: AVAudioChannelCount(wavChannels),
            interleaved: true
        )!
        
        // Calculate frame count
        let bytesPerFrame = Int(format.streamDescription.pointee.mBytesPerFrame)
        let frameCount = UInt32(chunkData.count / bytesPerFrame)
        
        guard frameCount > 0 else { return }
        
        // Create buffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            print("Failed to create audio buffer")
            return
        }
        
        buffer.frameLength = frameCount
        
        // Copy data to buffer
        chunkData.withUnsafeBytes { ptr in
            if let src = ptr.baseAddress,
               let dst = buffer.audioBufferList.pointee.mBuffers.mData {
                memcpy(dst, src, Int(buffer.audioBufferList.pointee.mBuffers.mDataByteSize))
            }
        }
        
        // Track this buffer
        let bufferIndex = scheduledBufferCount
        scheduledBufferCount += 1
        
        // Schedule buffer
        playerNode.scheduleBuffer(buffer) { [weak self] in
            guard let self = self else { return }
            self.playedBufferCount += 1
            
            // Debug playback progress
            if bufferIndex % 10 == 0 {
                print("Played buffer \(bufferIndex) of \(self.scheduledBufferCount)")
            }
            
            if !self.hasStartedPlaying {
                self.hasStartedPlaying = true
                print("First audio chunk playback started")
            }
        }
    }
}

// Test with fixed 44.1kHz format
let testFormat = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
