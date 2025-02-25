from daily import *
import numpy as np
import queue
import threading
import time
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from cartesia import Cartesia
import os
import asyncio
import librosa
import soundfile as sf

cartesia_client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
ws = cartesia_client.tts.websocket()

output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}


class Translator(EventHandler):
    def __init__(self, room_url, model):
        print("Initializing Translator...")  # Debug print
        self.client = CallClient(event_handler = self)
        self.is_running = True
        self.message_sent = False
        self.queue = queue.Queue()
        self.room_url = room_url
        self.model = model
        self.audio_buffer = []
        self.processing_lock = threading.Lock()
        self.is_collecting = False

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

        # VAD settings
        self.CHANNELS = 1  # Daily.js sends mono audio
        self.SAMPLE_RATE = 48000  # Daily.js uses 48kHz
        self.SPEECH_THRESHOLD = 0.80
        self.SPEECH_THRESHOLD_MS = 500
        self.SILENCE_THRESHOLD_MS = 1000
        self.VAD_RESET_PERIOD_MS = 2000
        self.SENTENCE_END_SILENCE_MS = 600  # 1 second of silence to mark end of sentence
        self.MAX_SENTENCE_MS = 5000  # Added for the new on_audio_frame method

        
        # Initialize native VAD
        self.vad = Daily.create_native_vad(
            reset_period_ms=self.VAD_RESET_PERIOD_MS,
            sample_rate=self.SAMPLE_RATE,
            channels=self.CHANNELS
        )
        
        # Initialize microphone device
        self.mic_device = Daily.create_microphone_device(
            "model-mic",
            sample_rate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            non_blocking=True
        )
        
        # Speech detection state
        self.speaking_status = "NOT_SPEAKING"
        self.started_speaking_time = 0
        self.last_speaking_time = 0

        self.client.update_inputs({
            "camera": False,
            "microphone": {"isEnabled": True, "settings": {"deviceId": "model-mic"}}
        })

    def on_participant_left(self, participant, reason):
        if len(self.client.participant_counts()) <=2: ##count is before the user has left
            self.is_running = False

    def on_participant_joined(self, participant):
        if not participant["info"]['isLocal']:
            print(f"🔊 Joined {participant['info']}")
            self.client.set_audio_renderer(
                    participant["id"], callback=self.on_audio_frame, audio_source="microphone"
                )

    def on_audio_frame(self, participant, frame):
        try:
            current_time_ms = time.time() * 1000
            confidence = self.vad.analyze_frames(frame.audio_frames)
            
            if confidence > self.SPEECH_THRESHOLD:
                if self.speaking_status == "NOT_SPEAKING":
                    self.started_speaking_time = current_time_ms
                    self.is_collecting = True
                    self.audio_buffer = []
                    print(f"🎤 Started new recording at {time.strftime('%H:%M:%S')}")
                
                self.speaking_status = "SPEAKING"
                self.last_speaking_time = current_time_ms
                
                if self.is_collecting:
                    # More precise conversion from int16
                    frame_data = np.frombuffer(frame.audio_frames, dtype=np.int16)
                    # Normalize to [-1, 1] range with better precision
                    frame_float = frame_data.astype(np.float32) / 32767.0
                    
                    # Ensure mono and handle any DC offset
                    if len(frame_float.shape) > 1:
                        frame_float = frame_float.mean(axis=1)
                    
                    # Remove DC offset
                    frame_float = frame_float - np.mean(frame_float)
                    
                    # Apply subtle noise reduction
                    if len(frame_float) > 0:
                        # Simple noise gate
                        noise_gate = 0.01  # Adjust this threshold if needed
                        frame_float[abs(frame_float) < noise_gate] = 0
                    
                    self.audio_buffer.append(frame_float)
                    
                    # Check if we've hit the maximum sentence length
                    if current_time_ms - self.started_speaking_time > self.MAX_SENTENCE_MS:
                        print(f"⚠️ Max sentence length reached ({self.MAX_SENTENCE_MS/1000:.1f}s)")
                        self.speaking_status = "NOT_SPEAKING"
                        self.is_collecting = False
                        
                        if self.audio_buffer:
                            self._process_audio_buffer(current_time_ms)
            else:
                if self.speaking_status == "SPEAKING":
                    print(f"👂 Detected silence: {confidence:.2f}")
                
                diff_ms = current_time_ms - self.last_speaking_time
                if diff_ms > self.SENTENCE_END_SILENCE_MS and self.is_collecting:
                    print(f"✨ Ending recording after {diff_ms/1000:.1f}s of silence")
                    self.speaking_status = "NOT_SPEAKING"
                    self.is_collecting = False

                    if self.audio_buffer:
                        self._process_audio_buffer(current_time_ms)

        except Exception as e:
            print(f"Audio frame processing error: {e}")
            import traceback
            print(traceback.format_exc())

    def _process_queue(self):
        """Background thread for processing audio"""
        while self.is_running:
            try:
                if not self.queue.empty():
                    audio_data = self.queue.get()
                    print("Processing new audio segment...")
                    
                    # Create and run the async task
                    asyncio.run(self._process_audio(audio_data))
                    self.queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()

    async def _process_audio(self, audio_np):
        """Async function to process audio with the model"""
        try:
            # Debug original audio
            print(f"Original audio length: {len(audio_np)} samples ({len(audio_np)/self.SAMPLE_RATE:.2f} seconds)")
            print(f"Original audio stats - min: {audio_np.min():.3f}, max: {audio_np.max():.3f}, mean: {audio_np.mean():.3f}")

            # Create a timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # Resample directly using librosa
            audio_for_model = librosa.resample(audio_np, orig_sr=self.SAMPLE_RATE, target_sr=16000)
            
            # Only process if audio is long enough
            if len(audio_for_model)/16000 < 0.2:
                print("Audio segment too short, skipping processing")
                return

            # Save the processed audio for debugging
            processed_file = f"debug_processed_{timestamp}.wav"
            sf.write(processed_file, audio_for_model, 16000, 'FLOAT')
            print(f"Saved processed audio to: {processed_file}")

            full_response = ""
            async for response in self.model.generate(
                prompt="<|audio|>\n",
                temperature=0.7,
                max_tokens=100,
                audio=audio_for_model
            ):
                print(f"Model response: {response}")
                full_response += response.text  # Note: added .text here

            print(f"Full response: '{full_response}'")  # Debug: Print the complete response

            # if full_response:
            #     # Generate speech from the response
            #     audio_chunks = []
            #     for output in ws.send(
            #         model_id="sonic-english",
            #         transcript=full_response,
            #         voice_id='97f4b8fb-f2fe-444b-bb9a-c109783a857a',
            #         output_format=output_format,
            #         stream=True,
            #     ):
            #         buffer = output["audio"]
            #         audio_chunks.append(buffer)

            #     # Combine all audio chunks and write to mic
            #     full_buffer = b''.join(audio_chunks)
            #     self.mic_device.write_frames(full_buffer)

        except Exception as e:
            print(f"Error processing audio response: {e}")
            import traceback
            traceback.print_exc()

    def join(self, url):
        self.client.join(url)
        time.sleep(4)

    def generate_speech(text: str):
        audio_chunks = []
        for output in ws.send(
            model_id="sonic-english",
            transcript=text,
            voice_id='97f4b8fb-f2fe-444b-bb9a-c109783a857a',
            output_format=output_format,
            stream=True,
        ):
            buffer = output["audio"]
            audio_chunks.append(buffer)

        full_buffer = b''.join(audio_chunks)

    def isRunning(self):
        return self.is_running

    def _process_audio_buffer(self, current_time_ms):
        """Helper method to process the audio buffer"""
        try:
            # Concatenate all frames
            complete_audio = np.concatenate(self.audio_buffer)
            
            # Apply some audio cleanup
            # Normalize
            max_val = np.max(np.abs(complete_audio))
            if max_val > 0:
                complete_audio = complete_audio / max_val * 0.95  # Leave some headroom
            
            # Apply a subtle smoothing filter
            from scipy import signal
            b, a = signal.butter(2, 0.99)  # 2nd order Butterworth filter
            complete_audio = signal.filtfilt(b, a, complete_audio)
            
            duration = len(complete_audio) / 16000
            
            print(f"📝 Processing audio segment:")
            print(f"  - Duration: {duration:.2f}s")
            print(f"  - Samples: {len(complete_audio)}")
            print(f"  - Total frames: {len(self.audio_buffer)}")
            print(f"  - Audio range: [{complete_audio.min():.3f}, {complete_audio.max():.3f}]")
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            debug_file = f"debug_complete_{timestamp}.wav"
            sf.write(debug_file, complete_audio, 16000, 'FLOAT')
            print(f"  - Saved to: {debug_file}")
            
            self.queue.put(complete_audio)
            self.audio_buffer = []
            
        except Exception as e:
            print(f"Error processing audio buffer: {e}")
            import traceback
            print(traceback.format_exc())