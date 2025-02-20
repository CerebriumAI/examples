from daily import *
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import io
import queue
import threading
import time
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch.nn.functional as F
import libsql_experimental as libsql
from supabase import create_client
from serpapi import GoogleSearch
import torch

# import transformers
# import numpy as np
# import librosa

# pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_5-llama-3_1-8b', trust_remote_code=True)

class Translator(EventHandler):
  def __init__(self, room_url):
    self.client = CallClient(event_handler = self)
    self.is_running = True
    self.message_sent = False
    self.queue = queue.Queue()
    self.room_url = room_url
    self.mic_device = Daily.create_microphone_device(
        "model-mic",
        sample_rate=16000,
        channels=1,
        non_blocking=True
    )
    self.frame_count = 0

    # self.consecutive_detections = {}  # To track consecutive detections

    # Initialize Silero VAD
    self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                         model='silero_vad')
    self.vad_model.eval()
    
    # Initialize buffer for collecting audio frames
    self.audio_buffer = []
    self.SAMPLE_RATE = 16000  # Silero VAD expects 16kHz
    self.THRESHOLD = 0.5  # VAD threshold, adjust as needed

    self.client.update_inputs({
        "camera": False,
        "microphone": True
    })


  def on_participant_left(self, participant, reason):
    if len(self.client.participant_counts()) <=2: ##count is before the user has left
      self.is_running = False

  def on_participant_joined(self, participant):
    if not participant["info"]['isLocal']:
        self.client.set_audio_renderer(
                participant["id"], callback=self.on_audio_frame, audio_source="microphone"
            )

  def on_audio_frame(self, participant, frame):
    # Convert frame to torch tensor
    audio_array = np.frombuffer(frame, dtype=np.float32)
    audio_tensor = torch.from_numpy(audio_array)
    
    # Add to buffer
    self.audio_buffer.append(audio_tensor)
    
    # Process when buffer reaches ~1 second (adjust based on your frame size)
    if len(self.audio_buffer) >= 50:  # Assuming 20ms frames
        # Concatenate buffer
        audio_concat = torch.cat(self.audio_buffer)
        
        # Get speech probability
        speech_prob = self.vad_model(audio_concat, self.SAMPLE_RATE).item()
        
        if speech_prob >= self.THRESHOLD:
            print(f"Speech detected! Probability: {speech_prob:.2f}")
            # Process the audio
            self.queue.put(audio_concat.numpy().tobytes())
        
        # Clear buffer
        self.audio_buffer = []

  def process_audio(self):
    try:
        audio_data = self.queue.get()
        # Add your audio processing here (translation, etc.)
        
        # Send processed audio to other participants
        self.mic_device.write_frames(audio_data)
    except Exception as e:
        print(f"Error processing audio: {e}")
    finally:
        self.queue.task_done()

  def join(self, url):
     self.client.join(url)
     time.sleep(4)

  def isRunning(self):
    return self.is_running