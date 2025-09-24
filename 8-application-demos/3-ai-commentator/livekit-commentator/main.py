import asyncio
import logging
import os
from livekit import rtc
from collections import deque
import time
from dotenv import load_dotenv
import base64
import httpx
from queue import Queue
from threading import Thread, Lock
import pyaudio
from cartesia import Cartesia
import requests
import numpy as np
import cv2
from fastapi import FastAPI
import sys

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero
from livekit.agents.llm import ChatMessage, ChatImage
from livekit.agents import metrics, WorkerType

app = FastAPI()

# Configure Logging
logger = logging.getLogger("ai-commentator")
logger.setLevel(logging.INFO)
load_dotenv()

SPORT_CONTEXT = """You are an AI sports commentator specializing in basketball analysis. 
Your expertise includes:
- Understanding basketball gameplay and strategy
- Recognizing player movements and formations
- Identifying key moments in the game
- Providing engaging, real-time commentary

Keep your observations concise, natural, and focused on the most interesting aspects of the game.
Maintain an enthusiastic but professional tone, similar to professional sports broadcasters."""
sport_voice_id = "41534e16-2966-4c6b-9670-111411def906"
sport_emotional_controls = {
                            "speed": "fastest",
                            "emotion": ["positivity:highest", "surprise:highest"],
                        }
sport_question = """
            Provide the next micro-moment for this exciting game between the warriors and mavericks.
            • Must be a *single very short sentence* but make sure to create a suspenseful commentary.
            • Avoid reusing any previous sentence verbatim.
            • Do not mention player names or player numbers.
            • Keep it intense, but do not repeat yourself.
        """

MOVIE_CONTEXT = """You are a magical storyteller welcoming viewers into an enchanted forest world. Your tale begins with a peaceful woodland scene that sets the stage for adventure.

Your storytelling style:
- Paint vivid pictures of the forest's natural beauty
- Bring the gentle morning atmosphere to life
- Notice the small, delightful details of nature
- Build a sense of peaceful wonder
- Let the forest's magic unfold gradually

Remember:
- Keep each line brief (under 9 words)
- Start with the forest setting and atmosphere
- Introduce characters only when they appear
- Build anticipation through gentle observation
- Let the morning forest charm shine through

You're opening the door to a magical world - make the entrance enchanting!"""
movie_voice_id = "97f4b8fb-f2fe-444b-bb9a-c109783a857a"
movie_emotional_controls = {
    "speed": "normal",
    "emotion": ["positivity:highest", "surprise:highest",  "curiosity:highest"],
}
movie_question = """
            Provide the next micro-moment for this magical story of Bucks adventure in the enchanted forest.
            • Must be a *single short sentence* but make sure to create a suspenseful story.
            • Avoid reusing any previous sentence verbatim.
            • Keep it enchanting, but do not repeat yourself.
        """
# Add these global variables after the existing ones
MAX_QUEUE_SIZE = 10  # Adjust based on your requirements
audio_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
audio_lock = Lock()
is_speaking = False
# stop_flag = False

# Initialize audio components
# p = pyaudio.PyAudio()
# audio_stream = None
cartesia_client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
ws = cartesia_client.tts.websocket()

# Set up Cartesia output format
output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

conversation_history = []
MAX_HISTORY_LENGTH = 5  # Adjust this value as needed


async def audio_worker():
    global is_speaking, audio_stream, audio_source
    while True:
        try:
            queue_item = await audio_queue.get()
            if queue_item is None:
                audio_queue.task_done()  # Mark the None sentinel as done
                break

            text, video_timestamp = queue_item
            with audio_lock:
                is_speaking = True
                try:
                    print(f"Cartesia processing video {video_timestamp:.2f}s: {text}")
                    audio_chunks = []
                    for output in ws.send(
                        model_id="sonic-english",
                        transcript=text,
                        voice_id=voice_id,
                        output_format=output_format,
                        stream=True,
                        _experimental_voice_controls=emotional_controls
                    ):
                        buffer = output["audio"]
                        audio_chunks.append(buffer)

                    full_buffer = b''.join(audio_chunks)
                    audio_data = np.frombuffer(full_buffer, dtype=np.float32)
                    audio_data = (audio_data * 32767).astype(np.int16)

                    audio_queue.task_done()

                    if audio_source:
                        audio_frame = rtc.AudioFrame(
                            data=audio_data.tobytes(),
                            samples_per_channel=len(audio_data),
                            sample_rate=44100,
                            num_channels=1
                        )
                        await audio_source.capture_frame(audio_frame)
                finally:
                    is_speaking = False
                
            if not audio_queue.empty():
                # Remove skipped items from conversation history
                while not audio_queue.empty():
                    try:
                        skipped_text, _ = await audio_queue.get()
                        if skipped_text in conversation_history:
                            conversation_history.remove(skipped_text)
                        audio_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
        except Exception as e:
            print(f"Error in audio worker: {e}")
            audio_queue.task_done()

def format_conversation_history(history):
    """Convert history into OpenAI chat format"""
    formatted_history = [
        {
            "role": "system",
            "content": AGENT_CONTEXT
        }
    ]
    
    for message in history:
        formatted_history.append({
            "role": "assistant",
            "content": message
        })
    return formatted_history


def generate_commentary_with_api(frames_base64, conversation_history):
    global question

    API_URL = "https://api.aws.us-east-1.cerebrium.ai/v4/p-02037836/realtime-video-explainer/run"
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('CEREBRIUM_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "images": frames_base64,
        "question": question,
        "temperature": 0.5,
        "conversation_history": conversation_history
    }

    current_sentence = ""
    recent_sentences = set(msg["content"] for msg in conversation_history[-5:] if isinstance(msg, dict) and "content" in msg)
    try:
        start_time = time.time()
        with requests.post(API_URL, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()
            first_word_returned = False
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    current_sentence += chunk
                    # 2. Break output more aggressively to keep it short
                    words = current_sentence.split()
                    
                    if not first_word_returned and words:
                        print(f"Time to first word: {time.time() - start_time:.2f} seconds")
                        first_word_returned = True
                    
                    # If we have 8+ words or end with punctuation, consider it complete
                    if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?']):
                        trimmed = current_sentence.strip()
                        
                        # Filter out duplicates
                        if trimmed.lower() not in (sent.lower() for sent in recent_sentences):
                            yield trimmed
                            recent_sentences.add(trimmed)
                            print(f"Yielded sentence: {trimmed}")
                            
                        current_sentence = ""
            
            # If any leftover words remain, yield them (unless duplicate)
            trimmed = current_sentence.strip()
            if trimmed and trimmed.lower() not in (sent.lower() for sent in recent_sentences):
                yield trimmed
                
    except requests.RequestException as e:
        print(f"Error calling API: {e}")
        yield "Error generating commentary."

async def handle_video_track(track: rtc.Track):
        frames = []
        start_time = time.time()
        last_process_time = time.time()  # Add tracking for last processing time

        video_stream = rtc.VideoStream(track)
        # await asyncio.sleep(2)  # Wait for 2 seconds before starting video stream
        try:
            async for event in video_stream:
                current_time = time.time()                
                # Skip frame processing if audio queue is not empty
                if not audio_queue.empty():
                    continue
                
                # Collect frame every 100ms
                if (current_time - start_time) >= 0.1:
                    frames.append(event.frame)
                    start_time = current_time
                
                # Keep only latest frame
                if len(frames) > 1:
                    frames = frames[-1:]
                
                # Process frames no more frequently than every 2 seconds
                # and only if we're not currently speaking
                if (len(frames) >= 1 and 
                    (current_time - last_process_time) >= 2.0 and 
                    not is_speaking and 
                    audio_queue.empty()):
                    
                    logger.info(f"Processing frame at {current_time}")
                    await process_frames(frames)
                    frames = []
                    last_process_time = current_time
                    
                    # Clear any accumulated frames to prevent backlog
                    frames = []

        except Exception as e:
            logger.error(f"Error processing video stream: {e}")
        finally:
            await video_stream.aclose()

async def process_frames(frames):
        """Process the collected frames"""

        global conversation_history

        logger.info(f"Processing batch of {len(frames)} frames")
        print(f"Processing batch of {len(frames)} frames")
        # Convert frames to base64 encoded images
        encoded_frames = []

        for frame in frames:
                       
            # Convert frame to RGB24 format
            rgb_frame = frame.convert(rtc.video_frame.proto_video.VideoBufferType.RGB24)
            frame_data = rgb_frame.data
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            
            # Reshape using the frame's actual dimensions
            frame_array = frame_array.reshape((rgb_frame.height, rgb_frame.width, 3))
                
            # Additional check for overall brightness
            mean_value = np.mean(frame_array)
            if mean_value < 20:  # Increased threshold
                print(f"Skipping dark frame (mean value: {mean_value:.2f})")
                continue
            
            # No need for BGR to RGB conversion since we're already getting RGB24
            _, buffer = cv2.imencode('.jpg', frame_array)
            
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            encoded_frames.append(frame_b64)
        
        logger.info(f"Encoded frames: {len(encoded_frames)}")
        commentary_generator = generate_commentary_with_api(encoded_frames, format_conversation_history(conversation_history))
        timestamp = time.time()
        
        for sentence in commentary_generator:
            conversation_history.append(sentence)
            # Keep only the most recent entries
            if len(conversation_history) > MAX_HISTORY_LENGTH:
                conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
                
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                    logger.info("Dropped the oldest sentence to make room for the new one.")
                except asyncio.QueueEmpty:
                    pass  # This should not happen as we checked `audio_queue.full()`
        
            await audio_queue.put((sentence, timestamp))
        # await audio_queue.join()

async def entrypoint(ctx: JobContext):
    global audio_source, audio_track, AGENT_CONTEXT, voice_id, emotional_controls, question        

    logger.info(f"Room name: {ctx.room.name}")
    print(f"Room name: {ctx.room.name}")
    if ctx.room.name.lower().startswith('movie'):
        logger.info("Movie context")
        print("Movie context")
        AGENT_CONTEXT = MOVIE_CONTEXT
        voice_id = movie_voice_id
        emotional_controls = movie_emotional_controls
        question = movie_question
    else:  # Default to basketball/sports context
        logger.info("Default to basketball/sports context")
        print("Default to basketball/sports context")
        AGENT_CONTEXT = SPORT_CONTEXT
        voice_id = sport_voice_id
        emotional_controls = sport_emotional_controls
        question = sport_question
    logger.info(f"AGENT_CONTEXT: {AGENT_CONTEXT}")
    logger.info(f"emotion: {emotional_controls}")

    rtc.Room()

    audio_worker_task = asyncio.create_task(audio_worker())


    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(handle_video_track(track))

    @ctx.room.on("disconnected")
    def on_disconnected(ctx: JobContext):  # Remove async
        logger.info("Disconnected from the room.")
        print("Disconnected from the room.")
        
        async def cleanup():
            await ctx.room.disconnect()
            await audio_queue.put(None)  # Signal worker to stop
            if audio_worker_task:
                await audio_worker_task
            # Clear the queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except:
                    break
        
        # Create task for async cleanup
        asyncio.create_task(cleanup())


    await ctx.connect()
    logger.info("Connected to the room initialized.")
    # Create audio source and track
    audio_source = rtc.AudioSource(sample_rate=44100, num_channels=1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("ai-voice", audio_source)
    
    # Publish the audio track to the room
    await ctx.room.local_participant.publish_track(audio_track)

if __name__ == '__main__':
    if len(sys.argv) == 1:
            sys.argv.append('start')
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM, port=8600))
