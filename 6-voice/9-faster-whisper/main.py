import base64
from faster_whisper import WhisperModel
import time
from tempfile import NamedTemporaryFile
DOWNLOAD_ROOT = "/tmp/"  # Change this to /persistent-storage/ if you want to save files to the persistent storage

model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2", 
    device="cuda", 
    compute_type="int8_float16",
)

def decode_base64_to_mp3(base64_string, output_path):
    with open(output_path, "wb") as mp3_file:
        mp3_file.write(base64.b64decode(base64_string))


def run(base64_string: str, vad_enabled: bool = False, language: str = "en", word_timestamps: bool = False):
    
    start_time = time.time()
    audio_data = base64.b64decode(base64_string)

    with NamedTemporaryFile(suffix='.wav') as temp:
        temp.write(audio_data)
        temp.flush()
        segments, _ = model.transcribe(
            temp.name,
            beam_size=1,
            language=language,
            task="transcribe",
            word_timestamps=word_timestamps,
            vad_filter=vad_enabled,
            vad_parameters={"min_silence_duration_ms": 400} if vad_enabled else None,
        )
        print(time.time()-start_time)
        segments = list(segments)
        transcription_text = "".join(s.text for s in segments)
        print(time.time()-start_time)

    return transcription_text