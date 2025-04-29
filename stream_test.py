#!/usr/bin/env python3
import argparse
import sys
import requests
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test streaming TTS endpoint")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--url", default="http://localhost:8000/api/tts/stream", help="Streaming TTS endpoint URL")
    parser.add_argument("--voice", default="Orpheus", help="Voice name")
    parser.add_argument("--use_cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("--samplerate", type=int, default=24000, help="Sample rate")
    args = parser.parse_args()

    payload = {"text": args.text, "voice": args.voice, "use_cuda": args.use_cuda}
    try:
        resp = requests.post(args.url, json=payload, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up audio stream
    try:
        stream = sd.RawOutputStream(samplerate=args.samplerate, channels=1, dtype='int16')
        stream.start()
    except Exception as e:
        print(f"Audio stream error: {e}", file=sys.stderr)
        sys.exit(1)

    header_len = 44
    buffer = b""
    data_started = False

    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buffer += chunk
        if not data_started:
            if len(buffer) < header_len:
                continue
            # Skip WAV header
            buffer = buffer[header_len:]
            data_started = True
        if buffer:
            try:
                stream.write(buffer)
            except Exception as e:
                print(f"Playback error: {e}", file=sys.stderr)
                break
            buffer = b""

    stream.stop()
    stream.close()
    print("Playback finished")

if __name__ == "__main__":
    main()
