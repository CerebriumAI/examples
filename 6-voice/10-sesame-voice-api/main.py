from generator import load_csm_1b, Segment
import torchaudio
import torch
from huggingface_hub import hf_hub_download
import os
import base64

# This device selection lets our code work on any Cerebrium hardware
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load the model - this happens once when the service starts
# The model will stay loaded in memory for faster inference
generator = load_csm_1b(device=device)

# These example conversations give the model context for how to speak
# The model will mimic the speaking style in these samples
speakers = [0, 1]  # Speaker 0 and Speaker 1
transcripts = [
    (
        "like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a"
    ),
    (
        "like a super Mario level. Like it's very like high detail. And like, once you get "
        "into the park, it just like, everything looks like a computer game and they have all "
        "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
        "will have like a question block. And if you like, you know, punch it, a coin will "
        "come out. So like everyone, when they come into the park, they get like this little "
        "bracelet and then you can go punching question blocks around."
    ),
]

# Download the audio samples that accompany the transcripts
# With Cerebrium, we use persistent storage to cache these files
audio_paths = [
    hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_a.wav",
    ),
    hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="prompts/conversational_b.wav",
    ),
]


def _load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """Helper function to load and resample audio files"""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)  # Remove channel dimension
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor


# This is the function Cerebrium will call when we hit our endpoint
def generate_audio(text: str):
    """
    Generate conversational speech from text, using the CSM-1B model.

    Args:
        text (str): The text to convert to speech

    Returns:
        dict: Contains base64-encoded audio data and format information
    """
    # Create context segments from our example conversations
    segments = [
        Segment(
            text=transcript,
            speaker=speaker,
            audio=_load_prompt_audio(audio_path, generator.sample_rate),
        )
        for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
    ]

    # Generate audio with speaker 1's voice (you can change to 0 if preferred)
    audio = generator.generate(
        text=text,
        speaker=1,
        context=segments,
        max_audio_length_ms=10_000,  # Limit to 10 seconds
        temperature=0.9,  # Controls randomness - higher = more variation
    )

    # Save to temporary WAV file, read it, and convert to base64
    torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
    with open("audio.wav", "rb") as f:
        wav_data = f.read()
    os.remove("audio.wav")  # Clean up the temporary file
    encoded_data = base64.b64encode(wav_data).decode("utf-8")

    return {"audio_data": encoded_data, "format": "wav", "encoding": "base64"}
