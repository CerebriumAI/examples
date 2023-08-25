from typing import Optional
from pydantic import BaseModel

from seamless_communication.models.inference import Translator
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub import AudioSegment
import torchaudio
import torch
import os
from cerebrium import get_secret
import boto3

class Item(BaseModel):
    # Add your input parameters here
    task: str
    target_lang: str
    src_lang: str
    url: Optional[str] = None
    text: Optional[str] = None


translator = Translator(
    "seamlessM4T_large",
    "vocoder_36langs",
    torch.device("cuda:0"),
    torch.float16
)

def split_audio_with_max_duration(input_file, output_directory, min_silence_len=2500, silence_thresh=-60, max_chunk_duration=15000):

    sound = AudioSegment.from_wav(input_file)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Splitting on silence
    audio_chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # split for max_chunk_duration
    final_audio_chunks = []
    for chunk in audio_chunks:
        if len(chunk) > max_chunk_duration:
            num_subchunks = len(chunk) // max_chunk_duration + 1
            subchunk_size = len(chunk) // num_subchunks
            for i in range(num_subchunks):
                start_idx = i * subchunk_size
                end_idx = (i + 1) * subchunk_size
                subchunk = chunk[start_idx:end_idx]
                final_audio_chunks.append(subchunk)
        else:
            final_audio_chunks.append(chunk)

    # Export wav
    for i, chunk in enumerate(final_audio_chunks):
        output_file = f"{output_directory}/chunk{i}.wav"
        print("Exporting file", output_file)
        chunk.export(output_file, format="wav")

def convert_chunks_to_file(task, target_lang, src_lang, output_directory):
    segments = []

    for filename in sorted(os.listdir(output_directory)):
        segment_path = os.path.join(output_directory, filename)

        translated_text, wav, sr = translator.predict(
            input=segment_path,
            task_str=task,#'s2st',
            tgt_lang=target_lang,#'eng',
            src_lang=src_lang,#'spa',
        )
        print(translated_text)

        if task == 's2st':
            torchaudio.save(
                segment_path,
                wav[0].cpu(),
                sample_rate=sr,
            )

            segment = AudioSegment.from_file(segment_path)
            segments.append(segment)
        else:
            segments.append(str(translated_text))
            
            
    if task == 's2st':
        print('here')
        combined_audio = sum(segments)
        return combined_audio.export(f'{output_directory}audio.wav', format="wav")
    else:
        print(type(segments[0]))
        return ''.join(segments)
        

def download_file_from_url(logger, url: str, run_id: str):
    logger.info("Downloading file...")

    import requests

    dir_name = f"/persistent-storage/{run_id}"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(f"/persistent-storage/{run_id}/original.wav", "wb") as f:
            f.write(response.content)

        return f"/persistent-storage/{run_id}/original.wav"

    else:
        logger.info(response)
        raise Exception("Download failed")

def save_to_s3(file_url, run_id):
    s3 = boto3.client('s3',
                    aws_access_key_id=get_secret('aws-access-key'),
                    aws_secret_access_key=get_secret('aws-secret-key'))

    # Upload the file
    bucket = get_secret('meta-seamless-bucket')
    file = f"meta-seamless/{run_id}.wav"
    s3.upload_file(file_url, bucket, f"{run_id}.wav")
    return f"https://{bucket}.s3.amazonaws.com/{file}"

def predict(item, run_id, logger):
    item = Item(**item)
    
    ##1. Is this a STT or a TTS
    ##Task has to be s2st,t2st, s2t
    print(item.task)
    if item.task == 's2st' or item.task == 's2tt':
        filename = download_file_from_url(logger, item.url, run_id)
        split_audio_with_max_duration(filename, f"/persistent-storage/split_segments/{run_id}")
        result = convert_chunks_to_file(item.task, item.target_lang, item.src_lang, f"/persistent-storage/{run_id}/")

        if item.task == 's2tt':
            print('here')
            return {"translation": result}
            
    elif item.task == "t2st" or item.task == "t2tt":
        translated_text, wav, sr = translator.predict(
            item.text,
            item.task,
            tgt_lang=item.target_lang,
            src_lang=item.src_lang
        )

        if item.task == "t2tt":
            return {"translation": translated_text}

        torchaudio.save(f"/persistent-storage/{run_id}/audio.wav", wav, sample_rate=item.sr)
    else:
        return {"message": "Unknown task type. It has to be one of s2st, s2tt, t2st or t2tt"}

    #Save file to s3
    uploaded_url = save_to_s3(f"/persistent-storage/{run_id}/audio.wav", run_id)

    return {"message": "File successfully translated","file_url": uploaded_url} # return your results 
