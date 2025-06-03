import os
import tempfile
import soundfile as sf
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv()
HF_ENDPOINT = os.getenv("HF_ENDPOINT")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def whisper_sliding_inference(audio_path: str, window_sec=20.0, stride_sec=19.5):
    waveform, sr = sf.read(audio_path)  # WAV 로드
    total_samples = waveform.shape[0]
    window_size = int(sr * window_sec)
    stride_size = int(sr * stride_sec)

    texts = []
    for start in range(0, total_samples, stride_size):
        end = start + window_size
        chunk = waveform[start:end]
        if len(chunk) < sr * 1:  # 1초 미만은 무시
            break

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, chunk, sr)
            text = inference_with_hf(tmp_wav.name)
            if text:
                texts.append(text)
            os.remove(tmp_wav.name)

    return " ".join(texts)

def inference_with_hf(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_data = f.read()
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "audio/wav"
    }
    response = requests.post(HF_ENDPOINT, headers=headers, data=audio_data)
    if response.status_code == 200:
        return response.json().get("text", "")
    return ""