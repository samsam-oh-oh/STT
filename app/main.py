from fastapi import FastAPI, UploadFile, File
import os
from app.whisper_sliding import whisper_sliding_inference
from app.postprocess import clean_transcription

app = FastAPI(
    title="Sliding Whisper API",
    description="슬라이딩 윈도우 기반 한국어 STT API",
    version="1.0"
)

@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"app/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    raw_text = whisper_sliding_inference(file_location)
    final_text = clean_transcription(raw_text)

    os.remove(file_location)
    return {
        "cleaned_transcription": final_text
    }