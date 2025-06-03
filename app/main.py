from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse

import os
from app.whisper_sliding import whisper_sliding_inference
from app.postprocess import clean_transcription

SAVE_PATH = "data/qa.txt"
os.makedirs("data",exist_ok=True)
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
    # 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    os.remove(file_location)
    return {
        "cleaned_transcription": final_text
    }

@app.get("/get-transcription/", response_class=PlainTextResponse)
async def get_transcription():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    return PlainTextResponse("No transcription found. Please run POST /transcribe-audio/ first.", status_code=404)