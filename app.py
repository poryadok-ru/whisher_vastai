from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile

app = FastAPI()

model = WhisperModel(
    "large-v2",         
    device="cuda",
    compute_type="float16",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        segments, info = model.transcribe(
            tmp.name,
            language="ru",
            beam_size=5,
        )
        text = " ".join(seg.text for seg in segments)
    return {
        "text": text
    }
