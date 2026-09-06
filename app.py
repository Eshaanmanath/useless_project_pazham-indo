from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from analyzer import process_puttu_and_banana

app = FastAPI(title="Puttu-Pazham Ratio Analyzer")

# Serve HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/analyze")
async def analyze_plate(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = process_puttu_and_banana(image_bytes)
    return result
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}
