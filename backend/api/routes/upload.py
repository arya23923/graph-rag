from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from datetime import datetime
from backend.config import config

router = APIRouter(prefix="/api/upload", tags=["Document Upload"])

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    file_path = config.UPLOAD_DIR / safe_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "success",
        "message": "Document uploaded successfully",
        "file_info": {
            "original_filename": file.filename,
            "saved_filename": safe_filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }
    }

@router.post("/process/{filename}")
async def process_document(filename: str):
    """Process uploaded document to extract text"""
    from backend.core.text_extractor import TextExtractor
    
    file_path = config.UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    extractor = TextExtractor()
    result = extractor.extract(file_path)
    
    if result["success"]:
        processed_path = config.PROCESSED_DIR / f"{Path(filename).stem}_extracted.txt"
        with open(processed_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        return {
            "status": "success",
            "extraction": result,
            "processed_file": str(processed_path)
        }
    else:
        raise HTTPException(status_code=500, detail=result["error"])