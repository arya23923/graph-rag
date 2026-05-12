import shutil
from pathlib import Path
from datetime import datetime
from backend.config import config
import magic

class FileHandler:
    
    @staticmethod
    def validate_file(file):
        """Validate uploaded file"""
        # Check extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            return {
                "valid": False,
                "error": f"File type {file_ext} not allowed. Allowed: {config.ALLOWED_EXTENSIONS}"
            }
        
        # Check size (content-length header might not be reliable, check actual size)
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Seek back to start
        
        if size > config.MAX_FILE_SIZE:
            return {
                "valid": False,
                "error": f"File too large. Max size: {config.MAX_FILE_SIZE / 1024 / 1024}MB"
            }
        
        # Check MIME type
        mime = magic.from_buffer(file.file.read(1024), mime=True)
        file.file.seek(0)
        
        allowed_mimes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        if mime not in allowed_mimes:
            return {
                "valid": False,
                "error": f"Invalid MIME type: {mime}"
            }
        
        return {"valid": True, "error": None}
    
    @staticmethod
    def save_upload(file):
        """Save uploaded file to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = config.UPLOAD_DIR / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "original_filename": file.filename,
            "saved_filename": safe_filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }