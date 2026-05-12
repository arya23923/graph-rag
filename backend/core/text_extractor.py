from pathlib import Path

class TextExtractor:
    @staticmethod
    def extract_from_txt(file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def extract(self, file_path: Path) -> dict:
        if not file_path.exists():
            return {"success": False, "error": "File not found"}
        
        try:
            text = self.extract_from_txt(file_path)
            return {
                "success": True,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}