import os
import sys
import json
import fitz  # PyMuPDF
import re
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 on Windows to avoid UnicodeEncodeErrors when printing unicode paths
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent  # 'rag_project'
BASE_DIR = PROJECT_DIR.parent  # 'c:/Users/shank/Downloads/For Rag'
SOURCE_DATA_DIR = BASE_DIR / "data"
DATA_DIR = PROJECT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"

# Ensure directories exist
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_text(pdf_rel_path):
    pdf_path = SOURCE_DATA_DIR / pdf_rel_path
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return None

    print(f"Extracting: {pdf_rel_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [ERROR] Failed to open PDF {pdf_path}: {e}")
        return None
        
    extracted_pages = []
    warnings = []

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")  # Plain text layout
        except Exception as e:
            print(f"  [WARNING] Failed to extract text from page {page_num + 1}: {e}")
            text = ""
            
        # Heuristics for low text density/scanned page
        text_len = len(text.strip())
        img_count = len(page.get_images())
        
        is_low_density = text_len < 150
        if is_low_density:
            warning_msg = f"Page {page_num + 1} has low text density (characters: {text_len}, images: {img_count})"
            warnings.append(warning_msg)
            print(f"  [WARNING] {warning_msg}")

        extracted_pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": text_len,
            "image_count": img_count,
            "low_density": is_low_density
        })

    # Save to text file
    rel_path_str = str(pdf_rel_path).replace("\\", "_").replace("/", "_")
    if rel_path_str.lower().endswith(".pdf"):
        rel_path_str = rel_path_str[:-4]
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', rel_path_str)
    out_txt_path = EXTRACTED_DIR / f"{safe_name}_extracted.txt"
    
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for p in extracted_pages:
            f.write(f"--- PAGE {p['page_number']} ---\n")
            f.write(p["text"])
            f.write("\n")

    return {
        "file": str(pdf_rel_path).replace("\\", "/"),  # Normalize to forward slashes for cross-platform manifest
        "output_file": str(out_txt_path.relative_to(PROJECT_DIR)),
        "page_count": len(doc),
        "warnings": warnings
    }

def main():
    # Clean up existing files in extracted directory
    print("Cleaning up old extracted text files...")
    for f in EXTRACTED_DIR.glob("*_extracted.txt"):
        try:
            f.unlink()
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    # Discover all PDF files
    print("Scanning for PDF files...")
    pdf_files = []
    for root, dirs, files in os.walk(SOURCE_DATA_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(SOURCE_DATA_DIR)
                pdf_files.append(rel_path)

    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files to extract.")

    manifest = []
    for idx, rel_path in enumerate(pdf_files):
        print(f"\n[{idx+1}/{len(pdf_files)}] ", end="")
        res = extract_pdf_text(rel_path)
        if res:
            manifest.append(res)
            
    manifest_path = DATA_DIR / "extracted_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nExtraction complete. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
