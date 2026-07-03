"""
OCR Fallback Script
====================
Re-extracts the 7 image-heavy PDFs using PyMuPDF page rendering + pytesseract.
No Poppler needed. Requires Tesseract OCR engine installed on system.

Install Tesseract from:
  https://github.com/UB-Mannheim/tesseract/wiki
  (Download tesseract-ocr-w64-setup-5.x.x.exe, install to default location)

Then run:
  python src/ocr_fallback.py
  python src/chunk.py
  python src/embed.py
"""

import os, sys, re, json, io
import fitz
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_DIR.parent
SOURCE_DATA_DIR = BASE_DIR / "data"
DATA_DIR = PROJECT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

OCR_TARGETS = [
    "Data Science/Data Science Cheat Sheet(Python_R).pdf",
    "Data Science/Top Data Science Libraries.pdf",
    "Deep Learning/Coursera Deep Learning course Notes.pdf",
    "Git/git_cheat_sheet.pdf",
    "Interview Questions/Top 100 Python questions.pdf",
    "Linux/Linux Cheat Sheet.pdf",
    "Python/Interview Questions.pdf",
]

def find_tesseract():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\shank\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\tesseract\tesseract.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    import shutil
    return shutil.which("tesseract")

def ocr_pdf(pdf_rel_path_str, tesseract_cmd):
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    pdf_path = SOURCE_DATA_DIR / pdf_rel_path_str
    if not pdf_path.exists():
        print(f"  [ERROR] PDF not found: {pdf_path}")
        return None
    print(f"  OCR-ing: {pdf_rel_path_str}")
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            text = pytesseract.image_to_string(img, lang="eng")
        except Exception as e:
            print(f"    [WARN] OCR page {page_num+1}: {e}")
            text = ""
        print(f"    Page {page_num+1}: {len(text.strip())} chars")
        pages_text.append((page_num + 1, text))
    rel = pdf_rel_path_str.replace("\\","_").replace("/","_")
    if rel.lower().endswith(".pdf"): rel = rel[:-4]
    safe = re.sub(r'[^a-zA-Z0-9_\-]','_', rel)
    out = EXTRACTED_DIR / f"{safe}_extracted.txt"
    with open(out, "w", encoding="utf-8") as f:
        for pn, t in pages_text:
            f.write(f"--- PAGE {pn} ---\n{t}\n")
    total = sum(len(t.strip()) for _,t in pages_text)
    print(f"  Saved {out} ({total} total chars, {len(pages_text)} pages)")
    return {"file": pdf_rel_path_str.replace("\\","/"), "output_file": str(out.relative_to(PROJECT_DIR)), "page_count": len(pages_text), "ocr": True}

def main():
    try:
        import pytesseract
    except ImportError:
        print("[ERROR] Run: pip install pytesseract pillow"); return
    tess = find_tesseract()
    if not tess:
        print("[ERROR] Tesseract not found. Install from:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        print("  Default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        return
    print(f"Tesseract: {tess}\nProcessing {len(OCR_TARGETS)} files...\n")
    results = []
    for pdf_rel in OCR_TARGETS:
        print(f"\n{'='*60}")
        r = ocr_pdf(pdf_rel, tess)
        if r: results.append(r)
    if results:
        mp = DATA_DIR / "extracted_manifest.json"
        with open(mp,"r",encoding="utf-8") as f: manifest = json.load(f)
        ocr_files = {r["file"] for r in results}
        manifest = [m for m in manifest if m["file"] not in ocr_files]
        manifest.extend(results)
        with open(mp,"w",encoding="utf-8") as f: json.dump(manifest,f,indent=4)
        print(f"\n[DONE] Updated extracted_manifest.json with {len(results)} OCR files.")
        print("\nNext steps:")
        print("  1. python src/chunk.py")
        print("  2. python src/embed.py")
    else:
        print("\n[WARN] Nothing processed.")

if __name__ == "__main__":
    main()
