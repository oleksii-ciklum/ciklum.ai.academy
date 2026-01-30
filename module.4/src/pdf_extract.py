from pathlib import Path
import fitz


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        parts.append(f"\n\n[PAGE {i+1}]\n{text}")
    return "\n".join(parts)


def save_pdf_text(pdf_path: Path, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(extract_pdf_text(pdf_path), encoding="utf-8")
