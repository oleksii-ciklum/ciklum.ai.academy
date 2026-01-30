from pathlib import Path
from src.pdf_extract import save_pdf_text

def main():
    pdf = Path("data/raw/rag_intro.pdf")
    out = Path("data/processed/docs/rag_intro.pdf.txt")
    save_pdf_text(pdf, out)
    print("Wrote", out)

if __name__ == "__main__":
    main()
