from typing import Tuple
from pypdf import PdfReader
from docx import Document

def read_any(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith('.pdf'):
        return read_pdf(file_bytes)
    if name.endswith('.docx'):
        return read_docx(file_bytes)
    return file_bytes.decode('utf-8', errors='ignore')

def read_pdf(file_bytes: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or '')
        except Exception:
            pass
    return '\n'.join(chunks).strip()

def read_docx(file_bytes: bytes) -> str:
    import io
    doc = Document(io.BytesIO(file_bytes))
    return '\n'.join([p.text for p in doc.paragraphs]).strip()

def clean_text(txt: str) -> str:
    txt = txt.replace('\r','\n')
    lines = [l.strip() for l in txt.split('\n')]
    out, blank = [], 0
    for l in lines:
        if l == '':
            blank += 1
            if blank <= 1:
                out.append('')
        else:
            blank = 0
            out.append(l)
    return '\n'.join(out).strip()
