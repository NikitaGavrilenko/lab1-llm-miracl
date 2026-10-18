from pathlib import Path
import sys
from pypdf import PdfReader
reader=PdfReader(sys.argv[1])
Path(sys.argv[2]).write_text('\f'.join(p.extract_text() or '' for p in reader.pages),encoding='utf-8')
