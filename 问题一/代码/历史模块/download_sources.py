"""Download the original authors' public supplementary data (no model pickle).

Run separately from modeling; all downloads are recorded with SHA-256 hashes.
"""
from pathlib import Path
from urllib.parse import quote
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'external_sources'
FILES = [
    'Supplemental Files and Figure source files.xlsx',
    'Main Figure Generator.Rmd',
    'Sensory data quality control - ANOVA on repeated samples.Rmd',
    'README.txt',
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in FILES:
        url = 'https://zenodo.org/records/10653704/files/' + quote(name) + '?download=1'
        target = OUT / name
        if not target.exists():
            response = requests.get(url, timeout=(20, 90))
            response.raise_for_status()
            target.write_bytes(response.content)
        raw = target.read_bytes()
        if name.endswith('.xlsx'):
            assert raw[:2] == b'PK', 'Downloaded workbook is not an XLSX archive'
        manifest.append(dict(file=name, url=url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        print(name, len(raw), flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
