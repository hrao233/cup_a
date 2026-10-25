"""Acquire publicly linked documents; record provenance and completeness clues.

No authentication tokens, login bypasses, or paid purchases are used.
"""
from pathlib import Path
import hashlib
import json
import sys
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'.tools'/'pdf_reader'))
from pypdf import PdfReader

DOCUMENTS=[
    ('GB_T_29604_2013',
     'https://cfsn-2024upload.oss-cn-beijing.aliyuncs.com/files/20260313/2026031316585069178.pdf',
     'https://www.cfsn.cn/zcwk/detail/2163/7853.html'),
    ('ISO_4121_2003',
     'https://img.antpedia.com/standard/pdf/X04/1705/ISO%204121-2003.pdf',
     'https://www.iso.org/standard/33817.html'),
    ('ISO_13299_2016',
     'https://standards.isodoc.my/iso/ISO%2013299-2016%20PDF.pdf',
     'https://www.iso.org/standard/58042.html'),
    ('Meilgaard_1979_Beer_flavour_terminology',
     'https://www.brassageamateur.com/wiki/images/2/25/J_Institute_Brewing_-_January%E2%80%90February_1979_-_Meilgaard_-_BEER_FLAVOUR_TERMINOLOGY1.pdf',
     'https://doi.org/10.1002/j.2050-0416.1979.tb06826.x'),
    ('Schreurs_2024_Beer_flavor_ML',
     'https://bio.kuleuven.be/eeb/tw/schreurs-etal-natcomm-2024.pdf',
     'https://doi.org/10.1038/s41467-024-46346-0'),
]


def main():
    out=ROOT/'fulltexts';out.mkdir(exist_ok=True)
    records=[]
    for name,url,authority in DOCUMENTS:
        record=dict(document=name,url=url,authority=authority)
        target=out/(name+'.pdf')
        try:
            if not target.exists():
                r=requests.get(url,timeout=(20,60))
                record['http_status']=r.status_code
                r.raise_for_status()
                if not r.content.lstrip().startswith(b'%PDF'):
                    raise ValueError('Response is not a PDF document')
                target.write_bytes(r.content)
            raw=target.read_bytes()
            reader=PdfReader(target)
            pages=[page.extract_text() or '' for page in reader.pages]
            (out/(name+'.txt')).write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+s for i,s in enumerate(pages)),encoding='utf-8')
            record.update(status='downloaded',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                          pdf_pages=len(pages),extracted_characters=sum(map(len,pages)),
                          contains_preview_marker=any('This is a preview' in p or 'STANDARD PREVIEW' in p for p in pages))
            print(name,'PAGES',len(pages),'CHARS',sum(map(len,pages)),flush=True)
            print('FIRST',pages[0][:350].replace('\n',' '),flush=True)
            print('LAST',pages[-1][-450:].replace('\n',' '),flush=True)
        except Exception as e:
            record.update(status='not_obtained',error=type(e).__name__+': '+str(e))
            print(name,record['error'],flush=True)
        records.append(record)
        (ROOT/'fulltext_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
