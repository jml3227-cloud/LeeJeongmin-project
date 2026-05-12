from Bio import Entrez
from xml.etree import ElementTree as ET
import json
import time
import os

Entrez.email = "jml3227@gmail.com"

queries = {
    "cell morphology": 2000,
    "bacterial morphology": 1000,
    "histopathology cell morphology": 3000,
    "hematoxylin eosin staining morphology": 4000,
}

def fetch_pmc_ids(query, max_results):
    handle = Entrez.esearch(
        db="pmc",
        term=query + " AND open access[filter]",
        retmax=max_results
    )
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_full_text(pmc_id):
    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml", retmode="xml")
    return handle.read()

def extract_text_from_xml(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
        texts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
        return " ".join(texts)
    except Exception:
        return ""

os.makedirs("data", exist_ok=True)

all_ids = set()
results = []

for query, max_count in queries.items():
    print(f"\nFetching: {query} ({max_count}편)")
    ids = fetch_pmc_ids(query, max_count)
    new_ids = [id for id in ids if id not in all_ids]
    all_ids.update(new_ids)
    print(f"  중복 제거 후: {len(new_ids)}개")

    for i, pmc_id in enumerate(new_ids):
        try:
            xml_bytes = fetch_full_text(pmc_id)
            text = extract_text_from_xml(xml_bytes)
            if not text:
                continue
            results.append({
                "pmc_id": pmc_id,
                "query": query,
                "text": text
            })
            if i % 100 == 0:
                print(f"  {i}/{len(new_ids)} 완료")
            time.sleep(0.34)
        except Exception as e:
            print(f"  Error {pmc_id}: {e}")
            continue

with open("data/pmc_raw.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"\n총 {len(results)}편 저장 완료")