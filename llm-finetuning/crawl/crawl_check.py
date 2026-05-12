from Bio import Entrez

Entrez.email = "jml3227@gmail.com"

queries = [
    "cell morphology",
    "histopathology",
    "bacterial morphology",
    "bacterial cell morphology histopathology"
]

for q in queries:
    handle = Entrez.esearch(db="pubmed", term=q, retmax=0)
    record = Entrez.read(handle)
    print(f"{q}: {int(record['Count']):,}편")

pmc_queries = [
    "bacterial morphology",
    "bacterial cell morphology histopathology",
    "cell morphology",
]

print("\n--- PMC Full Text ---")
for q in pmc_queries:
    handle = Entrez.esearch(db="pmc", term=q + " AND open access[filter]", retmax=0)
    record = Entrez.read(handle)
    print(f"{q}: {int(record['Count']):,}편")