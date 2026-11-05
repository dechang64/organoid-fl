"""Bootstrap: download HCMI organoid images + create eval scripts."""
import json
import csv
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = Path("data/hcmi_organoids")
(OUT / "images").mkdir(parents=True, exist_ok=True)

# 1. Query HCMI API
print("Querying HCMI API...")
q = {"query":'query($s:JSON){model{hits(first:700,filters:$s){total edges{node{name primary_site type disease_status clinical_diagnosis{clinical_tumor_diagnosis} files{hits(first:10){edges{node{file_name file_url magnification}}}}}}}}}',"variables":{"s":{"op":"and","content":[{"op":"in","content":{"field":"type","value":["3-D: Organoid"]}}]}}}
req = urllib.request.Request("https://hcmi-searchable-catalog.nci.nih.gov/api/graphql", data=json.dumps(q).encode(), headers={"Content-Type":"application/json"})
data = json.loads(urllib.request.urlopen(req, timeout=60).read())
entries = []
for e in data["data"]["model"]["hits"]["edges"]:
    n = e["node"]
    if n["type"] != "3-D: Organoid": continue
    for fe in n.get("files",{}).get("hits",{}).get("edges",[]):
        fn = fe["node"]
        entries.append({"model":n["name"],"center":n["name"].split("-")[1],"primary_site":n.get("primary_site",""),"diagnosis":n.get("clinical_diagnosis",{}).get("clinical_tumor_diagnosis",""),"file_name":fn["file_name"],"file_url":fn["file_url"],"magnification":fn.get("magnification","")})

# Save CSV
with open(OUT/"hcmi_organoid_images.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model","center","primary_site","diagnosis","file_name","magnification","file_url"])
    w.writeheader()
    w.writerows(entries)
print(f"Manifest: {len(entries)} images")

# 2. Download
def dl(e):
    d = OUT/"images"/e["center"]
    d.mkdir(parents=True, exist_ok=True)
    p = d/e["file_name"]
    if p.exists() and p.stat().st_size > 1000: return "skip"
    try:
        urllib.request.urlretrieve(e["file_url"], str(p))
        return "ok"
    except Exception as ex:
        return str(ex)

dl_count = skip_count = err_count = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(dl, e): e for e in entries}
    for i, f in enumerate(as_completed(futs)):
        r = f.result()
        if r == "ok": dl_count += 1
        elif r == "skip": skip_count += 1
        else: err_count += 1
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(entries)} (dl={dl_count} skip={skip_count} err={err_count})")

print(f"\nDone! Downloaded={dl_count} Skipped={skip_count} Errors={err_count}")
print(f"Images in: {OUT/'images'}")
print(f"CSV: {OUT/'hcmi_organoid_images.csv'}")
