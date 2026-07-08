"""Pack mouse_liver_v2 results — only json/txt/csv/yaml, preserve dir structure."""
import zipfile
import os

SRC = r"runs\mouse_liver_v2"
DST = r"mouse_liver_v2_results.zip"

if not os.path.isdir(SRC):
    print(f"[ERROR] {SRC} not found")
    raise SystemExit(1)

n = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(SRC):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".json", ".txt", ".csv", ".yaml", ".yml"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, SRC)
                zf.write(full, rel)
                n += 1

size_mb = os.path.getsize(DST) / 1024 / 1024
print(f"Done! {n} files -> {DST} ({size_mb:.1f} MB)")
