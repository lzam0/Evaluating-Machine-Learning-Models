import csv
import sys
import tempfile
import os
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python remove_last_column.py <csv_path> [output_csv_path]")
    sys.exit(1)

p = Path(sys.argv[1])
if not p.exists():
    print(f"File not found: {p}")
    sys.exit(1)

# If an output path is provided, write there and do not replace the original.
out_path = None
if len(sys.argv) >= 3:
    out_path = Path(sys.argv[2])

if out_path:
    with p.open(newline='', encoding='utf-8') as rf, out_path.open('w', newline='', encoding='utf-8') as wf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)
        for row in reader:
            writer.writerow(row[:-1] if row else [])
    print(f"Wrote new file without last column to {out_path}")
else:
    # Write to a temp file in the same directory then atomically replace original
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix='.tmp', text=True)
    os.close(tmp_fd)
    try:
        with p.open(newline='', encoding='utf-8') as rf, open(tmp_path, 'w', newline='', encoding='utf-8') as tf:
            reader = csv.reader(rf)
            writer = csv.writer(tf)
            for row in reader:
                writer.writerow(row[:-1] if row else [])
        os.replace(tmp_path, str(p))
        print(f"Removed last column from {p}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
