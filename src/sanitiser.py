import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "extracted_features" / "hand_landmarks.csv"
OUTPUT = ROOT / "data" / "extracted_features" / "hand_landmarks_sanitised.csv"

def centre_hand(flat):
    lm = flat.reshape(21, 3)
    wrist = lm[0]
    return (lm - wrist).reshape(-1)

def normalise_scale(flat):
    lm = flat.reshape(21, 3)
    scale = np.linalg.norm(lm[9])  # wrist → middle MCP
    if scale == 0:
        return None
    return (lm / scale).reshape(-1)

clean_rows = []

with INPUT.open(newline="") as f:
    reader = csv.reader(f)
    next(reader)  # Skips header

    for row in reader:
        # Skip malformed rows
        if len(row) != 65:
            continue

        label = row[-1]
        data = np.asarray(row[1:-1], dtype=float)  # skip instance_id, keep 63 values

        if not np.isfinite(data).all():
            continue

        data = centre_hand(data)
        data = normalise_scale(data)
        if data is None:
            continue

        ## Drop Z (optional but recommended)
        # data = data.reshape(21, 3)[:, :2].reshape(-1)

        # Final sanity filter
        if np.abs(data).max() > 5:
            continue

        clean_rows.append([label, *data])

np.savetxt(
    OUTPUT,
    clean_rows,
    delimiter=",",
    fmt="%s"
)
