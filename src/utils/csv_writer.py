from typing import List, Dict
import os
import csv

def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    keys = list(rows[0].keys())
    file_exists = os.path.exists(path)

    if file_exists and os.path.getsize(path) > 0:
        with open(path, "r", newline="", encoding="utf-8") as f_read:
            reader = csv.DictReader(f_read)
            existing_keys = reader.fieldnames
        if existing_keys != keys:
            raise ValueError(f"CSV header mismatch: existing {existing_keys} vs new {keys}")

    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists or os.path.getsize(path) == 0:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

