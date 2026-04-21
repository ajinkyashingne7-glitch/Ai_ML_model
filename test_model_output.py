#!/usr/bin/env python
"""Test current model output on test images"""
from pathlib import Path
from inference import extract_invoice_data
from utils import parse_output

folder = Path('testing img')
files = sorted([f for f in folder.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

print("=" * 60)
print("TESTING INVOICE EXTRACTION")
print("=" * 60)

for i, image_file in enumerate(files[:3], 1):
    print(f"\n{i}. {image_file.name}")
    print("-" * 40)
    
    raw = extract_invoice_data(str(image_file))
    print(f"Raw model output: {repr(raw)}")
    
    if raw:
        invoice_no, date = parse_output(raw)
        print(f"Invoice No: {invoice_no}")
        print(f"Date: {date}")
    else:
        print("Model returned None")
