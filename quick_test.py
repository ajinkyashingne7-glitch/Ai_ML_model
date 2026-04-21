#!/usr/bin/env python
"""Quick debug to see raw model output"""
from pathlib import Path
from inference import extract_invoice_data
from utils import parse_output

folder = Path('testing img')
files = sorted([f for f in folder.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

if not files:
    print(" No test images found")
else:
    print(f"Testing first image: {files[0].name}\n")
    
    raw_output = extract_invoice_data(str(files[0]))
    print(f"Raw model output: {repr(raw_output)}")
    print()
    
    if raw_output:
        invoice_no, date = parse_output(raw_output)
        print(f"Parsed Invoice No: {invoice_no}")
        print(f"Parsed Date: {date}")
    else:
        print(" Model returned None")
