#!/usr/bin/env python
"""Debug: See raw model output before cleaning"""
from pathlib import Path
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

model_path = "./Final_Trained_Model_files"

print("Loading model...")
processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
model.eval()

print("Loading test image...")
folder = Path('testing img')
files = sorted([f for f in folder.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])

if not files:
    print(" No test images")
else:
    image = Image.open(str(files[0])).convert("RGB")
    
    print(f"\n Processing: {files[0].name}\n")
    
    inputs = processor(image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    print(f"Pixel values shape: {pixel_values.shape}")
    
    # Generate with detailed output
    print("\n Generating output...\n")
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=512,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    sequences = outputs.sequences
    print(f"Generated sequence length: {sequences.shape}")
    print(f"Generated token IDs: {sequences[0]}")
    
    # Decode with special tokens
    result_with_special = processor.batch_decode(sequences, skip_special_tokens=False)[0]
    print(f"\n Raw output (with special tokens):")
    print(f"  {repr(result_with_special)}")
    
    # Decode without special tokens
    result_no_special = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    print(f"\nCleaned output (no special tokens):")
    print(f"  {repr(result_no_special)}")
    
    # Manual cleaning
    result_manual = result_with_special.replace("<s_invoice>", "").replace("</s>", "").strip()
    print(f"\n Manually cleaned output:")
    print(f"  {repr(result_manual)}")
