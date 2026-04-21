#!/usr/bin/env python
"""Verify that the trained model files are valid"""
import os
from pathlib import Path

model_path = "./Final_Trained_Model_files"

print("=" * 60)
print("CHECKING MODEL FILES")
print("=" * 60)

required_files = [
    'config.json',
    'generation_config.json', 
    'model.safetensors',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json'
]

all_exist = True
for f in required_files:
    path = Path(model_path) / f
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    status = "Yes" if exists else "No"
    print(f"{status} {f:30s} {'(' + str(size//1024) + ' KB)' if exists else '(MISSING)'}")
    if not exists:
        all_exist = False

print("\n" + "=" * 60)
if all_exist:
    print("All model files present")
    
    # Try to load the model
    print("\nAttempting to load model...\n")
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        print("Loading processor...")
        processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
        print("Processor loaded")
        
        print("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
        print("Model loaded")
        
        model.eval()
        print("Model in eval mode")
        
        # Check model config
        print(f"\nModel config: {model.config.decoder.hidden_size} hidden dim")
        print(f"Model has {model.config.decoder.num_hidden_layers} decoder layers")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print("Some model files are missing!")
    print("You need to train the model first with: python train.py")
