"""
Verification script to check if the invoice extraction model is properly set up
Run this before processing actual invoices
"""

import os
import sys
import json
from pathlib import Path

def check_structure():
    """Check if all required files and directories exist"""
    print(" Checking project structure...\n")
    
    required_files = [
        "main.py",
        "inference.py",
        "utils.py",
        "train.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "Final_Trained_Model_files",
        "testing img"
    ]
    
    # Check files
    print("Checking Python files:")
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   {file}")
        else:
            print(f"   {file} - NOT FOUND")
            all_good = False
    
    print("\nChecking directories:")
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"{dir_name}")
        else:
            print(f"{dir_name} - NOT FOUND")
            all_good = False
    
    return all_good

def check_model_files():
    """Check if all model files are present"""
    print("\nChecking model files in Final_Trained_Model_files/...\n")
    
    model_dir = "Final_Trained_Model_files"
    if not os.path.exists(model_dir):
        print(f" Model directory not found!")
        return False
    
    required_model_files = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "generation_config.json"
    ]
    
    all_present = True
    for file in required_model_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"   {file} ({size:.2f} MB)")
        else:
            print(f"   {file} - NOT FOUND")
            all_present = False
    
    return all_present

def check_imports():
    """Check if all required packages are installed"""
    print("\nChecking Python packages...\n")
    
    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "datasets": "Datasets"
    }
    
    all_installed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   {name}")
        except ImportError:
            print(f"   {name} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n   Install missing packages with: pip install -r requirements.txt")
    
    return all_installed

def check_test_images():
    """Check if there are test images available"""
    print("\nChecking test images...\n")
    
    test_dir = "testing img"
    if not os.path.exists(test_dir):
        print(f"   '{test_dir}' directory not found")
        return False
    
    image_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in os.listdir(test_dir) 
                   if os.path.splitext(f)[1].lower() in image_extensions]
    
    if image_files:
        print(f"Found {len(image_files)} test image(s):")
        for img in image_files[:5]:  # Show first 5
            print(f"      - {img}")
        if len(image_files) > 5:
            print(f"      ... and {len(image_files) - 5} more")
        return True
    else:
        print(f" No test images found in '{test_dir}'")
        print(f" Place .jpg, .png, or .jpeg files there to test")
        return False

def check_model_loading():
    """Try to load the model to verify it works"""
    print("\nTesting model loading...\n")
    
    try:
        print("Loading processor...")
        from transformers import DonutProcessor
        processor = DonutProcessor.from_pretrained("./Final_Trained_Model_files")
        print("Processor loaded successfully")
        
        print("   Loading model...")
        from transformers import VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained("./Final_Trained_Model_files")
        print("Model loaded successfully")
        
        model.eval()
        print("Model set to evaluation mode")
        
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("Invoice Extraction Model - Verification Check")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all checks
    results.append(("Project Structure", check_structure()))
    results.append(("Model Files", check_model_files()))
    results.append(("Python Packages", check_imports()))
    results.append(("Test Images", check_test_images()))
    results.append(("Model Loading", check_model_loading()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60 + "\n")
    
    all_critical_ok = all(r[1] for r in results if r[0] in [
        "Project Structure",
        "Model Files",
        "Python Packages",
        "Model Loading"
    ])
    
    for check_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{check_name:.<40} {status}")
    
    print("\n" + "=" * 60)
    
    if all_critical_ok:
        print("All checks passed! Ready to use:")
        print("   python main.py")
    else:
        print("Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Install requirements: pip install -r requirements.txt")
        print("  2. Place test images in 'testing img/' folder")
        print("  3. Ensure Final_Trained_Model_files/ has all files")
    
    print("=" * 60 + "\n")
    
    return 0 if all_critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())
