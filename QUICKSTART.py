#!/usr/bin/env python
"""
QUICK START GUIDE - Invoice Extraction Model
======================================
Follow these steps to get started with the invoice extractor.
"""

def print_step(step_num, title, description, command=None):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print('='*60)
    print(f"\n{description}")
    if command:
        print(f"\n💻 Run this command:")
        print(f"   {command}")

def main():
    print("\n" + "="*60)
    print("INVOICE EXTRACTION MODEL - QUICK START")
    print("="*60)
    
    # Step 1: Verify Setup
    print_step(
        1,
        "Verify Your Setup",
        """Before running the model, let's verify everything is installed correctly.
This will check:
- All project files exist
- Required Python packages are installed  
- Model files are in place
- Test images are available""",
        "python verify_setup.py"
    )
    
    # Step 2: Prepare Images
    print_step(
        2,
        "Prepare Invoice Images",
        """Place the invoice images you want to extract data from in the 'testing img/' folder.

Supported formats: .jpg, .png, .jpeg

Example:
  testing img/invoice_001.jpg
  testing img/invoice_002.png
  testing img/invoice_003.jpeg
  
Tips for best results:
- Use clear, well-lit images
- Ensure the entire invoice is visible
- Avoid heavily rotated images (keep mostly upright)
- Images should be at least 300x300 pixels""",
        None
    )
    
    # Step 3: Extract Data
    print_step(
        3,
        "Extract Invoice Data",
        """Run the main extraction script to process all images in the folder.
The system will:
1. Find all images in 'testing img/'
2. Process each image with the Donut model
3. Extract invoice number and date
4. Display results in console
5. Show a summary report""",
        "python main.py"
    )
    
    # Step 4: Interpret Results
    print_step(
        4,
        "Interpret the Results",
        """The output will show:

Green checkmarks = Successfully extracted
Yellow warns = Partial extraction (some fields missing)  
Red X's = Failed to process

Example output:
  Processing: invoice_001.jpg
  ==================================================
Raw Model Output:
  {\"invoice_number\": \"INV-123456\", \"invoice_date\": \"2024-01-15\"}
  
 EXTRACTED DATA:
     Invoice Number: INV-123456
     Invoice Date:   2024-01-15""",
        None
    )
    
    # Advanced Options
    print("\n" + "="*60)
    print("ADVANCED OPTIONS")
    print("="*60)
    
    print("\nEdit Invoice Validation (utils.py)")
    print("   If the system is too strict/lenient, you can adjust:")
    print("   - is_valid_invoice_number() function")
    print("   - Invoice number length limits")
    print("   - Allowed character patterns")
    
    print("\n Retrain the Model (train.py)")
    print("   To train with your own data:")
    print("   1. Prepare dataset in dataset/train.json")
    print("   2. Place images in dataset/ folder")
    print("   3. Run: python train.py")
    print("   4. Wait for training to complete")
    print("   5. Model saves to Final_Trained_Model_files/")
    
    print("\nDense Output (For Debugging)")
    print("   To see detailed model internals:")
    print("   - Check console output for raw JSON from model")
    print("   - Compare with parsed results")
    print("   - Adjust parsing logic in utils.py if needed")
    
    # Troubleshooting
    print("\n" + "="*60)
    print(" TROUBLESHOOTING")
    print("="*60)
    
    issues = [
        ("verify_setup.py fails", [
            "Ensure you're in the project root directory",
            "Run: pip install -r requirements.txt",
            "For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        ]),
        ("'No images found' error", [
            "Check that 'testing img/' folder exists",
            "Place .jpg or .png files in the folder",
            "Ensure filenames don't have unusual characters"
        ]),
        ("Model extraction returns 'N/A'", [
            "Image quality might be too low or illegible",
            "Compare extracted data with actual invoice",
            "Ensure invoice format matches training data"
        ]),
        ("Out of memory error", [
            "Close other applications",
            "If using CPU: Use fewer images at a time",
            "If using GPU: Update CUDA drivers"
        ])
    ]
    
    for issue, solutions in issues:
        print(f"\n {issue}")
        for solution in solutions:
            print(f"   → {solution}")
    
    # Next Steps
    print("\n" + "="*60)
    print(" NEXT STEPS")
    print("="*60)
    print("""
1.  Run: python verify_setup.py
2.  Add invoice images to testing img/ folder
3.  Run: python main.py
4.  Review extracted data
5.  Adjust validation in utils.py if needed (optional)
6.  Consider fine-tuning model if accuracy needs improvement

For complete documentation, see README.md
    """)
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
