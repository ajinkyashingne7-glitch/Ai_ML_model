import os
import sys
from inference import extract_invoice_data
from utils import parse_output

def process_invoices(folder="testing img"):
    """Process all invoice images in a folder"""
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(f" Error: Folder '{folder}' not found!")
        return
    
    image_files = [f for f in os.listdir(folder) 
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    if not image_files:
        print(f" No image files found in '{folder}'")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
    results = []
    
    for file in image_files:
        image_path = os.path.join(folder, file)
        
        print("=" * 50)
        print(f"Processing: {file}")
        print("=" * 50)
        
        try:
            # Extract data from image
            result = extract_invoice_data(image_path)
            
            if result is None:
                print("Failed to process image - No output generated")
                results.append({
                    "file": file,
                    "status": "error",
                    "error": "No output generated"
                })
                continue
            
            print(f"\nRaw Model Output:\n{result}\n")
            
            # Parse the output - handle None gracefully
            try:
                invoice_no, date = parse_output(result)
            except Exception as parse_error:
                print(f"Error parsing output: {str(parse_error)}")
                invoice_no, date = None, None
            
            print("EXTRACTED DATA:")
            print(f"   Invoice Number: {invoice_no if invoice_no else 'Not found'}")
            print(f"   Invoice Date:   {date if date else 'Not found'}")
            
            results.append({
                "file": file,
                "invoice_number": invoice_no,
                "invoice_date": date,
                "status": "success" if invoice_no or date else "partial"
            })
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            results.append({
                "file": file,
                "status": "error",
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    for r in results:
        status_icon = "Yes" if r["status"] == "success" else "Warning" if r["status"] == "partial" else "No"
        print(f"{status_icon} {r['file']}: Invoice No: {r.get('invoice_number', 'N/A')} | Date: {r.get('invoice_date', 'N/A')}")

if __name__ == "__main__":
    # Process images
    process_invoices("testing img")