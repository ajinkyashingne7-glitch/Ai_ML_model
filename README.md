# Invoice Data Extractor - Donut Model

A deep learning-based invoice data extraction system using the Donut (Document Understanding Transformer) model to automatically extract invoice numbers and dates from invoice images.

## System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration) - Optional but recommended
- 4GB+ RAM minimum, 8GB+ recommended

## Installation

### 1. Install Dependencies

First, rename `requirments.txt` to `requirements.txt` (if not already done):

```bash
# Install required packages
pip install -r requirements.txt
```

If you have a GPU (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pillow accelerate
```

### 2. Verify Model Files

Ensure you have the trained model files in `Final_Trained_Model_files/`:
- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `tokenizer.json`
- `special_tokens_map.json`
- `generation_config.json`
- `added_tokens.json`
- `tokenizer_config.json`
- `sentencepiece.bpe.model`

## Project Structure

```
Invoice_donut_converter/
├── main.py                          # Main execution script
├── inference.py                     # Model inference logic
├── utils.py                         # Parsing and validation utilities
├── train.py                         # Training script (reference)
├── requirements.txt                 # Python dependencies
├── Final_Trained_Model_files/       # Pre-trained model
├── dataset/                         # Training dataset
├── testing img/                     # Test invoice images
└── README.md                        # This file
```

## Usage

### Basic Usage (Extract from Images)

1. Place invoice images in the `testing img/` folder

2. Run the main script:
```bash
python main.py
```

3. View extracted data in the console output

### Output Format

The model expects JSON output with the following structure:
```json
{
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15"
}
```

#### Example Output:
```
==================================================
Processing: invoice_001.jpg
==================================================

📄 Raw Model Output:
{"invoice_number": "INV-123456", "invoice_date": "2024-01-15"}

✅ EXTRACTED DATA:
   Invoice Number: INV-123456
   Invoice Date:   2024-01-15
```

## Key Improvements Made

### ✅ Fixed Issues:
1. **Removed invalid `decoder_input_ids` parameter** - The original code used a non-existent parameter that caused runtime errors
2. **Fixed requirements.txt typo** - Changed "requirments" to "requirements"
3. **Improved exception handling** - Replaced bare `except:` with specific exception types
4. **Added input validation** - Better error checking in model loading and image processing
5. **Enhanced output formatting** - Better user-friendly console output

### 🔧 Code Features:
- **Robust error handling** - Graceful failure with informative messages
- **Batch processing** - Process multiple images in one run
- **Validation** - Invoice number format validation to filter out invalid entries
- **Summary report** - Shows success/failure status for all processed images

## File Descriptions

### main.py
Orchestrates the invoice extraction pipeline. Handles:
- Folder validation
- Image discovery
- Batch processing
- Results summarization

### inference.py
Loads and runs the pre-trained Donut model. Features:
- Model loading from local files
- Image preprocessing
- Model inference with proper parameters
- Error handling for image loading and processing

### utils.py
Utility functions for output parsing:
- JSON parsing of model output
- Invoice number validation
- Text cleaning
- Filters invalid patterns (phone numbers, GST numbers, etc.)

### train.py
Reference training script. Shows:
- How the model was trained
- Dataset preparation
- Training configuration
- Model serialization

## Troubleshooting

### Issue: Model not found error
**Solution:** Ensure `Final_Trained_Model_files/` directory exists with all required files

### Issue: No images found in testing img folder
**Solution:** Place `.jpg`, `.png`, or `.jpeg` files in the `testing img/` folder

### Issue: Out of memory error
**Solution:** 
- Reduce batch size if training
- Close other applications
- Use GPU if available (install CUDA)

### Issue: Invoice number not being extracted
**Solution:** 
- Check image quality (must be legible)
- Ensure invoice format matches training data
- Check model output for raw JSON structure

## Model Details

- **Base Model:** Donut (naver-clova-ix/donut-base)
- **Task:** Document Understanding (Invoice extraction)
- **Output Format:** JSON with invoice_number and invoice_date
- **Max Length:** 512 tokens

## Next Steps

1. Test with your invoice images
2. Adjust validation rules in `utils.py` if needed
3. Fine-tune the model with your specific invoice formats if accuracy is low
4. Deploy as a microservice or API

## Support

For issues or improvements, check:
1. Console error messages
2. Image file format and quality
3. Model file integrity
4. Python and package versions
