import sys
sys.path.append('.')
from inference import extract_invoice_data
from utils import parse_output
from pathlib import Path

folder = Path('testing img')
files = list(folder.glob('*'))
files = [f for f in files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

if files:
    result = extract_invoice_data(str(files[0]))
    print('Raw model output:', repr(result))
    invoice_no, date = parse_output(result)
    print('Parsed invoice_no:', invoice_no)
    print('Parsed date:', date)
else:
    print('No test images found')