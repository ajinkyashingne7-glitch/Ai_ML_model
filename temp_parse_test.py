from utils import parse_output

examples = [
    '{"Invoice No": "878"}',
    '{"invoice_number": "INV-123"}',
    '{"Invoice No":"25-26/388","Invoice Date":"2025-05-31"}',
    'Invoice No: 904',
    '2025-05-31',
]

for text in examples:
    invoice_no, date = parse_output(text)
    print('INPUT:', text)
    print('OUTPUT:', invoice_no, date)
    print('---')
