from inference import extract_invoice_data

path = 'Images/IMG_20250531_0001.png'
print('IMAGE:', path)
print('OUTPUT:', extract_invoice_data(path))
