from PIL import Image
from transformers import DonutProcessor

proc = DonutProcessor.from_pretrained('Final_Trained_Model_files', local_files_only=True)
img = Image.open('Images/IMG_20250531_0001.png').convert('RGB')
inputs = proc(img, text='<s_invoice>', return_tensors='pt')
print('keys=', list(inputs.keys()))
for k, v in inputs.items():
    print(k, type(v), getattr(v, 'shape', None), getattr(v, 'dtype', None))
