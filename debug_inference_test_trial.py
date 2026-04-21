from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

model_path = 'trial'
processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
model.eval()

image_path = 'Images/IMG_20250531_0001.png'
image = Image.open(image_path).convert('RGB')
inputs = processor(image, return_tensors='pt')
pixel_values = inputs.pixel_values
outputs = model.generate(pixel_values=pixel_values, max_length=512, num_beams=1, length_penalty=1.0, num_return_sequences=1)
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print('OUTPUT:', repr(result))
print('TOKENS:', outputs[0][:20])
