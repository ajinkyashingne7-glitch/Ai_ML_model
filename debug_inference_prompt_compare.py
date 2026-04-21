from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

image_path = 'Images/IMG_20250531_0001.png'
image = Image.open(image_path).convert('RGB')
for model_path in ['Final_Trained_Model_files', 'trial']:
    print('MODEL:', model_path)
    processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    for use_prompt in [False, True]:
        print(' use_prompt=', use_prompt)
        if use_prompt:
            inputs = processor(image, text='<s_invoice>', return_tensors='pt')
        else:
            inputs = processor(image, return_tensors='pt')
        pixel_values = inputs.pixel_values
        out = model.generate(pixel_values=pixel_values, max_length=512, num_beams=1, length_penalty=1.0, num_return_sequences=1)
        result = processor.batch_decode(out, skip_special_tokens=True)[0]
        print('  output:', repr(result[:300]))
        print('  tokens:', out[0][:20].tolist())
    print('---')
