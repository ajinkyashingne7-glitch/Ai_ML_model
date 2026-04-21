from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

img = Image.open('Images/IMG_20250531_0001.png').convert('RGB')
for model_path in ['Final_Trained_Model_files', 'trial']:
    print('MODEL', model_path)
    proc = DonutProcessor.from_pretrained(model_path, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    for text in [None, '<s_invoice>', '<s_iitcdip>', '<s_synthdog>']:
        print(' text=', text)
        kwargs = {'return_tensors': 'pt'}
        if text is not None:
            kwargs['text'] = text
        inputs = proc(img, **kwargs)
        out = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=120,
            num_beams=1,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        result = proc.batch_decode(out, skip_special_tokens=True)[0]
        print('  output=', repr(result))
        print('  tokens=', out[0].tolist())
