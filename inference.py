import os
import torch
from PIL import Image

# Use your exact folder name or override with INVOICE_MODEL_PATH
model_path = os.getenv("INVOICE_MODEL_PATH", "./Final_Trained_Model_files")
processor = None
model = None


def load_model():
    global processor, model
    if processor is None or model is None:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        processor = DonutProcessor.from_pretrained(model_path, local_files_only=True)
        model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True)
        model.to(torch.device("cpu"))
        model.eval()


def extract_invoice_data(image_path):
    load_model()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {image_path} - {str(e)}")
        return None

    try:
        inputs = processor(image, text="<s_invoice>", return_tensors="pt")
        inputs = {k: v.to(torch.device("cpu")) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=1,
            length_penalty=1.0,
            num_return_sequences=1,
        )

        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print(f"[DEBUG] Raw model output (tokens): {outputs[0]}")
        print(f"[DEBUG] Raw model output (text): {repr(result)}")

        # Remove any prompt tokens that are not handled by skip_special_tokens
        result = result.replace("<s_invoice>", "")
        result = result.replace("<s>", "")
        result = result.replace("</s>", "")
        result = result.strip()

        if not result or result in ["<unk>", ""]:
            return None

        return result

    except Exception as e:
        print(f"Error processing image with model: {str(e)}")
        return None