import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from PIL import Image
import json
import os

# Paths
model_name = "./Final_Trained_Model_files"
dataset_path = "./Training Data Set/invoice_only.json"
image_folder = "./Images"

#  Load processor & model
print("Loading processor and model...")
processor = DonutProcessor.from_pretrained(model_name, local_files_only=True)
try:
    model = VisionEncoderDecoderModel.from_pretrained(
        model_name,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded.")
except OSError as e:
    print("Model load failed. This usually means your system does not have enough CPU memory or pagefile to load the model.")
    print("Options:")
    print("  1) Increase Windows virtual memory (pagefile size).")
    print("  2) Use a GPU-enabled machine if available.")
    print("  3) Use a smaller model checkpoint if possible.")
    raise


# Dataset class
class InvoiceDataset(Dataset):
    def __init__(self, json_path, image_folder, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.image_folder = image_folder

        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Filter data to only include existing images
        self.data = [item for item in self.data if os.path.exists(os.path.join(self.image_folder, os.path.basename(item["image_path"])))]
        print(f"Filtered to {len(self.data)} valid samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.image_folder, os.path.basename(item["image_path"]))
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        target = "<s_invoice>" + json.dumps(item["ground_truth"])

        labels = self.processor.tokenizer(
            target,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids.squeeze()

        # Replace pad token ID with -100 for loss computation
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# Load dataset
train_dataset = InvoiceDataset(dataset_path, image_folder, processor)
print(f"Dataset length: {len(train_dataset)}")
if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"Sample pixel_values shape: {sample['pixel_values'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")
    print("Sample loaded successfully")
else:
    print("Dataset is empty!")


# Training config
training_args = TrainingArguments(
    output_dir="./trained_model",
    per_device_train_batch_size=1,  # Reduced for CPU
    num_train_epochs=1,  # Start with 1 epoch for testing
    learning_rate=3e-5,
    logging_steps=1,  # Log every step to see progress
    save_strategy="no",  # Disable saving during training for speed
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    fp16=False,  # Disable fp16 since no GPU
    remove_unused_columns=False,
    gradient_accumulation_steps=2,  # Accumulate gradients for effective batch size
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

if __name__ == "__main__":
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        raise
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

    # Save model after training with retry for Windows file locking issues
    print("Saving model...")
    import time
    import shutil
    import gc
    
    temp_save_dir = "./temp_model_save"
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Force garbage collection and clear cache before saving
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Wait a moment for file handles to release
            time.sleep(1)
            
            # Save to temp directory first
            if os.path.exists(temp_save_dir):
                shutil.rmtree(temp_save_dir, ignore_errors=True)
                time.sleep(0.5)
            
            os.makedirs(temp_save_dir, exist_ok=True)
            trainer.save_model(temp_save_dir)
            processor.save_pretrained(temp_save_dir)
            
            # Move temp files to final location
            final_dir = "./Final_Trained_Model_files"
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
                time.sleep(0.5)
            
            shutil.move(temp_save_dir, final_dir)
            
            print("Model saved successfully to Final_Trained_Model_files/")
            print("Training Complete!")
            break
            
        except Exception as e:
            print(f"Save attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Failed to save model after all retries.")
                print("Error details: The model was trained successfully but couldn't be saved due to file locking.")
                print("Try closing any file explorer windows and running the script again.")
                raise