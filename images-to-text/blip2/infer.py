import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import datasets

peft_model_id = "output"
config = PeftConfig.from_pretrained(peft_model_id)
processor = Blip2Processor.from_pretrained(config.base_model_name_or_path)

model = AutoModelForVision2Seq.from_pretrained(config.base_model_name_or_path, device_map="cuda:0")
model = PeftModel.from_pretrained(model, peft_model_id)

dataset = datasets.load_from_disk("football-dataset")["train"]

item = dataset[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()

encoding = processor(images=item["image"], padding="max_length", return_tensors="pt")
encoding = {k: v.squeeze() for k, v in encoding.items()}
encoding["text"] = item["text"]
print(encoding.keys())
processed_batch = {}
for key in encoding.keys():
    if key != "text":
        processed_batch[key] = torch.stack([example[key] for example in [encoding]])
    else:
        text_inputs = processor.tokenizer(
            [example["text"] for example in [encoding]], padding=True, return_tensors="pt"
        )
        processed_batch["input_ids"] = text_inputs["input_ids"]
        processed_batch["attention_mask"] = text_inputs["attention_mask"]


pixel_values = processed_batch.pop("pixel_values").to(device, torch.float16)
print("----------")
generated_output = model.generate(pixel_values=pixel_values)
print(processor.batch_decode(generated_output, skip_special_tokens=True))
