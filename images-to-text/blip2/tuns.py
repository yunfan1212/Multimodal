
import torch
import datasets

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse

from peft import LoraConfig, get_peft_model
import os



class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--pretrain-model-path", dest="pretrain_model_path", required=False, type=str, default="/home/blip2-opt-2.7b",
                        help="预训练模型路径")
    parser.add_argument("--output-path", type=str, default="output", help="模型输出路径")

    args = parser.parse_args()

    output_path = args.output_path
    pretrain_model_path = args.pretrain_model_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    peft_model_id = output_path

    # We load our model and processor using `transformers`    加载模型
    model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path)
    processor = AutoProcessor.from_pretrained(pretrain_model_path)

    # Let's define the LoraConfig
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    # Get our peft model and print the number of trainable parameters
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.cuda()

    # Let's load the dataset here!
    # dataset = load_dataset("ybelkada/football-dataset", split="train")

    def collator(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch


    dataset = datasets.load_from_disk("football-dataset")["train"]

    train_dataset = ImageCaptioningDataset(dataset, processor)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()
    loss_list = []
    for epoch in range(11):
        print("Epoch:", epoch)
        sum_loss_list = []
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)
            #语言模型输入=输出
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

            loss = outputs.loss

            print("Loss:", loss.item())

            sum_loss_list.append(float(loss.item()))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:
                generated_output = model.generate(pixel_values=pixel_values)
                print(processor.batch_decode(generated_output, skip_special_tokens=True))

        avg_sum_loss = sum(sum_loss_list) / len(sum_loss_list)
        print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
        loss_list.append(float(avg_sum_loss))

    if not os.path.exists(peft_model_id):
        os.makedirs(peft_model_id)

    print("model_output:", peft_model_id)
    model.save_pretrained(peft_model_id)



if __name__ == "__main__":
    main()