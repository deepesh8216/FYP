import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import os

class FakeNewsDataset(Dataset):
    def __init__(
        self,
        csv_path,
        max_text_len=512,
        img_size=224,
        augment=False,
        use_text=True,
        use_image=True
    ):
        self.df = pd.read_csv(csv_path)
        self.max_text_len = max_text_len
        self.img_size = img_size
        self.use_text = use_text
        self.use_image = use_image

        # ── Text tokenizer ───────────────────────────────
        self.tokenizer = None
        if self.use_text:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased'
            )

        # ── Image transforms ─────────────────────────────
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomCrop(img_size, padding=16),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.3, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.2),
            ])
        else:
            # Val/Test: no augmentation
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Text ─────────────────────────────────────────
        if self.use_text:
            encoding = self.tokenizer(
                str(row['text']),
                max_length=self.max_text_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            # Dummy tensors keep batch shape consistent for modality-specific runs.
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        # ── Image ─────────────────────────────────────────
        if self.use_image:
            try:
                img = Image.open(row['image_path']).convert('RGB')
                img = self.transform(img)
            except Exception:
                # If image fails, use blank tensor.
                img = torch.zeros(3, self.img_size, self.img_size)
        else:
            img = torch.zeros(3, self.img_size, self.img_size)

        # ── Label ─────────────────────────────────────────
        label = torch.tensor(int(row['label']), dtype=torch.long)

        return {
            'input_ids'      : input_ids,
            'attention_mask' : attention_mask,
            'image'          : img,
            'label'          : label
        }