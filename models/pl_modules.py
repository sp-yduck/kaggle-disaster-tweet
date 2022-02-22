from pytorch_lightning import LightningDataModule, LightningModule
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        self.csv_file = csv_file
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.csv_file['text'].iloc[idx]
        label = self.csv_file['target'].iloc[idx]
        input = self.tokenizer.encode_plus(
            input,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return input, label


class TweetDataModule(LightningDataModule):
    def __init__(self, tokenizer, train_csv, val_csv, test_csv, batch_size):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        if tokenizer == "bert":
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise AttributeError

    def train_dataloader(self):
        train_set = TweetDataset(
            csv_file=self.train_csv, tokenizer=self.tokenizer)
        return DataLoader(dataset=train_set, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_set = TweetDataset(
            csv_file=self.val_csv, tokenizer=self.tokenizer)
        return DataLoader(dataset=val_set, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = TweetDataset(
            csv_file=self.test_csv, tokenizer=self.tokenizer)
        return DataLoader(dataset=test_set, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True)


class NlpModule(LightningModule):
    def __init__(self, model_name, num_classes):
        super(NlpModule, self).__init__()
        self.model_name = model_name
        if model_name == "bert":
            from transformers import BertForSequenceClassification
            self.nlpmodel = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", early_stopping=False, num_labels=num_classes)
        else:
            raise AttributeError

    def forward(self, **x):
        return self.nlpmodel(**x)

    def training_step(self, batch, batch_idx):
        input, label = batch
        outputs = self.forward(
            input_ids=input["input_ids"].squeeze(),
            attention_mask=input["attention_mask"].squeeze(),
            labels=label.squeeze(),
        )
        loss = outputs.loss
        # logits = outputs.logits
        self.log("train/loss", loss, logger=True, on_epoch=True, on_step=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, label = batch
        outputs = self.forward(
            input_ids=input["input_ids"].squeeze(),
            attention_mask=input["attention_mask"].squeeze(),
            labels=label.squeeze(),
        )
        loss = outputs.loss
        # logits = outputs.logits
        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return {"batch_loss": loss}

    def test_step(self, batch, batch_idx):
        input, label = batch
        outputs = self.forward(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            labels=label
        )
        loss = outputs.loss
        logits = outputs.logits
        self.log("test/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        return {"test_loss": loss, "label": label, "logits": logits}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters())
        return optimizer
