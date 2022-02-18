import os
import numpy as np

import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW


class TweetDataset(Dataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.csv_file['text'].iloc[idx]
        output = self.csv_file['target'].iloc[idx]
        print(input, output)
        return input, output


class TweetDataModule(LightningDataModule):
    def __init__(self, train_csv, val_csv, test_csv, batch_size):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size

    def train_dataloader(self):
        train_set = TweetDataset(csv_file=self.train_csv)
        return DataLoader(dataset=train_set, num_workers=4, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        val_set = TweetDataset(csv_file=self.val_csv)
        return DataLoader(dataset=val_set, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = TweetDataset(csv_file=self.test_csv)
        return DataLoader(dataset=test_set, num_workers=4, batch_size=self.batch_size, shuffle=False, pin_memory=True)


class NlpModule(LightningModule):
    def __init__(self, model_name):
        super(NlpModule, self).__init__()
        self.loss = MSELoss()
        self.model_name = model_name
        if model_name == "resnet":
            self.nlpmodel = ResNet34(out_dim=5)
        elif model_name == "unet":
            self.nlpmodel = UNet(n_channels=3, n_classes=1)

    def forward(self, x):
        return self.nlpmodel(x)

    def training_step(self, batch, batch_idx):
        input, output = batch
        pred = self.forward(input)
        loss = self.loss(output, pred.squeeze())
        self.log("train/loss", loss, logger=True, on_epoch=True, on_step=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input, output = batch
        pred = self.forward(input)
        loss = self.loss(output, pred.squeeze())
        self.log("val/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return {"batch_loss": loss}

    def test_step(self, batch, batch_idx):
        input, output = batch
        pred = self.forward(input)
        loss = self.loss(output, pred.squeeze())
        self.log("test/loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return {"test_loss": loss, "output": output, "pred": pred}

    def test_epoch_end(self, outputs):
        output_df = pd.DataFrame(columns=[])
        preds = torch.cat([x["pred"] for x in outputs], dim=0).squeeze()
        preds = np.array(preds.tolist())
        labels = torch.cat([x["output"] for x in outputs], dim=0).squeeze()
        labels = np.array(labels.tolist())
        os.makedirs('outputs', exist_ok=True)
        output_df.to_csv("outputs/output.csv")
        return

    def configure_optimizers(self):
        optimizer = AdamW()
        return optimizer