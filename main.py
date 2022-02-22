import os
import pandas as pd
import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models.pl_modules import TweetDataModule, NlpModule


def main(cfg, train_step=False, test_step=False):
    new_cwd = cfg.RUN.DIR
    os.makedirs(new_cwd, exist_ok=True)
    os.chdir(new_cwd)

    train_csv = pd.read_csv(cfg.DATASET.ROOT + "train.csv")
    val_csv = pd.read_csv(cfg.DATASET.ROOT + "val.csv")
    test_csv = pd.read_csv(cfg.DATASET.ROOT + "test.csv")

    datamodule = TweetDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        tokenizer=cfg.TRAIN.MODEL,
    )

    if train_step:
        trainmodule = NlpModule(cfg.TRAIN.MODEL, num_classes=2)
        tb_logger = pl_loggers.TensorBoardLogger("logs/")
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss", mode="min", save_top_k=5, save_last=True)
        trainer = Trainer(
            max_epochs=cfg.TRAIN.NUM_EPOCHS,
            gpus=cfg.TRAIN.NUM_GPUS,
            logger=tb_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )
        trainer.fit(trainmodule, datamodule)
        print("train step complete")

    if test_step:
        MODEL_PATH = f"logs/default/version_{cfg.TEST.VERSION}/checkpoints/{cfg.TEST.CKPT}"
        testmodule = NlpModule(cfg.TRAIN.MODEL, num_classes=2).load_from_checkpoint(
            checkpoint_path=MODEL_PATH, model_name=cfg.TRAIN.MODEL, num_classes=2)
        trainer = Trainer(gpus=cfg.TRAIN.NUM_GPUS, logger=False)
        trainer.test(model=testmodule, datamodule=datamodule,
                     ckpt_path=MODEL_PATH, verbose=True)
        print("test step complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    cfg = OmegaConf.load(parser.parse_args().cfg)
    train_step = parser.parse_args().train
    test_step = parser.parse_args().test

    main(
        cfg=cfg,
        train_step=train_step,
        test_step=test_step,
    )
