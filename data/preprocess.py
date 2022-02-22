import os
import pandas as pd
from sklearn.model_selection import KFold


def get_one_fold(data_csv, seed):
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(data_csv)):
        data_csv.loc[tr_idx, "type"] = 1
        data_csv.loc[val_idx, "type"] = 0
    train_csv = data_csv[data_csv["type"] == 1].drop("type", axis=1)
    val_csv = data_csv[data_csv["type"] == 0].drop("type", axis=1)
    return train_csv.reset_index(drop=True), val_csv.reset_index(drop=True)


def split_dataset(file_path, save_for_dir, seed=1234):
    """
    Ratio of the number of data
    train : val : test = 64 : 16 : 20
    """
    os.makedirs(save_for_dir, exist_ok=True)
    data_df = pd.read_csv(file_path)
    train_df, test_csv = get_one_fold(data_df, seed)
    train_csv, val_csv = get_one_fold(train_df, seed)
    train_csv.to_csv(save_for_dir+"train.csv", index=False)
    val_csv.to_csv(save_for_dir+"val.csv", index=False)
    test_csv.to_csv(save_for_dir+"test.csv", index=False)


if __name__ == "__main__":
    split_dataset(
        file_path="data/nlp-getting-started/train.csv",
        save_for_dir="data/inputs/",
        seed=1234,
    )
