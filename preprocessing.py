import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from Util import setseed


def data_processing(dataset_name, subset_name, data_type=torch.float32):
    """
    Preprocess time-series dataset for anomaly detection.

    Parameters:
    ----------
    dataset_name : str
        Name of the dataset folder under ./Dataset/
    subset_name : str
        Filename of the test split (e.g., "test", "subset_A")
    data_type : torch.dtype
        Data type for the output tensors (default: torch.float32)

    Returns:
    -------
    Normalized Data

    Notes:
    -----
    - This function assumes a directory structure like:
        Dataset/{dataset_name}/
            ├── train.csv
            └── {subset_name}.csv

    - The anomaly labels are stored in a column named "Attack_label" (0 for normal and 1 for anomaly).
    - You can customize this function for different dataset formats, as long as
      the output tensors are standardized (shape & type) for downstream models.
    """

    setseed(42)

    train_df = pd.read_csv(f'Dataset\\{dataset_name}\\train.csv', low_memory=False)
    test_df = pd.read_csv(f'Dataset\\{dataset_name}\\{subset_name}.csv', low_memory=False)

    train_labels = train_df['Attack_label']
    test_labels = test_df['Attack_label']

    train_df = train_df.drop(columns=['Attack_label'])
    test_df = test_df.drop(columns=['Attack_label'])

    train_df, test_df = norm(train_df, test_df)

    train_data = torch.tensor(train_df, dtype=data_type)
    test_data = torch.tensor(test_df, dtype=data_type)

    train_labels = torch.tensor(train_labels.values, dtype=data_type)
    test_labels = torch.tensor(test_labels.values, dtype=data_type)

    return train_data, train_labels, test_data, test_labels


def norm(train, test):
    normalizer = StandardScaler().fit(train)
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret
