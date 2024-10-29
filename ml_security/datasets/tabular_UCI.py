"""
    Adapted from https://github.com/yu-rp/KANbeFair/blob/main/src/data/uciml.py
"""

import warnings

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ml_security.logger import logger

warnings.filterwarnings("ignore")


def process_x(df: pd.DataFrame):
    """
    Process a DataFrame for machine learning model input.

    This function performs various preprocessing steps on the input DataFrame:
    - Identifies and retains float, integer, and string columns while dropping others.
    - Converts string columns to lowercase and strips unwanted characters.
    - Encodes categorical string features into integer representations.
    - Converts integer and categorical columns to float.
    - Fills missing values with zeros.
    - Standardizes numerical columns using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to process. It can contain mixed data types including
        float, int, and object (string) types.

    Returns
    -------
    np.ndarray
        A numpy array of the processed DataFrame, suitable for use as input to machine learning models.

    Notes
    -----
    - All non-float, non-int, and non-string columns are dropped.
    - Categorical features are encoded such that each unique value is assigned a unique integer.
    - The final output is a 2D numpy array with all features standardized.
    - Missing values are filled with zero prior to standardization.
    """
    float_cols = [col for col in df.columns if df[col].dtype == "float64"]
    int_cols = [col for col in df.columns if df[col].dtype == "int64"]
    str_cols = [col for col in df.columns if df[col].dtype == "object"]
    other_cols = [
        col for col in df.columns if col not in float_cols + int_cols + str_cols
    ]

    df = df.drop(other_cols, axis=1)

    for col in str_cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip("!\"#%&'()*,./:;?@[\\]^_`{|}~" + " \n\r\t")

    category_feature_map = [{} for cat in str_cols]

    def encode(encoder, x):
        len_encoder = len(encoder)
        try:
            id = encoder[x]
        except KeyError:
            id = len_encoder
        return id

    for i, cat in enumerate(str_cols):
        category_feature_map[i] = {
            l: id for id, l in enumerate(df.loc[:, cat].astype(str).unique())
        }
        df[cat] = (
            df[cat].astype(str).apply(lambda x: encode(category_feature_map[i], x))
        )

    for col in int_cols + str_cols:
        df[col] = df[col].astype(float)

    df[float_cols + int_cols + str_cols] = df[float_cols + int_cols + str_cols].fillna(
        0
    )

    scaler = StandardScaler()
    df[float_cols + int_cols + str_cols] = scaler.fit_transform(
        df[float_cols + int_cols + str_cols]
    )

    return df.values


def process_y(df):
    """
    Process the target variable DataFrame for machine learning model input.

    This function preprocesses a DataFrame that is expected to contain a single column
    representing the target variable. It handles both categorical and integer types,
    converts categorical string values to a standardized integer representation, and
    ensures that all string values are cleaned by removing unwanted characters and
    converting to lowercase.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing a single column which represents the target variable.
        The column can be of type `int64` or `object` (string).

    Returns
    -------
    np.ndarray
        A numpy array of the processed target variable, suitable for use as input
        to machine learning models.

    Raises
    ------
    AssertionError
        If the input DataFrame does not contain exactly one column or if the column
        data type is neither `int64` nor `object`.

    Notes
    -----
    - The function cleans the string values by converting them to lowercase and stripping
      unwanted characters.
    - Categorical features are encoded such that each unique value is assigned a unique integer.
    - If a value is not present in the encoding map, it is assigned an index equal to the
      current size of the map, effectively treating it as a new category.
    """

    assert len(df.columns) == 1
    assert df.dtypes[0] in ["int64", "object"]

    if df.dtypes[0] == "object":
        df[df.columns[0]] = df[df.columns[0]].str.lower()
        df[df.columns[0]] = df[df.columns[0]].str.strip(
            "!\"#%&'()*,./:;?@[\\]^_`{|}~" + " \n\r\t"
        )

    def encode(encoder, x):
        len_encoder = len(encoder)
        try:
            id = encoder[x]
        except KeyError:
            id = len_encoder
        return id

    feature_map = {
        l: id for id, l in enumerate(df.loc[:, df.columns[0]].astype(str).unique())
    }
    df[df.columns[0]] = (
        df[df.columns[0]].astype(str).apply(lambda x: encode(feature_map, x))
    )

    return df.values


def construct_dataset(x, y):
    """
    Construct a PyTorch dataset from feature and target tensors.

    This function converts input feature data and target labels into a
    PyTorch `TensorDataset`, which can be used with DataLoader for
    training machine learning models.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        A tensor or array-like structure containing the feature data.
        This will be converted to a PyTorch tensor of type `float32`.

    y : np.ndarray or torch.Tensor
        A tensor or array-like structure containing the target labels.
        This will be converted to a PyTorch tensor of type `int64`.
        The shape of `y` should be either 1D or 2D, where if it is 2D,
        it must have a shape of (N, 1) or (1, N).

    Returns
    -------
    TensorDataset
        A PyTorch `TensorDataset` containing the feature and target tensors.

    Raises
    ------
    ValueError
        If `y` is neither 1D nor 2D with the specified shape constraints.

    Notes
    -----
    - The function will squeeze the target tensor `y` to ensure it is 1D
      if it is a 2D tensor with only one column or one row.
    - If `y` does not meet the shape requirements, a ValueError will be raised,
      providing feedback on the expected shape.
    """

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    if len(y.shape) == 2:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
        else:
            raise ValueError("y should be 1D or 2D with shape[1]=1 or shape[0]=1.")
    elif len(y.shape) == 1:
        pass
    else:
        raise ValueError("y should be 1D or 2D with shape[1]=1 or shape[0]=1.")

    dataset = TensorDataset(x, y)

    return dataset


def split_dataset(dataset, split_ratio: float = 0.8):
    """
    Splits a dataset into training and testing subsets.

    This function divides the given dataset into two parts: a training dataset
    and a testing dataset, based on the specified split ratio. The training dataset
    will contain a specified percentage of the total samples, while the rest will
    be allocated to the testing dataset.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset object to be split. This should be an instance of
        `torch.utils.data.Dataset`, containing the samples to be divided.

    split_ratio : float, optional
        The ratio of the dataset to be used for training. Must be between 0 and 1.
        Default is 0.8, meaning 80% of the dataset will be used for training and
        20% for testing.

    Returns
    -------
    tuple
        A tuple containing two datasets: the training dataset and the testing dataset.

    Raises
    ------
    ValueError
        If `split_ratio` is not between 0 and 1.

    Notes
    -----
    - The function uses `torch.utils.data.random_split` to perform the split,
      which ensures randomness in selecting samples for each subset.
    - If the split ratio results in non-integer sizes, the function will round
      down for the training dataset size, leading to a slight imbalance.
    """
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def from_uciml_to_dataset(uci_id: int, split_ratio: float):
    """
    Fetches a dataset from the UCI Machine Learning Repository and processes it
    into a PyTorch dataset.

    This function retrieves a dataset using its UCI ID, processes the features
    and targets, and then splits the dataset into training and testing subsets
    based on the provided split ratio.

    Parameters
    ----------
    uci_id : int
        The unique identifier for the dataset in the UCI Machine Learning Repository.
        This ID is required to fetch the specific dataset.

    split_ratio : float
        The ratio of the dataset to be used for training. Must be between 0 and 1.
        A value of 0.8 indicates that 80% of the dataset will be used for training
        and 20% for testing.

    Returns
    -------
    tuple
        A tuple containing two PyTorch datasets:
        - The training dataset.
        - The testing dataset.

    Raises
    ------
    ImportError
        If the `ucimlrepo` package is not installed, an ImportError will be raised,
        prompting the user to install the package.

    ValueError
        If `split_ratio` is not between 0 and 1, a ValueError will be raised during
        the dataset splitting process.

    Notes
    -----
    - This function relies on the `ucimlrepo` package to fetch datasets from the UCI
      repository. Ensure that this package is installed in your Python environment.
    - The processing functions `process_x` and `process_y` are used to prepare
      the features and targets respectively before constructing the PyTorch datasets.
    - The function uses the `split_dataset` function to divide the constructed
      dataset into training and testing subsets based on the provided split ratio.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        logger.error(
            'ucimlrepo not installed. Please install it by running "pip install ucimlrepo"'
        )
        raise ImportError(
            'ucimlrepo not installed. Please install it by running "pip install ucimlrepo"'
        )
    ucimldataset = fetch_ucirepo(id=uci_id)

    x = ucimldataset.data.features
    y = ucimldataset.data.targets

    x = process_x(x)
    y = process_y(y)

    dataset = construct_dataset(x, y)
    train_dataset, test_dataset = split_dataset(dataset, split_ratio=split_ratio)
    return train_dataset, test_dataset
