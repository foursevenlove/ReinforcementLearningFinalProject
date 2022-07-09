import os

import pandas as pd

from tools.funcs import mkdir_for_file


def save_df(df, file_path, compression='snappy'):
    """
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
    """
    file_path = f"{file_path}.parquet"
    if compression:
        file_path = f"{file_path}.{compression}"
    print(f"save DataFrame to : {file_path}")
    mkdir_for_file(file_path)
    df.to_parquet(path=file_path, engine="pyarrow", compression=compression)


def load_df(file_path, compression='snappy', columns=None):
    """
    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
    """
    file_path = f"{file_path}.parquet"
    if compression:
        file_path = f"{file_path}.{compression}"
    return pd.read_parquet(path=file_path, engine="pyarrow", columns=columns)


def is_exist_df(file_path, compression='snappy'):
    file_path = f"{file_path}.parquet"
    if compression:
        file_path = f"{file_path}.{compression}"
    return os.path.exists(file_path)
