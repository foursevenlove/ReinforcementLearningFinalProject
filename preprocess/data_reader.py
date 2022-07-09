import logging
import math
import os
import numpy as np
import pandas as pd
from dateutil.parser import parse
import math

from constants import *
from tools.decorators import cache_output_df
from tools.funcs import load_json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# 存储数据的根目录
# ROOT_DATA_PATH = str(Path(__file__).parent.absolute().with_name("data"))

_PROJECT_KEEP_KEY_WORDS_ = ["sub_category", "category", "entry_count", "total_awards"]
_ENTRY_KEEP_KEY_WORDS_ = ["entry_number", 'award_value', 'winner']
_CACHE_ROOT_PATH_ = f"{ROOT_DATA_PATH}/data_reader"



def get_raw_worker_quality_df():
    _df = pd.read_csv(f"{ROOT_DATA_PATH}/worker_quality.csv", dtype=np.int)
    return _df


def get_raw_project_list_df():
    _df = pd.read_csv(f"{ROOT_DATA_PATH}/project_list.csv", names=["project_id", "entry_count"], dtype=np.int)
    return _df

# def get_project_id_encode_dict():
#     project_id_encode_dict = {}
#     _df = pd.read_csv(f"{ROOT_DATA_PATH}/project_list.csv", names=["project_id", "entry_count"], dtype=np.int)
#     # 使用label encoder 编码
#     le = preprocessing.LabelEncoder()
#     project_id_list = _df["project_id"].tolist()
#     le.fit(project_id_list)
#     project_id_encode_list = list(le.transform(project_id_list))
#     project_id_encode_dict = dict(zip(project_id_list, project_id_encode_list))
#     return project_id_encode_dict

# def get_worker_id_encode_dict():
#     worker_id_encode_dict = {}
#     _df = pd.read_csv(f"{ROOT_DATA_PATH}/worker_quality.csv", dtype=np.int)
#     # 使用label encoder 编码
#     le = preprocessing.LabelEncoder()
#     worker_id_list = _df["worker_id"].tolist()
#     le.fit(worker_id_list)
#     worker_id_encode_list = list(le.transform(worker_id_list))
#     worker_id_encode_dict = dict(zip(worker_id_list, worker_id_encode_list))
#     return worker_id_encode_dict

@cache_output_df(_CACHE_ROOT_PATH_)
def get_raw_project_info_df():
    data_list = []
    for project_file_name in os.listdir(f"{ROOT_DATA_PATH}/project"):
        project_id = int(project_file_name.split("_")[1].split(".")[0])
        project_info = load_json(f"{ROOT_DATA_PATH}/project/{project_file_name}")
        project_info = {k: project_info[k] for k in _PROJECT_KEEP_KEY_WORDS_ + ["start_date", "deadline"]}
        project_info["project_id"] = project_id
        data_list.append(project_info)
    _df = pd.DataFrame(data_list)
    for date_col in ["start_date", "deadline"]:
        _df[date_col] = _df[date_col].apply(parse)
    logging.debug("sort by project start date...")
    _df.sort_values(by="start_date", ascending=True, inplace=True, ignore_index=True)
    return _df


@cache_output_df(_CACHE_ROOT_PATH_)
def get_raw_entry_info_df():
    data_list = []
    for entry_file_name in os.listdir(f"{ROOT_DATA_PATH}/entry"):
        _, project_id, _ = entry_file_name.split("_")
        project_id = int(project_id)
        entry_group = load_json(f"{ROOT_DATA_PATH}/entry/{entry_file_name}")
        for entry_result in entry_group["results"]:
            worker_id = entry_result["author"]
            score = sum([item["score"] for item in entry_result['revisions']])
            entry_result = {k: entry_result[k] for k in _ENTRY_KEEP_KEY_WORDS_ + ["entry_created_at"]}
            entry_result["project_id"] = project_id
            entry_result["worker_id"] = worker_id
            entry_result["score"] = score
            data_list.append(entry_result)
    _df = pd.DataFrame(data_list)
    _df["entry_created_at"] = _df["entry_created_at"].apply(parse)
    logging.debug("sort by entry create time ...")
    _df.sort_values(by="entry_created_at", ascending=True, inplace=True, ignore_index=True)
    return _df


@cache_output_df(_CACHE_ROOT_PATH_)
def get_worker_info_df():
    _df = pd.read_csv(f"{ROOT_DATA_PATH}/worker_quality.csv", dtype=np.int)
    _df = _df[_df.worker_quality > 0]
    _df["worker_id_encode"] = LabelEncoder().fit_transform(_df["worker_id"])
    _df = min_max_normalization(_df, col=['worker_quality'])
    return _df


@cache_output_df(_CACHE_ROOT_PATH_)
def get_project_info_df():
    project_list_df = pd.read_csv(f"{ROOT_DATA_PATH}/project_list.csv",
                                  names=["project_id", "entry_count"],
                                  dtype=np.int)
    project_id_set = set(project_list_df.project_id)
    data_list = []
    for project_file_name in os.listdir(f"{ROOT_DATA_PATH}/project"):
        project_id = int(project_file_name.split("_")[1].split(".")[0])
        if project_id not in project_id_set:
            continue
        project_info = load_json(f"{ROOT_DATA_PATH}/project/{project_file_name}")
        project_info = {k: project_info[k] for k in
                        ["sub_category", "category", "entry_count", "total_awards", "start_date", "deadline"]}
        project_info["project_id"] = project_id
        data_list.append(project_info)
    _df = pd.DataFrame(data_list)
    for date_col in ["start_date", "deadline"]:
        _df[date_col] = _df[date_col].apply(parse)
    for category_col in ['sub_category', 'category']:
        _df[category_col] = LabelEncoder().fit_transform(_df[category_col])
    _df["project_id_encode"] = LabelEncoder().fit_transform(_df["project_id"])
    _df = min_max_normalization(_df, col=['total_awards'])
    logging.debug("sort by project start date...")
    _df.sort_values(by="start_date", ascending=True, inplace=True, ignore_index=True)
    return _df


@cache_output_df(_CACHE_ROOT_PATH_)
def get_entry_info_df():
    project_id_set = set(get_project_info_df().project_id)
    worker_id_set = set(get_worker_info_df().worker_id)
    data_list = []
    for entry_file_name in os.listdir(f"{ROOT_DATA_PATH}/entry"):
        _, project_id, _ = entry_file_name.split("_")
        project_id = int(project_id)
        if project_id not in project_id_set:
            continue
        entry_group = load_json(f"{ROOT_DATA_PATH}/entry/{entry_file_name}")
        for entry_result in entry_group["results"]:
            worker_id = entry_result["author"]
            if worker_id not in worker_id_set:
                continue
            score = sum([item["score"] for item in entry_result['revisions']])
            entry_result = {k: entry_result[k] for k in ["entry_number", 'award_value', 'winner', "entry_created_at"]}
            entry_result["project_id"] = project_id
            entry_result["worker_id"] = worker_id
            entry_result["score"] = score
            data_list.append(entry_result)
    _df = pd.DataFrame(data_list)
    _df["entry_created_at"] = _df["entry_created_at"].apply(parse)
    _df["award_value"] = _df["award_value"].fillna(0.0)
    _df = min_max_normalization(_df, col=["award_value", "score"])
    logging.debug("sort by entry create time ...")
    _df.sort_values(by="entry_created_at", ascending=True, inplace=True, ignore_index=True)
    return _df

@cache_output_df(_CACHE_ROOT_PATH_)
def get_joined_data_df():
    print("df columns: \n")
    entry_df = get_raw_entry_info_df()

    project_list_df = get_raw_project_list_df()[["project_id"]]
    project_info_df = get_raw_project_info_df()
    project_info_df = project_info_df.join(project_list_df.set_index("project_id"), on="project_id", how="inner")

    worker_quality_df = get_raw_worker_quality_df()
    worker_quality_df = worker_quality_df[worker_quality_df.worker_quality > 0].copy()

    _df = entry_df.rename(columns={col: f"entry_{col}" for col in entry_df.columns if
                                   (not col.startswith("entry_") and (col not in {"worker_id", "project_id"}))})
    _df = _df.join(project_info_df.set_index("project_id").add_prefix("project_"), on="project_id", how="inner")
    _df = _df.join(worker_quality_df.set_index("worker_id"), on="worker_id", how="inner")

    _df["entry_award_value"] = _df["entry_award_value"].fillna(0.0)
    _df = min_max_normalization(_df, col=["entry_award_value", "entry_score", 'project_total_awards', 'worker_quality'])
    
    # 加入encode
    _df["worker_id_encode"] = LabelEncoder().fit_transform(_df["worker_id"])
    _df["project_id_encode"] = LabelEncoder().fit_transform(_df["project_id"])
    
    print("df columns: \n")
    print(_df.columns)
    
    return _df


def dataframe_hori_split(df, ratio=0.9):
    length = len(df)
    df = df.sort_values(by=["entry_created_at"])
    idx = math.floor(length * ratio)
    return df.iloc[:idx, :], df.iloc[idx:, :]


def min_max_normalization(df, col=["score"]):
    df.loc[:, col] = df.loc[:, col].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return df


if __name__ == '__main__':
    # os.environ["REBUILD_CACHE"] = "true"
    worker_info = get_raw_worker_quality_df()
    project_info_df = get_raw_project_info_df()
    entry_info_df = get_raw_entry_info_df()
    data_df = get_joined_data_df()
    train_data, test_data = dataframe_hori_split(data_df, 0.9)

