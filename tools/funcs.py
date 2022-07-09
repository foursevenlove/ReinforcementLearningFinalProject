import hashlib
import os
import pickle
import json


def consistent_hash(*objs, code_len=6):
    return hashlib.md5(pickle.dumps(objs)).hexdigest()[:code_len]


def mkdir_for_file(file_path):
    _dir = os.path.dirname(file_path)
    _dir and os.makedirs(_dir, exist_ok=True)


def load_json(file_path):
    with open(file_path, "r", encoding='UTF-8') as _r:
        json_str = _r.read()
        return json.loads(json_str)
