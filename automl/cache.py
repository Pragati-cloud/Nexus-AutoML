import hashlib
import os
import pickle

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_hash(df, target_column):
    data_bytes = df.to_csv(index=False).encode()
    return hashlib.md5(data_bytes + target_column.encode()).hexdigest()


def save_cache(key, data):
    with open(f"{CACHE_DIR}/{key}.pkl", "wb") as f:
        pickle.dump(data, f)


def load_cache(key):
    path = f"{CACHE_DIR}/{key}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
import hashlib
import pickle
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_hash(df, target_column):
    data_bytes = df.to_csv(index=False).encode()
    return hashlib.md5(data_bytes + target_column.encode()).hexdigest()


def save_cache(key, data):
    with open(f"{CACHE_DIR}/{key}.pkl", "wb") as f:
        pickle.dump(data, f)


def load_cache(key):
    path = f"{CACHE_DIR}/{key}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None