import hashlib
import pickle
import os

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def hash_dataset(df):
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()


def hash_target(dataset_hash, target_column):
    return f"{dataset_hash}_{target_column}"


def save_cache(key, data):
    with open(f"{CACHE_DIR}/{key}.pkl", "wb") as f:
        pickle.dump(data, f)


def load_cache(key):
    path = f"{CACHE_DIR}/{key}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
