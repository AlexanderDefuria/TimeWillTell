
import os

def get_hf_cache_dir():
    if "HF_HOME" in os.environ:
        print(f"Using HF_HOME from environment: {os.environ['HF_HOME']}")
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        return os.environ["HF_HOME"]
    else:
        print("HF_HOME not set, using default cache directory.")
        dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(dir, exist_ok=True)
        return dir

