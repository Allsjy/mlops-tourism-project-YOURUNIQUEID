import pandas as pd
from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables. Please set HF_TOKEN.")

HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
if not HF_DATASET_REPO:
    raise ValueError("Hugging Face dataset repository ID not found in environment variables. Please set HF_DATASET_REPO (e.g., 'your-username/your-dataset-name').")

api = HfApi(token=HF_TOKEN)

# Paths to the locally saved split datasets
split_files = {
    "Xtrain.csv": "Xtrain.csv",
    "Xtest.csv": "Xtest.csv",
    "ytrain.csv": "ytrain.csv",
    "ytest.csv": "ytest.csv",
}

print(f"Uploading split datasets to Hugging Face Hub: {HF_DATASET_REPO}")

for local_filename, hf_filename in split_files.items():
    if os.path.exists(local_filename):
        # The file is uploaded to the 'data_splits/' subfolder within the Hugging Face dataset repo
        api.upload_file(
            path_or_fileobj=local_filename,
            path_in_repo=f"data_splits/{hf_filename}",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Add {hf_filename} data split"
        )
        print(f"Uploaded {local_filename} to {HF_DATASET_REPO}/data_splits/{hf_filename}")
    else:
        print(f"Warning: {local_filename} not found, skipping upload.")

print("All split datasets upload process completed.")
