import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
import os

# NOTE: You will need to create a Hugging Face token and store it as a GitHub Secret.
# For example, name the secret `HF_TOKEN` in GitHub, and it will be available as an environment variable.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables. Please set HF_TOKEN.")

# Read the Hugging Face dataset repository ID from environment variable
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")
if not HF_DATASET_REPO:
    raise ValueError("Hugging Face dataset repository ID not found in environment variables. Please set HF_DATASET_REPO (e.g., 'your-username/your-dataset-name').")

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_pandas(df)

# Push to Hugging Face Hub
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
hf_dataset.push_to_hub(HF_DATASET_REPO, private=False) # Make it public by default, user can change if needed

print(f"Raw dataset successfully pushed to Hugging Face Hub: https://huggingface.co/datasets/{HF_DATASET_REPO}")
