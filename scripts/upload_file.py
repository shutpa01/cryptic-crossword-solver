from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

# Load API key
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Path to the dataset
project_root = Path(__file__).resolve().parents[1]
jsonl_path = project_root / "data" / "processed" / "train.jsonl"

print("Uploading:", jsonl_path)

# Upload the file
uploaded_file = client.files.create(
    file=open(jsonl_path, "rb"),
    purpose="fine-tune"
)

print("Upload complete.")
print("File ID:", uploaded_file.id)
