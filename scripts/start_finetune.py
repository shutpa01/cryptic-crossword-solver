from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

training_file_id = "file-WwHj6fKuLe3QdAy7kgeieG"

print("Starting fine-tune using file:", training_file_id)

job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini-tts",
    training_file=training_file_id,
    suffix="cryptic_solver_v1"
)

print("Fine-tune job started.")
print("Job ID:", job.id)
