from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env so OPENAI_API_KEY is available
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file or environment variables.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are a cryptic crossword explainer. "
    "Given a cryptic clue and its answer, you explain how the clue works.\n\n"
    "Always use this format:\n"
    "Answer: <ANSWER>\n"
    "Definition: <short definition from the clue>\n"
    "Wordplay: <step-by-step breakdown of the wordplay>\n\n"
    "Be concise (2–4 short sentences) and use standard cryptic terminology "
    "like 'anagram', 'container', 'charade', 'double definition', etc."
)


def explain_with_ai(clue: str, answer: str) -> str:
    """
    Ask the OpenAI API to explain how the cryptic clue works,
    given the clue text and the known answer.
    """
    user_prompt = (
        f"Explain this cryptic crossword clue.\n\n"
        f"Clue: {clue}\n"
        f"Answer: {answer}\n\n"
        f"Remember to identify the definition and explain the wordplay."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    # NEW: use .content instead of ["content"] to avoid the TypeError
    return response.choices[0].message.content.strip()
