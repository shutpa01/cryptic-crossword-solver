import os
from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """
    Load API key from .env and return an OpenAI client.
    """
    # Load from .env in project root
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Check your .env file in the project root.")

    return OpenAI(api_key=api_key)


def solve_clue(clue: str) -> str:
    """
    Send a single cryptic clue to the model and return a formatted answer + explanation string.
    """
    client = get_client()

    system_prompt = (
        "You are an expert British cryptic crossword solver.\n"
        "Given a single cryptic clue (with enumeration in brackets if present), "
        "you MUST respond in this exact format:\n\n"
        "ANSWER: <answer in UPPERCASE, no enumeration>\n"
        "EXPLANATION: <brief but precise parse, showing definition and wordplay>\n\n"
        "Rules:\n"
        "- Always identify the definition part.\n"
        "- Explain the wordplay (anagram, charade, container, homophone, etc.).\n"
        "- If you are unsure, still give your best guess but say 'Uncertain' in the explanation.\n"
    )

    user_content = f"Clue: {clue}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=200,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


def main():
    print("Cryptic solver – baseline (Ctrl+C to quit)\n")
    while True:
        try:
            clue = input("Enter clue (or blank to exit): ").strip()
            if not clue:
                break
            print("\nThinking...\n")
            result = solve_clue(clue)
            print(result)
            print("\n" + "-" * 40 + "\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
