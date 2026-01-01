from solver.dataset_lookup import lookup_clue


def main():
    print("Cryptic solver – dataset lookup (no API)")
    print("Ctrl+C or blank line to quit.\n")

    while True:
        clue = input("Enter clue (or blank to exit): ").strip()
        if not clue:
            break

        print("\nThinking (dataset lookup)...\n")

        result = lookup_clue(clue)

        if result is None:
            print("No close match found in dataset.\n")
        else:
            answer, explanation, original_clue, score = result
            print(f"BEST MATCH (score {score:.2f}): {original_clue}")
            print(f"ANSWER: {answer}")
            if explanation:
                print(f"EXPLANATION: {explanation}")
            print()

        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
