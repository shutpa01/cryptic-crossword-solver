# check_wordlist.py

from solver_engine.resources import build_wordlist

def main():
    wl = build_wordlist()
    print(f"Total wordlist size: {len(wl)}")

    target = "color"
    if target in (w.lower() for w in wl):
        print(f"'{target}' IS present in the wordlist")
    else:
        print(f"'{target}' is NOT present in the wordlist")

if __name__ == "__main__":
    main()
