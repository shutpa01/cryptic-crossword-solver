# check_wordlist.py

from solver.solver_engine.resources import build_wordlist

def main():
    wl = build_wordlist()
    print(f"Total wordlist size: {len(wl)}")

    if "color" in (w.lower() for w in wl):
        print("'color' IS present in the wordlist")
    else:
        print("'color' is NOT present in the wordlist")

if __name__ == "__main__":
    main()
