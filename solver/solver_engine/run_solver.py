# run_solver.py

from solver.orchestration.master_solver import main as run_master

print("Top of run_solver.py")


def main():
    # Entry point only. No cohort/wordlist logic here.
    run_master()


if __name__ == "__main__":
    print("Entered main() in run_solver")
    main()
