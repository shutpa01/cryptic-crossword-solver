# solver/orchestration/output_presets.py

def print_summary(analyser):
    analyser.print_report()


def active_output(analyser):
    # Population summary only. No provenance / per-candidate output here.
    print_summary(analyser)
