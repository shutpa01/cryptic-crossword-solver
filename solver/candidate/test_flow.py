
import sys
import os

# Add project root so "solver" becomes importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(PROJECT_ROOT)
sys.path.append(PARENT)

from solver.definition.definition_model import extract_definition_tokens

clue = "Your setter, beind pub, gets italian port"
expected_definition = "novelist"

tokens = extract_definition_tokens(clue)
predicted_definition = " ".join(tokens)

print("CLUE:", clue)
print("EXPECTED:", expected_definition)
print("TOKENS:", tokens)
print("PREDICTED:", predicted_definition)

if expected_definition in predicted_definition:
    print("RESULT: CORRECT")
else:
    print("RESULT: INCORRECT")
