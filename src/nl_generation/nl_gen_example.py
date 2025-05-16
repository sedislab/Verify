# Example script to demonstrate direct usage of DeepSeekGenerator for a few LTL/ITL pairs.
# This script does NOT interact with a database for its core demonstration.
import os
from dotenv import load_dotenv
import sys
import random
from typing import Dict
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

try:
    from nl_generator import DatabaseManager, DeepSeekGenerator, DOMAINS
except ImportError as e:
    print(f"Error: Could not import from 'nl_generator.py': {e}")
    print("Ensure 'nl_generator.py' is in the same directory and all its dependencies are met.")
    sys.exit(1)

EXAMPLE_ENV_FILE = ".env.example"

# Sample LTL formulas and their corresponding ITL representations
SAMPLE_FORMULAS = [
    {
        "id": 1,
        "formula": "G(req -> F(ack))",
        "spot_formulas": "G(req -> F(ack))",
        "itl_id": 101,
        "itl_text": "Always, if request, then Eventually, acknowledge"
    },
    {
        "id": 2,
        "formula": "F(G(error))",
        "spot_formulas": "F(G(error))",
        "itl_id": 102,
        "itl_text": "Eventually, Always, error"
    },
    {
        "id": 3,
        "formula": "p U q",
        "spot_formulas": "p U q",
        "itl_id": 103,
        "itl_text": "p until q"
    }
]

# The DeepSeekGenerator expects a DatabaseManager for domain selection.
# We'll create a simplified mock that doesn't need a real database for this example.
class MockDatabaseManager:
    def __init__(self, worker_id=0):
        self.worker_id = worker_id
        self.domain_counts = {domain: random.randint(0,5) for domain in DOMAINS}

    def get_domain_distribution(self) -> Dict[str, int]:
        return self.domain_counts

    def update_domain_count(self, domain: str) -> None:
        if domain in self.domain_counts:
            self.domain_counts[domain] += 1
        print(f"(MockDB) Updated count for domain: {domain}")

    def close_connection(self):
        pass

def run_nl_generation_example():
    """Demonstrates generating NL for a few sample LTL/ITL pairs."""
    print("--- Starting Natural Language Generation Example ---")

    if not os.path.exists(EXAMPLE_ENV_FILE):
        print(f"Error: Environment file '{EXAMPLE_ENV_FILE}' not found.")
        print(f"Please create it with your DEEPSEEK_API key (e.g., DEEPSEEK_API='your_key').")
        return
    load_dotenv(EXAMPLE_ENV_FILE)
    api_key = os.getenv("DEEPSEEK_API")

    if not api_key:
        print(f"Error: DEEPSEEK_API key not found in '{EXAMPLE_ENV_FILE}'.")
        return

    # Initialize the mock DatabaseManager and DeepSeekGenerator
    mock_db_manager = MockDatabaseManager()
    nl_generator = DeepSeekGenerator(api_key=api_key, db_manager=mock_db_manager)

    for i, formula_dict in enumerate(SAMPLE_FORMULAS):
        print(f"\n--- Processing Sample Formula #{i + 1} ---")
        print(f"LTL: {formula_dict.get('spot_formulas', formula_dict['formula'])}")
        print(f"ITL: {formula_dict['itl_text']}")

        # Generate natural language translation
        # The generate_translation method expects a dictionary similar to what db_manager.get_complex_formulas returns.
        generated_data = nl_generator.generate_translation(formula_dict)

        if generated_data:
            print(f"Selected Domain: {generated_data['domain']}")
            print(f"Generated Activity: {generated_data['activity']}")
            print(f"Generated NL Translation: {generated_data['translation']}")
            print(f"API Generation Time: {generated_data.get('generation_time', 'N/A'):.2f}s")
        else:
            print("Failed to generate translation for this sample.")

    print("\n--- Natural Language Generation Example Finished ---")

if __name__ == "__main__":
    run_nl_generation_example()
