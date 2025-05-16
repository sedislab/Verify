import random

import enumerate_ltl
import canonicalize_ltl_with_spot

try:
    from enumerate_ltl import LTLGenerator, Formula, Operator
except ImportError:
    print("Error: Could not import from 'enumerate_ltl.py'. Make sure it's in the same directory.")
    exit()

try:
    from canonicalize_ltl_with_spot import FormulaParser, validate_with_spot
except ImportError:
    print("Error: Could not import from 'canonicalize_ltl_with_spot.py'. Make sure it's in the same directory.")
    exit()

def generate_and_process_formulas(num_formulas_to_generate: int = 10):
    """
    Generates a specified number of LTL formulas, converts them to LaTeX,
    then to Spot syntax, and finally verifies them using the Spot library.
    Results are printed to the console.

    Args:
        num_formulas_to_generate (int): The number of LTL formulas to generate.
    """
    print(f"--- Starting LTL Formula Generation and Processing for {num_formulas_to_generate} formulas ---")

    # Initialize the LTLGenerator from the enumerate_ltl.py script.
    # We are not using the database storage features for this example, so the storage_dir can be None.
    # We will call the internal _generate_formula method directly.

    ltl_gen = LTLGenerator(max_depth=3, storage_dir=None)

    # Initialize the FormulaParser from your ltl_spot_processor.py script.
    formula_parser = FormulaParser()

    for i in range(num_formulas_to_generate):
        print(f"\n--- Processing Formula #{i + 1} ---")

        # 1. Generate a raw LTL Formula object
        raw_formula_obj: Formula = ltl_gen._generate_formula(depth=0)
        
        print(f"Raw Formula Object: {raw_formula_obj}") # This will print its LaTeX form due to __str__

        # 2. Convert the Formula object to its LaTeX representation
        latex_formula_str = raw_formula_obj.to_latex()
        print(f"LaTeX Representation: {latex_formula_str}")

        # 3. Parse the LaTeX string into Spot syntax
        spot_syntax_formula_str = formula_parser.parse(latex_formula_str)
        print(f"Spot Syntax Attempt:  {spot_syntax_formula_str}")

        # 4. Validate the Spot syntax string using the Spot library
        is_valid, spot_canonical_form_or_error = validate_with_spot(spot_syntax_formula_str)

        print(f"Spot Verification Valid: {is_valid}")
        if is_valid:
            print(f"Spot Canonical Form:   {spot_canonical_form_or_error}")
        else:
            print(f"Spot Verification Error: {spot_canonical_form_or_error}")

    print("\n--- Finished LTL Formula Generation and Processing ---")

if __name__ == "__main__":
    random.seed(42)

    # Generate and process 10 formulas
    generate_and_process_formulas(num_formulas_to_generate=10)

    # --- Example of how you might test a specific known LaTeX formula ---
    print("\n\n--- Testing a specific LaTeX formula ---")
    specific_latex = "G {p \\rightarrow F {q}}"
    print(f"Specific LaTeX Formula: {specific_latex}")

    parser = FormulaParser()
    specific_spot_syntax = parser.parse(specific_latex)
    print(f"Spot Syntax Attempt: {specific_spot_syntax}")

    is_valid, spot_output = validate_with_spot(specific_spot_syntax)
    print(f"Spot Verification Valid: {is_valid}")

    if is_valid:
        print(f"Spot Canonical Form: {spot_output}")
    else:
        print(f"Spot Verification Error: {spot_output}")
