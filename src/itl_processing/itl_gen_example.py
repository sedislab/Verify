# This script demonstrates how to use the LTL enumeration and ITL processing utilities.
import random
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

try:
    from ltl_generation.enumerate_ltl import LTLGenerator, Formula
except ImportError as e:
    print(f"Error: Could not import from 'enumerate_ltl.py': {e}")
    print("Please ensure 'enumerate_ltl.py' contains LTLGenerator and Formula classes and is in this directory.")
    sys.exit(1)

try:
    from ltl_generation.canonicalize_ltl_with_spot import FormulaParser, validate_with_spot
except ImportError:
    print("Error: Could not import from 'canonicalize_ltl_with_spot.py'. Make sure it's in the same directory.")
    exit()

try:
    from itl_pipeline import (
        LTLParser as SpotLTLParser,
        ITLTransformer, 
        ITLParser as ITLToLTLParser,
        Verifier
    )
except ImportError as e:
    print(f"Error: Could not import from 'itl_pipeline.py': {e}")
    print("Please ensure 'itl_pipeline.py' and its dependencies (like spot) are correctly set up.")
    sys.exit(1)

def run_ltl_to_itl_example_pipeline(num_formulas: int = 10):
    """
    Generates LTL formulas, converts to LaTeX, then to Spot syntax,
    then to a custom AST, then to canonical ITL, and finally verifies the ITL.
    Outputs are printed to the console.
    """
    print(f"\n--- Starting LTL to Canonical ITL Example Pipeline for {num_formulas} formulas ---") # Announce start.

    # 1. Initialize the LTL Generator
    ltl_generator = LTLGenerator(max_depth=2, storage_dir=None)

    # 2. Initialize components from itl_pipeline.py
    spot_ltl_parser = SpotLTLParser() # Parser for LTL string -> Spot obj -> custom AST.
    itl_transformer = ITLTransformer(spot_ltl_parser) # Transformer for LTL AST -> Canonical ITL.
    itl_to_ltl_parser = ITLToLTLParser(spot_ltl_parser) # Parser for ITL string -> LTL string.
    verifier = Verifier(spot_ltl_parser, itl_to_ltl_parser) # Verifier for ITL vs LTL.
    formula_parser = FormulaParser() # Parser for LaTeX -> Spot syntax.

    for i in range(num_formulas):
        print(f"\n--- Formula #{i + 1} ---")

        # Step A: Generate an LTL Formula object using LTLGenerator
        ltl_formula_obj: Formula = ltl_generator._generate_formula(0)
        print(f"1. Generated LTL Object (internal representation might vary): {type(ltl_formula_obj)}")

        # Step B: Convert the LTL Formula object to its LaTeX string representation
        ltl_latex_str: str = ltl_formula_obj.to_latex()
        print(f"2. LTL (LaTeX format): {ltl_latex_str}")

        # Step C: Convert LaTeX LTL string to Spot LTL string syntax
        ltl_spot_syntax_str = formula_parser.parse(ltl_latex_str)
        print(f"3. LTL (Spot syntax): {ltl_spot_syntax_str}")

        # Step D: Parse the Spot LTL string into the custom AST using SpotLTLParser
        formula_data_for_ast = {"spot_formulas": ltl_spot_syntax_str, "formula": ltl_spot_syntax_str}
        try:
            ltl_ast_node = spot_ltl_parser.parse_ltl(formula_data_for_ast)
            print(f"4. LTL (Custom AST): {str(ltl_ast_node)[:100] + '...' if len(str(ltl_ast_node)) > 100 else str(ltl_ast_node)}") # Print custom AST

            # Step E: Transform the LTL AST into a Canonical ITL string
            canonical_itl_str: str = itl_transformer._generate_canonical_itl(ltl_ast_node)
            print(f"5. Canonical ITL: {canonical_itl_str}")

            # Step F: Verify the Canonical ITL against the LTL (Spot syntax)
            # The verifier will parse the ITL back to LTL and check equivalence with the original LTL.
            is_correct, verification_details = verifier.verify_itl(ltl_spot_syntax_str, canonical_itl_str)
            print(f"6. ITL Verification: Correct = {is_correct}")
            if not is_correct:
                print(f"Verification Details: {verification_details}")
        except Exception as e:
            print(f"ERROR during ITL processing: {e}")
            print(f"Skipping ITL steps for this formula.")

    print("\n--- Example Pipeline Finished ---")

if __name__ == "__main__":
    random.seed(42)

    run_ltl_to_itl_example_pipeline(num_formulas=10)