# ITL Processing Utilities

This directory contains Python scripts for the generation, processing, and verification of the Intermediate Technical Language (ITL) representations of LTL. This suite is a core component of the VERIFY dataset creation pipeline.

## Overview

The scripts provide functionalities for:
1.  Parsing LTL formulas using the Spot model-checking library to create a custom Abstract Syntax Tree (AST).
2.  Transforming the LTL AST into a rule-based, canonical ITL string.
3.  (Within `itl_pipeline.py`) Generating diverse ITL paraphrases.
4.  Parsing ITL strings back into LTL formula strings.
5.  Verifying the semantic equivalence between an ITL representation and its source LTL formula using Spot.
6.  Managing storage of LTL formulas and ITL representations in an SQLite database.

## Key Scripts

* **`itl_pipeline.py`**:
    * The main engine for ITL generation and verification.
    * `LTLParser`: Parses LTL strings (via Spot) into a custom AST.
    * `ITLTransformer`: Converts an LTL AST to a canonical ITL string using predefined templates (`LTL_TO_CANONICAL`).
    * `ITLParser`: Parses ITL strings back into LTL formula strings for verification.
    * `Verifier`: Checks semantic equivalence between an LTL formula and an ITL string (by converting ITL to LTL and using Spot).
    * `DatabaseManager`: Handles SQLite storage for ITL representations.
    * Global constants at the top for configuring paths, batch sizes, and operational modes when running this script directly.
    * Includes various helper classes and functions (e.g., `ASTNode`, `NodeType`, `latex_to_spot_syntax`).
* **`itl_gen_example.py`**:
    * The script you are currently reading about in this README.
    * Demonstrates a simplified end-to-end workflow: generating an LTL formula, converting it through various stages (LaTeX, Spot syntax, AST, Canonical ITL), and finally verifying the ITL.
    * Prints all steps to the console and does **not** use database storage, serving as a clear usage example of the library components.

## Getting Started

You can use these scripts in two primary ways:

### 1. As a Library in Your Own Scripts (Recommended for Customization)

This method allows you to import classes and functions into your custom Python projects, giving you full control over the workflow.

**Steps:**
1.  Ensure `ltl_enumerator.py` and `itl_pipeline.py` are in your Python path (e.g., in the same directory as your script).
2.  Import the components you need. For example:

    ```python
    # In your_script.py
    from ltl_enumerator import LTLGenerator, Formula # For LTL generation
    from itl_pipeline import (
        LTLParser as SpotLTLParser,
        ITLTransformer,
        ITLParser as ITLToLTLParser, # For parsing ITL back to LTL
        Verifier
    )

    # --- Example usage ---
    # Initialize LTL generator (no DB for this example)
    ltl_gen = LTLGenerator(max_depth=2, storage_dir=None)
    raw_ltl_obj = ltl_gen._generate_formula(0)
    ltl_latex = raw_ltl_obj.to_latex()
    print(f"LTL LaTeX: {ltl_latex}")

    # Initialize ITL processing components
    spot_ltl_parser_inst = SpotLTLParser()
    itl_transformer_inst = ITLTransformer(spot_ltl_parser_inst)
    # ... and so on, similar to itl_gen_example.py

    # For more detailed usage, please refer to `itl_gen_example.py`.
    ```

### 2. Modifying and Running `itl_pipeline.py` Directly

The `itl_pipeline.py` script is designed to be run as a standalone script for large-scale batch processing, database interaction, and model fine-tuning.

**Steps:**
1.  **Configure Constants**: Open `itl_pipeline.py`. At the top of the file, you'll find global constants like:
    * `DB_PATH`: **You must set this** to the path of your SQLite database (e.g., the one populated by `ltl_enumerator.py` if you ran it for database generation).
    * `VERIFY_ONLY_MODE`: Set to `True` if you only want to run the verification step on existing ITL entries in the database. Default is `False`.
    * `BATCH_SIZE_PROCESSING`: Batch size for database operations.
    * `NUM_WORKERS_PROCESSING`: Number of parallel workers.

2.  **Run the Script**: Execute from your terminal:
    ```bash
    python itl_pipeline.py
    ```
    The script will perform operations based on the configured constants (e.g., generate canonical ITL for all LTLs in the DB and then verify them).

## Running the Example Script (`itl_gen_example.py`)

This script demonstrates the core logic without needing a database or writing files.
1.  Run from the terminal:
    ```bash
    python itl_gen_example.py
    ```
    The script will generate 10 LTL formulas, show their LaTeX representation, convert them to Spot syntax, parse them into a custom AST, transform the AST to canonical ITL, and then verify the ITL against the original LTL. All steps will be printed to the console.