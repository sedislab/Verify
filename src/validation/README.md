# LTL/ITL/NL Verification using Llama 3.3

This script (`llm_judge.py`) leverages a Large Language Model (LLM), specifically Llama 3.3 (e.g., `meta-llama/Llama-3.3-70B-Instruct`), to act as a judge for assessing the semantic correctness of Natural Language (NL) translations against their corresponding Linear Temporal Logic (LTL) formulas and Intermediate Technical Language (ITL) representations.

## Overview

The system performs the following key operations:
1.  Loads a specified Llama 3.3 model with 8-bit quantization for efficiency.
2.  Connects to an SQLite database containing LTL formulas, ITL representations, and their NL translations.
3.  Selects a random percentage (e.g., 18%) of the NL translations that have not yet been verified.
4.  For each selected entry, it prompts the Llama model to evaluate if the NL translation accurately reflects the meaning of the LTL/ITL, paying close attention to temporal semantics.
5.  Parses the LLM's structured JSON output (containing `is_correct`, `score`, `issues`, `reasoning`).
6.  Stores these verification results back into the database in a `verification_results` table.
7.  Calculates and stores overall verification statistics (e.g., correctness percentage, average score) in a `verification_statistics` table and also saves them to a CSV file.
8.  Includes a test mode to run the verifier on a predefined set of examples.

## Features

* **LLM-as-a-Judge**: Uses Llama 3.3 to perform semantic verification.
* **Quantized Model Loading**: Loads the Llama model using 8-bit quantization via `bitsandbytes` for reduced memory footprint.
* **Database Integration**: Reads data from `formulas`, `itl_representations`, `nl_translations` tables and writes to `verification_results`, `verification_statistics`.
* **Selective Sampling**: Verifies a configurable percentage of the available NL translations.
* **Structured Output Parsing**: Parses JSON responses from the LLM.
* **Comprehensive Statistics**: Calculates and stores detailed statistics about the verification process.
* **Test Mode**: Allows for quick checks on predefined examples.

## Prerequisites

1.  **SQLite Database**: An existing database (e.g., `formulas.db`) populated with:
    * `formulas` table (with `id`, `formula`, `spot_formulas`).
    * `itl_representations` table (with `id`, `formula_id`, `itl_text`).
    * `nl_translations` table (with `id`, `formula_id`, `itl_id`, `domain`, `activity`, `translation`).
    The script will create `verification_results` and `verification_statistics` tables.
4.  **Llama 3.3 Model Access**:
    * The script will attempt to download the model specified by `MODEL_ID` (e.g., `meta-llama/Llama-3.3-70B-Instruct`) from Hugging Face Hub and cache it in `MODEL_DIR`.
    * Ensure you have accepted the model license on Hugging Face and are logged in via `huggingface-cli login` if necessary.
    * Alternatively, if you have the model downloaded locally, point `MODEL_DIR` to its location.

## Setup

1.  **Model Configuration**:
    * Open `llm_judge.py`.
    * Modify the global constant `MODEL_DIR` to specify the directory where the Llama model should be downloaded/cached (or where it already exists).
2.  **Database Path**:
    * Modify the global constant `DB_PATH_VERIFICATION`. **This is required if `RUN_TEST_EXAMPLES_MODE` is `False`**. Set it to the full path of your SQLite database file.
3.  **Results Directory**:
    * Modify `RESULTS_DIR` if you want verification statistics CSV files saved to a different location. The script will create this directory if it doesn't exist.
4.  **Other Constants**: Review and adjust other global constants at the top of `llm_judge.py` as needed:
    * `VERIFICATION_PERCENTAGE`: Percentage of translations to sample for verification.
    * `RUN_TEST_EXAMPLES_MODE`: Set to `True` to run only the built-in test examples without DB interaction.
    * `BATCH_SIZE_VERIFICATION`: Batch size for processing entries from the DB (though the current DB processing loop is serial per entry for LLM calls).

## Running the Script

Once the constants (especially `DB_PATH_VERIFICATION`, `MODEL_DIR`, `MODEL_ID`) are correctly set in `llm_judge.py`:

1.  **To run on predefined test examples (no DB needed, good for initial setup check):**
    * Set `RUN_TEST_EXAMPLES_MODE = True` in `llm_judge.py`.
    * Execute: `python llm_judge.py`

2.  **To run full database verification:**
    * Set `RUN_TEST_EXAMPLES_MODE = False`.
    * Ensure `DB_PATH_VERIFICATION` points to your populated database.
    * Execute: `python llm_judge.py`

The script will:
* Load the Llama 3.3 model (this can take significant time and GPU memory, especially for the 70B model).
* If not in test mode, connect to the database and select a sample of NL translations.
* Iterate through the sample, prompting the Llama model for each.
* Store verification results in the database.
* Log progress to the console and to `verification.log`.
* Save final aggregate statistics to the database and a CSV file in the `RESULTS_DIR`.

## Output

* **Database**:
    * `verification_results`: Stores detailed judgments for each verified NL translation.
    * `verification_statistics`: Stores summary statistics for each verification run.
* **CSV File**: A CSV file (e.g., `RESULTS_DIR/verification_stats_YYYYMMDD_HHMMSS.csv`) containing the summary statistics.

## Example Usage (`example_llama_judge.py`)

An example script, `example_llama_judge.py`, is provided to demonstrate direct usage of the `LlamaVerifier` class. This script:
* Loads the Llama model using the `MODEL_DIR` and `MODEL_ID` constants from `llm_judge.py`.
* Runs verification on a small, hardcoded set of LTL/ITL/NL examples.
* Prints detailed results to the console.
* It does **not** interact with the database.

This example is useful for quickly testing if the model loads correctly and how the verification prompt and response parsing work, without needing the full database setup.

To run it:
1.  Ensure `llm_judge.py` is in the same directory (as it's imported).
2.  Make sure `MODEL_DIR` and `MODEL_ID` are correctly configured in `llm_judge.py`.
3.  Execute: `python example_llama_judge.py`
