# Natural Language (NL) Generation Scripts

This script (`nl_generator.py`) automates the generation of natural language descriptions for Linear Temporal Logic (LTL) formulas and their Intermediate Technical Language (ITL) representations. It uses the DeepSeek API (via the OpenAI SDK) for NL generation, processes formulas in parallel using multiple worker processes, and stores the results in an SQLite database.

## Features

* **Parallel Processing**: Leverages `multiprocessing` to generate NL for multiple formulas concurrently, significantly speeding up the process for large datasets.
* **DeepSeek API Integration**: Utilizes a specified DeepSeek model (e.g., `deepseek-reasoner`) for generating domain-specific activities and NL translations.
* **Database Integration**:
    * Reads LTL formulas and their canonical ITL representations from an existing SQLite database (expected tables: `formulas`, `itl_representations`).
    * Stores the generated domain, activity context, NL translation, and metadata into a new `nl_translations` table.
    * Tracks domain distribution in a `domain_stats` table to facilitate balanced domain selection.
* **Balanced Domain Selection**: Implements a strategy to select domains for NL generation in a balanced way, aiming for roughly equal representation across the defined `DOMAINS`.
* **Robustness**: Includes signal handling for graceful shutdown, worker monitoring, and restarts for dead processes.
* **Configurable**: Uses command-line arguments to control database paths, API key location, batch sizes, number of workers, etc.

## Prerequisites

1.  **SQLite Database**: An existing database (e.g., `formulas.db`) containing:
    * A `formulas` table with LTL formulas (columns like `id`, `formula`, `spot_formulas`).
    * An `itl_representations` table with ITL data (columns like `id`, `formula_id`, `itl_text`, `canonical_form=1` for the canonical ITLs).
    This script will create/manage the `nl_translations` and `domain_stats` tables.
2.  **DeepSeek API Key**: You must have a valid API key from DeepSeek.

## Setup

1.  **Database**:
    * Ensure your SQLite database file (e.g., `formulas.db`) is accessible and populated with the required LTL and ITL data.
    * Update the `DB_PATH_MAIN` constant to point to this database file.

2.  **API Key Configuration**:
    * Create a file named `.env` in the same directory as `nl_generator.py` (or in the path you specify with the `--env` argument).
    * Add your DeepSeek API key to this file in the following format:
        ```env
        DEEPSEEK_API="your_actual_deepseek_api_key_here"
        ```

3.  **Script Constants (If modifying `main()` to not use `argparse`):**
    
    Modify the constants at the top of the file to control the generation:
    
    * Fill in the placeholder values for `PROCESSING_BATCH_SIZE_MAIN` and `NUM_WORKER_PROCESSES_MAIN` in the global constants section. 
    * Update `DB_PATH_MAIN` and `ENV_FILE_PATH` as needed in the global constants.

## Running the Script

Execute the script from your terminal.

**Example Run:**

```bash
python nl_generator.py
```