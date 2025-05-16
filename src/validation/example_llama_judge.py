# Example script to demonstrate LlamaVerifier on predefined test cases.
# This script loads the Llama model and runs verification for a few examples, printing to console.

import logging
import sys

try:
    from llm_judge import LlamaVerifier, logger
except ImportError as e:
    print(f"Error: Could not import from 'llm_judge.py': {e}")
    print("Ensure 'llm_judge.py' is in this directory and all dependencies (torch, transformers, etc.) are installed.")
    sys.exit(1)

# Test examples
TEST_EXAMPLES = [
    {
        "ltl_formula": "G(p -> Fq)",
        "itl_text": "Always, if p then eventually q",
        "nl_translation": "Whenever a transaction is initiated, it will eventually be completed.",
        "expected_correct": True
    },
    {
        "ltl_formula": "F(p & Gq)",
        "itl_text": "Eventually, p and always q thereafter",
        "nl_translation": "At some point, the system will enter maintenance mode and remain stable indefinitely.",
        "expected_correct": True
    },
    {
        "ltl_formula": "G(p -> Xq)",
        "itl_text": "Always, if p then in the next state q",
        "nl_translation": "If an error occurs, the system will eventually restart.", # Incorrect: X vs F
        "expected_correct": False
    },
    {
        "ltl_formula": "p U q",
        "itl_text": "p until q",
        "nl_translation": "The system checks for errors and sends notifications.", # Incorrect: Missing temporal
        "expected_correct": False
    }
]

def run_llama_judge_examples():
    """Loads LlamaVerifier and runs verification on predefined examples."""
    logger.info("--- Starting Llama Judge Example Script ---")

    try:
        logger.info("Initializing LlamaVerifier (this may take a while to load the model)...")
        verifier = LlamaVerifier()
    except Exception as e:
        logger.error(f"Failed to initialize LlamaVerifier: {e}", exc_info=True)
        logger.error("Please check MODEL_DIR and MODEL_ID constants in 'llm_judge.py' and model files.")
        return

    logger.info("LlamaVerifier initialized. Processing test examples...")

    for i, example in enumerate(TEST_EXAMPLES):
        logger.info(f"\n--- Verifying Example #{i + 1} ---")
        logger.info(f"LTL: {example['ltl_formula']}")
        logger.info(f"ITL: {example['itl_text']}")
        logger.info(f"NL:  {example['nl_translation']}")
        logger.info(f"Expected Correct: {example['expected_correct']}")

        result = verifier.verify_translation(
            example["ltl_formula"],
            example["itl_text"],
            example["nl_translation"]
        )

        logger.info(f"LLM Verdict: Correct = {result.get('is_correct', False)}, Score = {result.get('score', 0)}/10")
        logger.info(f"LLM Issues: {result.get('issues', [])}")
        logger.info(f"LLM Reasoning (first 150 chars): {result.get('reasoning', '')[:150]}...")
        logger.info(f"Verification Time: {result.get('verification_time', 0):.2f}s")

        matches_expected = (result.get('is_correct', False) == example['expected_correct'])
        logger.info(f"Matches Expected Outcome? {'YES' if matches_expected else 'NO'}")

    logger.info("\n--- Llama Judge Example Script Finished ---")

if __name__ == "__main__":
    run_llama_judge_examples()
