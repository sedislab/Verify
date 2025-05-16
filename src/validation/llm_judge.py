import os
import sys
import time
import json
import random
import sqlite3
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch
import transformers
import bitsandbytes as bnb
import accelerate
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('verification.log')
    ]
)
logger = logging.getLogger('llama_verification')

# Constants
VERIFICATION_PERCENTAGE = 18.0
MODEL_DIR = "/path/to/local/llama33"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
RESULTS_DIR = "/path/to/results"
RUN_TEST_EXAMPLES_MODE = False  # Set to True to run test examples
DB_PATH_VERIFICATION = None # Update this path if not running in test mode to "/path/to/your/verification_database.db"
BATCH_SIZE_VERIFICATION = 20

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RESULTS_DIR, exist_ok=True)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str):
        """Initialize database manager."""
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self) -> None:
        """Set up verification tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verification_results (
                    id INTEGER PRIMARY KEY,
                    formula_id INTEGER NOT NULL,
                    itl_id INTEGER NOT NULL,
                    translation_id INTEGER NOT NULL,
                    is_correct BOOLEAN,
                    score INTEGER,
                    issues TEXT,
                    reasoning TEXT,
                    verification_time REAL,
                    verification_timestamp TEXT,
                    FOREIGN KEY (formula_id) REFERENCES formulas(id),
                    FOREIGN KEY (itl_id) REFERENCES itl_representations(id),
                    FOREIGN KEY (translation_id) REFERENCES nl_translations(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verification_statistics (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    total_entries INTEGER,
                    total_verified INTEGER,
                    correct_percentage REAL,
                    incorrect_percentage REAL,
                    average_score REAL,
                    median_score REAL,
                    average_time REAL,
                    statistics_json TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ver_formula_id ON verification_results(formula_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ver_itl_id ON verification_results(itl_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ver_translation_id ON verification_results(translation_id)")
            
            conn.commit()
    
    def get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def count_translations(self) -> int:
        """Count the total number of translations in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nl_translations")
            return cursor.fetchone()[0]
    
    def select_verification_sample(self) -> List[Dict]:
        """Select a random 18% sample of translations for verification."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all translations
            cursor.execute("""
                SELECT 
                    nt.id as translation_id,
                    nt.formula_id,
                    nt.itl_id,
                    f.formula,
                    f.spot_formulas,
                    i.itl_text,
                    nt.domain,
                    nt.activity,
                    nt.translation
                FROM nl_translations nt
                JOIN formulas f ON nt.formula_id = f.id
                JOIN itl_representations i ON nt.itl_id = i.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM verification_results vr 
                    WHERE vr.translation_id = nt.id
                )
            """)
            all_entries = [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]
        
        # Calculate sample size (18%)
        total_entries = len(all_entries)
        sample_size = int(total_entries * VERIFICATION_PERCENTAGE / 100)
        
        sample_size = min(sample_size, total_entries)
        
        verification_sample = random.sample(all_entries, sample_size) if sample_size < total_entries else all_entries
        
        logger.info(f"Selected {len(verification_sample)} entries for verification from {total_entries} total entries")
        return verification_sample
    
    def store_verification_result(self, result: Dict) -> None:
        """Store a verification result in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO verification_results
                (formula_id, itl_id, translation_id, is_correct, score, issues, 
                 reasoning, verification_time, verification_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result["formula_id"],
                result["itl_id"],
                result["translation_id"],
                1 if result.get("is_correct", False) else 0,
                result.get("score", 0),
                json.dumps(result.get("issues", [])),
                result.get("reasoning", ""),
                result.get("verification_time", 0),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def store_verification_statistics(self, stats: Dict) -> None:
        """Store verification statistics in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO verification_statistics
                (timestamp, total_entries, total_verified, correct_percentage, 
                 incorrect_percentage, average_score, median_score, average_time, statistics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                TIMESTAMP,
                stats.get("total_entries", 0),
                stats.get("total_verified", 0),
                stats.get("correct_percentage", 0),
                stats.get("incorrect_percentage", 0),
                stats.get("average_score", 0),
                stats.get("median_score", 0),
                stats.get("average_time", 0),
                json.dumps(stats)
            ))
            conn.commit()

class LlamaVerifier:
    """Verifies LTL/ITL/NL translations using Llama 3.3."""
    
    def __init__(self, model_dir: str = MODEL_DIR, model_id: str = MODEL_ID):
        """Initialize the Llama verifier."""
        self.model_dir = model_dir
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        
        self.stats = {
            "total_verified": 0,
            "correct_translations": 0,
            "incorrect_translations": 0,
            "verification_scores": [],
            "verification_times": [],
            "errors": 0,
            "domain_stats": {},
            "score_distribution": {i: 0 for i in range(11)}  # 0-10 scores
        }
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load the Llama 3.3 model with 8-bit quantization."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_files_exist = any(Path(self.model_dir).rglob("*.safetensors"))
        if model_files_exist:
            logger.info("Local model files found. Loading from cache...")
            model_path = self.model_dir
        else:
            logger.info("Local model files not found. The model will be downloaded.")
            model_path = self.model_id
       
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8"
        )
        
        logger.info(f"Loading model to {self.model_dir}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.model_dir,
            local_files_only=model_files_exist
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=self.model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=model_files_exist
        )
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Model is using {self.model.get_memory_footprint() / 1e9:.2f} GB of memory")
    
    def verify_translation(self, ltl_formula: str, itl_text: str, nl_translation: str) -> Dict:
        """Verify if a natural language translation correctly captures the LTL formula."""
        prompt = f"""You are an expert in formal verification and temporal logic.

    Task: Analyze if this natural language translation PRECISELY captures the meaning of the temporal logic formula.

    LTL Formula: {ltl_formula}
    ITL Representation: {itl_text}
    Natural Language: {nl_translation}

    Pay SPECIAL ATTENTION to temporal operators:
    - G (Always/Globally): Must be maintained throughout all states
    - F (Eventually/Finally): Must happen at some future point
    - X (Next): Must happen in the IMMEDIATELY next state
    - U (Until): First condition must hold until second occurs
    - R (Release): Second condition must hold until and including when first happens

    The translation is INCORRECT if any of these temporal relationships are changed or relaxed.
    For example, "eventually" (F) is NOT equivalent to "next" (X).

    Provide your assessment as a JSON object with these fields:
    - is_correct (true/false)
    - score (0-10)
    - issues (list of any specific problems found)
    - reasoning (explanation)
    """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            result = self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            result = {
                "is_correct": False,
                "score": 0,
                "issues": [f"Error during verification: {str(e)}"],
                "reasoning": "Verification failed due to an error"
            }
            self.stats["errors"] += 1
        
        verification_time = time.time() - start_time
        result["verification_time"] = verification_time
        self.stats["verification_times"].append(verification_time)
        
        self.stats["total_verified"] += 1
        if result.get("is_correct", False):
            self.stats["correct_translations"] += 1
        else:
            self.stats["incorrect_translations"] += 1
        
        score = result.get("score", 0)
        self.stats["verification_scores"].append(score)
        self.stats["score_distribution"][score] += 1
        
        logger.info(f"Verification completed in {verification_time:.2f} seconds")
        logger.info(f"Result: correct={result.get('is_correct', False)}, score={score}/10")
        
        return result
    
    def _parse_response(self, response: str) -> Dict:
        """Enhanced parser for model responses with better error detection."""
        try:
            clean_response = response.replace("```json", "").replace("```", "").strip()
            
            import re
            json_match = re.search(r'(\{.*\})', clean_response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except json.JSONDecodeError:
                    pass
            
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    pass
            
            is_correct_match = re.search(r'is_correct[": ]*(true|false)', clean_response, re.IGNORECASE)
            is_correct = False
            if is_correct_match:
                is_correct = is_correct_match.group(1).lower() == "true"
            else:
                correct_indicators = [
                    r'\bcorrect\b.*\btranslation\b',
                    r'\baccurately\s+capture',
                    r'\btrue\s+to\s+the\s+formula\b'
                ]
                incorrect_indicators = [
                    r'\bincorrect\b.*\btranslation\b',
                    r'\bdoes\s+not\s+preserve\b',
                    r'\bmisrepresents\b',
                    r'\bnot\s+equivalent\b'
                ]
                
                for pattern in correct_indicators:
                    if re.search(pattern, clean_response, re.IGNORECASE):
                        is_correct = True
                        break
                        
                for pattern in incorrect_indicators:
                    if re.search(pattern, clean_response, re.IGNORECASE):
                        is_correct = False
                        break
            
            score_pattern = r'score[": ]*(\d+)'
            score_match = re.search(score_pattern, clean_response, re.IGNORECASE)
            score = 0
            if score_match:
                score = int(score_match.group(1))
            else:
                score = 8 if is_correct else 3
            
            issues = []
            issues_pattern = r'issues[": ]*\[(.*?)\]'
            issues_match = re.search(issues_pattern, clean_response, re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1)
                issues = [issue.strip(' "\'') for issue in re.findall(r'"([^"]*)"', issues_text)]
                if not issues:
                    issues = [issue.strip() for issue in issues_text.split(',') if issue.strip()]
            
            if not issues and not is_correct:
                issues = ["Temporal relationship not preserved"]
            
            return {
                "is_correct": is_correct,
                "score": score,
                "issues": issues,
                "reasoning": clean_response
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {
                "is_correct": False,
                "score": 0,
                "issues": [f"Error parsing response: {str(e)}"],
                "reasoning": "Could not parse response",
                "raw_response": response
            }
    
    def calculate_statistics(self) -> Dict:
        """Calculate verification statistics."""
        stats = self.stats.copy()
        
        total_verified = stats["total_verified"]
        if total_verified > 0:
            stats["correct_percentage"] = (stats["correct_translations"] / total_verified) * 100
            stats["incorrect_percentage"] = (stats["incorrect_translations"] / total_verified) * 100
            
            scores = stats["verification_scores"]
            stats["average_score"] = sum(scores) / len(scores) if scores else 0
            stats["median_score"] = np.median(scores) if scores else 0
            
            times = stats["verification_times"]
            stats["average_time"] = sum(times) / len(times) if times else 0
            stats["min_time"] = min(times) if times else 0
            stats["max_time"] = max(times) if times else 0
        
        return stats

class VerificationSystem:
    """Main system for LTL/ITL/NL translation verification."""
    
    def __init__(self, db_path: str, batch_size: int = 20):
        """Initialize the verification system."""
        self.db_manager = DatabaseManager(db_path)
        self.verifier = LlamaVerifier()
        self.batch_size = batch_size
        self.stats = {}
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def run_verification(self) -> None:
        """Run the complete verification process."""
        logger.info(f"Starting verification process at {datetime.now()}")
        
        total_entries = self.db_manager.count_translations()
        self.stats["total_entries"] = total_entries
        logger.info(f"Found {total_entries} total translations in the database")
        
        verification_sample = self.db_manager.select_verification_sample()
        self.stats["selected_for_verification"] = len(verification_sample)
        
        if not verification_sample:
            logger.info("No translations to verify. Exiting.")
            return
        
        for i in range(0, len(verification_sample), self.batch_size):
            batch = verification_sample[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(verification_sample) + self.batch_size - 1)//self.batch_size}")
            
            for entry in tqdm(batch, desc="Verifying translations"):
                formula_id = entry["formula_id"]
                itl_id = entry["itl_id"]
                translation_id = entry["translation_id"]
                
                ltl_formula = entry.get("spot_formulas") if entry.get("spot_formulas") else entry["formula"]
                itl_text = entry["itl_text"]
                nl_translation = entry["translation"]
                
                result = self.verifier.verify_translation(ltl_formula, itl_text, nl_translation)
                
                result["formula_id"] = formula_id
                result["itl_id"] = itl_id
                result["translation_id"] = translation_id
                
                self.db_manager.store_verification_result(result)
            
            interim_stats = self.verifier.calculate_statistics()
            logger.info(f"Interim statistics: verified={interim_stats['total_verified']}, "
                       f"correct={interim_stats.get('correct_percentage', 0):.2f}%, "
                       f"avg_score={interim_stats.get('average_score', 0):.2f}/10")
        
        self.stats.update(self.verifier.calculate_statistics())
        
        self.db_manager.store_verification_statistics(self.stats)
        
        self._save_statistics_to_csv()
        
        logger.info(f"Verification process completed at {datetime.now()}")
        logger.info(f"Verified {self.stats['total_verified']} translations")
        logger.info(f"Correct: {self.stats['correct_translations']} ({self.stats.get('correct_percentage', 0):.2f}%)")
        logger.info(f"Incorrect: {self.stats['incorrect_translations']} ({self.stats.get('incorrect_percentage', 0):.2f}%)")
        logger.info(f"Average Score: {self.stats.get('average_score', 0):.2f}/10")
    
    def _save_statistics_to_csv(self) -> None:
        """Save verification statistics to CSV."""
        stats_csv_path = f"{RESULTS_DIR}/verification_stats_{TIMESTAMP}.csv"
        
        with open(stats_csv_path, 'w', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            
            writer.writerow(['Verification Statistics', TIMESTAMP])
            writer.writerow(['Total Entries', self.stats["total_entries"]])
            writer.writerow(['Selected for Verification', self.stats["selected_for_verification"]])
            writer.writerow(['Verifications Completed', self.stats["total_verified"]])
            writer.writerow(['Correct Translations', self.stats["correct_translations"]])
            writer.writerow(['Incorrect Translations', self.stats["incorrect_translations"]])
            writer.writerow(['Correct Percentage', f"{self.stats.get('correct_percentage', 0):.2f}%"])
            writer.writerow(['Incorrect Percentage', f"{self.stats.get('incorrect_percentage', 0):.2f}%"])
            writer.writerow(['Average Score', f"{self.stats.get('average_score', 0):.2f}"])
            writer.writerow(['Median Score', f"{self.stats.get('median_score', 0):.2f}"])
            writer.writerow(['Average Verification Time', f"{self.stats.get('average_time', 0):.2f} seconds"])
            writer.writerow(['Errors Encountered', self.stats["errors"]])
            writer.writerow([])
            
            writer.writerow(['Score Distribution'])
            writer.writerow(['Score', 'Count', 'Percentage'])
            for score, count in self.stats["score_distribution"].items():
                percentage = (count / self.stats["total_verified"]) * 100 if self.stats["total_verified"] > 0 else 0
                writer.writerow([score, count, f"{percentage:.2f}%"])
            
        logger.info(f"Statistics saved to {stats_csv_path}")

def run_test_examples():
    """Run a series of test examples to verify the model's performance."""
    logger.info("Running test examples...")
    
    verifier = LlamaVerifier()
    
    test_examples = [
        # Example 1: Correct translation
        {
            "ltl_formula": "G(p → Fq)",
            "itl_text": "Always, if p then eventually q",
            "nl_translation": "Whenever a transaction is initiated, it will eventually be completed.",
            "expected_correct": True
        },
        # Example 2: Correct translation
        {
            "ltl_formula": "F(p & Gq)",
            "itl_text": "Eventually, p and always q thereafter",
            "nl_translation": "At some point, the system will enter maintenance mode and remain stable indefinitely.",
            "expected_correct": True
        },
        # Example 3: Incorrect translation (wrong temporal relationship)
        {
            "ltl_formula": "G(p → Xq)",
            "itl_text": "Always, if p then in the next state q",
            "nl_translation": "If an error occurs, the system will eventually restart.",
            "expected_correct": False
        },
        # Example 4: Incorrect translation (missing temporal operator)
        {
            "ltl_formula": "p U q",
            "itl_text": "p until q",
            "nl_translation": "The system checks for errors and sends notifications.",
            "expected_correct": False
        },
        # Example 5: Correct but complex
        {
            "ltl_formula": "G(p → (q U r))",
            "itl_text": "Always, if p then q until r",
            "nl_translation": "Whenever a user logs in, they remain in the authenticated state until they explicitly log out.",
            "expected_correct": True
        }
    ]
    
    results = []
    
    for i, example in enumerate(test_examples):
        logger.info(f"\nTesting example {i+1}/{len(test_examples)}")
        logger.info(f"LTL: {example['ltl_formula']}")
        logger.info(f"ITL: {example['itl_text']}")
        logger.info(f"NL: {example['nl_translation']}")
        logger.info(f"Expected: {'Correct' if example['expected_correct'] else 'Incorrect'}")
        
        result = verifier.verify_translation(
            example["ltl_formula"],
            example["itl_text"],
            example["nl_translation"]
        )
        
        result["example_id"] = i+1
        result["ltl_formula"] = example["ltl_formula"]
        result["itl_text"] = example["itl_text"] 
        result["nl_translation"] = example["nl_translation"]
        result["expected_correct"] = example["expected_correct"]
        
        results.append(result)
        
        logger.info(f"Model verdict: {'Correct' if result.get('is_correct', False) else 'Incorrect'}")
        logger.info(f"Score: {result.get('score', 0)}/10")
        logger.info(f"Issues: {result.get('issues', [])}")
        logger.info(f"Reasoning: {result.get('reasoning', '')[:100]}...")
        
        matches_expected = (result.get('is_correct', False) == example['expected_correct'])
        logger.info(f"Matches expected? {'Yes' if matches_expected else 'No'}")
    
    test_results_path = f"{RESULTS_DIR}/test_results_{TIMESTAMP}.json" 
    with open(test_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTest results saved to {test_results_path}")
    
    correct_assessments = sum(1 for r in results if r.get('is_correct', False) == r['expected_correct'])
    accuracy = (correct_assessments / len(results)) * 100
    
    logger.info(f"\nTest Statistics:")
    logger.info(f"Total examples: {len(results)}")
    logger.info(f"Correct assessments: {correct_assessments}/{len(results)} ({accuracy:.2f}%)")
    logger.info(f"Average score: {sum(r.get('score', 0) for r in results) / len(results):.2f}/10")
    logger.info(f"Average time: {sum(r.get('verification_time', 0) for r in results) / len(results):.2f} seconds")

def main():
    if RUN_TEST_EXAMPLES_MODE:
        logger.info("Running in test examples mode.")
        run_test_examples()
    elif DB_PATH_VERIFICATION:
        logger.info(f"Running database verification with DB: {DB_PATH_VERIFICATION}")
        if not os.path.exists(DB_PATH_VERIFICATION):
            logger.error(f"Database file not found: {DB_PATH_VERIFICATION}")
            sys.exit(1)
        
        system = VerificationSystem(DB_PATH_VERIFICATION, batch_size=BATCH_SIZE_VERIFICATION)
        system.run_verification()
    else:
        message = (
            "Script not configured to run. Please set either:\n"
            "1. RUN_TEST_EXAMPLES_MODE = True\n"
            "2. DB_PATH_VERIFICATION = \"/path/to/your/database.db\"\n"
            "at the top of the script."
        )
        print(message)
        logger.info(message)
        sys.exit(1)

if __name__ == "__main__":
    main()