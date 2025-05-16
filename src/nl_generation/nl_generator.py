import argparse
import json
import logging
import multiprocessing
import os
import random
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nl_generation.log')
    ]
)
logger = logging.getLogger('nl_generation')

DOMAINS = [
    "Home Automation", "Robotics", "Automotive/Autonomous Vehicles", 
    "Aerospace", "Medical Devices", "Industrial Automation/Manufacturing",
    "Networking/Distributed Systems", "Financial/Transaction Systems", 
    "Web Services/APIs", "Smart Grid/Energy Management", 
    "Build Pipelines and CI/CD", "Version Control and Code Reviews",
    "Security and Authentication"
]

DB_PATH_MAIN = "/path/to/your/database.db"
ENV_FILE_PATH = ".env"
MIN_FORMULA_DEPTH_MAIN = 1
PROCESSING_BATCH_SIZE_MAIN = # number of formulas you want processed
API_SLEEP_TIME_MAIN = 0.1
NUM_WORKER_PROCESSES_MAIN = # number of worker processes

TERMINATE = multiprocessing.Value('i', 0)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str, worker_id: int = 0):
        """Initialize database manager."""
        self.db_path = db_path
        self.worker_id = worker_id
        self.conn = None
        self.lock = multiprocessing.Lock()
        self._setup_database()
    
    def _setup_database(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nl_translations (
                    id INTEGER PRIMARY KEY,
                    formula_id INTEGER NOT NULL,
                    itl_id INTEGER NOT NULL,
                    domain TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    generation_time REAL,
                    timestamp TEXT,
                    worker_id INTEGER,
                    FOREIGN KEY (formula_id) REFERENCES formulas(id),
                    FOREIGN KEY (itl_id) REFERENCES itl_representations(id)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nl_formula_id ON nl_translations(formula_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nl_itl_id ON nl_translations(itl_id)")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_stats (
                    id INTEGER PRIMARY KEY,
                    domain TEXT UNIQUE,
                    count INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("PRAGMA table_info(nl_translations)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if 'worker_id' not in columns:
                try:
                    cursor.execute("ALTER TABLE nl_translations ADD COLUMN worker_id INTEGER")
                    logger.info("Added worker_id column to nl_translations table")
                except sqlite3.OperationalError:
                    pass
            
            cursor.execute("SELECT COUNT(*) FROM domain_stats")
            if cursor.fetchone()[0] == 0:
                for domain in DOMAINS:
                    cursor.execute("INSERT INTO domain_stats (domain, count) VALUES (?, 0)", (domain,))
            
            conn.commit()
    
    def get_connection(self):
        """Get a database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, timeout=60.0)
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA busy_timeout = 30000")
        return self.conn
    
    def close_connection(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
    
    def get_complex_formulas(self, min_depth: int = 5, limit: int = 20, offset: int = 0) -> List[Dict]:
        """Get formulas with depth ≥ min_depth with pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT f.id, f.formula, f.canonical_hash, f.depth, f.spot_formulas, 
                       i.id as itl_id, i.itl_text
                FROM formulas f
                JOIN itl_representations i ON f.id = i.formula_id
                WHERE f.depth >= ? 
                  AND i.canonical_form = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM nl_translations nt 
                      WHERE nt.formula_id = f.id
                  )
                ORDER BY f.id
                LIMIT ? OFFSET ?
            """
            
            try:
                cursor.execute(query, (min_depth, limit, offset))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                logger.error(f"Worker {self.worker_id}: Database error in get_complex_formulas: {e}")
                return []
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Get current distribution of domains with proper locking."""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT domain, count FROM domain_stats")
                    return {row[0]: row[1] for row in cursor.fetchall()}
            except sqlite3.Error as e:
                logger.error(f"Worker {self.worker_id}: Database error in get_domain_distribution: {e}")
                return {domain: 0 for domain in DOMAINS}
    
    def update_domain_count(self, domain: str) -> None:
        """Update count for a domain with proper locking."""
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE domain_stats SET count = count + 1 WHERE domain = ?", 
                        (domain,)
                    )
                    conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Worker {self.worker_id}: Database error in update_domain_count: {e}")
    
    def store_translation(self, data: Dict) -> int:
        with self.lock:
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO nl_translations
                        (formula_id, itl_id, domain, activity, translation,
                         generation_time, timestamp, worker_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data["formula_id"],
                        data["itl_id"],
                        data["domain"],
                        data["activity"],
                        data["translation"],
                        data.get("generation_time", 0),
                        datetime.now().isoformat(),
                        self.worker_id
                    ))
                    conn.commit()
                    logger.debug(f"Worker {self.worker_id}: Stored translation for formula {data['formula_id']}")
                    return cursor.lastrowid
            except sqlite3.Error as e:
                logger.error(f"Worker {self.worker_id}: Database error storing translation for formula {data.get('formula_id', 'N/A')}: {e}")
                return -1

class DeepSeekGenerator:
    """Generates natural language using DeepSeek API."""
    
    def __init__(self, api_key: str, db_manager: DatabaseManager, model: str = "deepseek-reasoner"):
        """Initialize DeepSeek generator."""
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.db_manager = db_manager
    
    def select_balanced_domain(self, domain_distribution: Dict[str, int], temperature: float = 0.3) -> str:
        """
        Select a domain with a balanced probability distribution.
        """
        domains = list(domain_distribution.keys())
        counts = np.array([domain_distribution[d] for d in domains], dtype=np.float32)
        
        counts = counts + 1.0
        inverse_weights = 1.0 / counts
        
        logits = np.log(inverse_weights) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        selected_domain = np.random.choice(domains, p=probabilities)
        
        return selected_domain
    
    def generate_translation(self, formula: Dict) -> Optional[Dict]:
        """Generate a natural language translation for a formula."""
    
        domain_distribution = self.db_manager.get_domain_distribution()
        domain = self.select_balanced_domain(domain_distribution)

        system_prompt = """You are an expert in formal verification and natural language processing, specializing in translating formal specifications into clear, intuitive language for domain experts.

            Your task is to translate a Linear Temporal Logic (LTL) formula into natural language that domain specialists would understand. Your translation must:

            1. Be precise: Preserve ALL temporal relationships from the original formula
            2. Be concise: Direct and clear without unnecessarily complex language
            3. Be contextual: Include meaningful variable interpretations for the chosen domain
            4. Be natural: Use language that domain experts would actually use

            Your response MUST follow this exact format:
            <domain>Selected domain</domain>
            <activity>Detailed activity context explaining what the variables represent</activity>
            <translation>Clear, concise natural language translation of the formula</translation>

            Guidelines:
            - The <translation> should be 1-2 sentences, focused and precise
            - The <activity> should establish what atomic propositions represent
            - Think deeply before responding to ensure semantic correctness
            - Avoid overly technical language, analogies, or filler phrases
            - Ensure your answer would be immediately useful for formal verification research"""

        ltl_formula = formula["spot_formulas"] if formula.get("spot_formulas") else formula["formula"]
        itl_text = formula["itl_text"]
        
        user_prompt = f"""Please translate this temporal logic formula into natural language.

            LTL Formula: {ltl_formula}
            ITL Representation: {itl_text}

            Choose this specific domain: {domain}

            Remember:
            1. Preserve EXACT logical meaning
            2. Be concise but complete
            3. Focus on the temporal aspects
            4. Use domain-specific terminology

            Respond ONLY with the three tags (<domain>, <activity>, <translation>) as instructed."""

        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                temperature=0.7,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content
            generation_time = time.time() - start_time
            
            domain = self._extract_tag(response_text, "domain")
            activity = self._extract_tag(response_text, "activity")
            translation = self._extract_tag(response_text, "translation")
            
            if not domain or not activity or not translation:
                logger.warning(f"Worker {self.db_manager.worker_id}: Invalid response format: {response_text}")
                return None
                
            domain = self._normalize_domain(domain)
            
            self.db_manager.update_domain_count(domain)
            
            return {
                "formula_id": formula["id"],
                "itl_id": formula["itl_id"],
                "domain": domain,
                "activity": activity,
                "translation": translation,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Worker {self.db_manager.worker_id}: API call failed: {e}")
            return None
    
    def _extract_tag(self, text: str, tag_name: str) -> Optional[str]:
        """Extract content between XML tags."""
        import re
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain name to match our list."""
        domain = domain.strip()
        for d in DOMAINS:
            if domain.lower() in d.lower() or d.lower() in domain.lower():
                return d
        
        logger.warning(f"Worker {self.db_manager.worker_id}: Unknown domain: {domain}, defaulting to {DOMAINS[0]}")
        return DOMAINS[0]

def should_terminate():
    """Check if we should terminate based on time or global flag."""
    return TERMINATE.value == 1

def worker_process(worker_id, db_path, env_path, max_cost, min_depth, batch_size, sleep_time=0.1):
    try:
        np.random.seed(int(time.time()) + worker_id)

        logger.info(f"Worker {worker_id}: Starting with offset {worker_id * batch_size}")

        print(f"Attempting to load .env from: '{env_path}'")
        load_dotenv(env_path)
        api_key = os.getenv("DEEPSEEK_API")

        if not api_key:
            logger.error(f"Worker {worker_id}: API key not found in {env_path}")
            return

        db_manager = DatabaseManager(db_path, worker_id)
        generator = DeepSeekGenerator(api_key, db_manager)

        total_processed = 0
        successful = 0
        failed = 0
        skipped_pre_check = 0
        offset = worker_id * batch_size

        while not should_terminate():
            try:
                formulas = db_manager.get_complex_formulas(
                    min_depth=min_depth,
                    limit=batch_size,
                    offset=offset
                )

                if not formulas:
                    logger.info(f"Worker {worker_id}: No more formulas to process at offset {offset}")
                    offset += 120 * batch_size
                    time.sleep(sleep_time * 2)
                    continue

                logger.info(f"Worker {worker_id}: Processing {len(formulas)} formulas at offset {offset}")

                for formula in formulas:
                    if should_terminate():
                        logger.info(f"Worker {worker_id}: Termination signal received")
                        break

                    formula_id_to_check = formula["id"]
                    translation_exists = False
                    try:
                        # Check if ANY translation exists before generating
                        with db_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT 1 FROM nl_translations WHERE formula_id = ? LIMIT 1",
                                (formula_id_to_check,)
                            )
                            if cursor.fetchone():
                                translation_exists = True
                    except sqlite3.Error as e:
                         logger.error(f"Worker {worker_id}: Database error checking formula {formula_id_to_check}: {e}")
                         time.sleep(sleep_time)
                         continue

                    if translation_exists:
                         logger.debug(f"Worker {worker_id}: Skipping formula {formula_id_to_check}, translation already exists.")
                         skipped_pre_check += 1
                         total_processed += 1
                         continue

                    result = generator.generate_translation(formula)

                    if result:
                        store_result = db_manager.store_translation(result)
                        if store_result != -1:
                           successful += 1
                        else:
                           failed += 1
                    else:
                        failed += 1
                    total_processed += 1
                    time.sleep(sleep_time)
                offset += batch_size

                if total_processed % 10 == 0:
                    logger.info(f"Worker {worker_id}: Processed {total_processed} formulas,"
                                f"Success: {successful}, Failed: {failed}, Skipped: {skipped_pre_check},")

            except Exception as e:
                logger.error(f"Worker {worker_id}: Error in processing loop: {e}")
                time.sleep(sleep_time * 5)

        db_manager.close_connection()

        logger.info(f"Worker {worker_id}: Finished! Processed {total_processed} formulas,"
                    f"Success: {successful}, Failed: {failed}, Skipped: {skipped_pre_check},")

    except Exception as e:
        logger.error(f"Worker {worker_id}: Fatal error: {e}")

def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info("Received termination signal, setting termination flag")
    with TERMINATE.get_lock():
        TERMINATE.value = 1

def main():
    """Main function to run the parallel script using hardcoded constants."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not os.path.exists(DB_PATH_MAIN):
        logger.error(f"Database not found: {DB_PATH_MAIN}")
        sys.exit(1)

    try:
        db_manager_init = DatabaseManager(DB_PATH_MAIN, 0)
        db_manager_init.close_connection()
    except Exception as e:
        logger.warning(f"Could not pre-initialize DatabaseManager (optional step): {e}")

    processes = []
    logger.info(f"Starting {NUM_WORKER_PROCESSES_MAIN} worker processes")

    for i in range(NUM_WORKER_PROCESSES_MAIN):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, DB_PATH_MAIN, ENV_FILE_PATH,
                  MIN_FORMULA_DEPTH_MAIN, PROCESSING_BATCH_SIZE_MAIN, API_SLEEP_TIME_MAIN)
        )
        p.daemon = True
        p.start()
        processes.append(p)
        time.sleep(0.05)

    logger.info(f"All {NUM_WORKER_PROCESSES_MAIN} worker processes started")

    try:
        while not should_terminate():
            for i, p in enumerate(processes):
                if not p.is_alive():
                    logger.warning(f"Worker process {i} (PID {p.pid}) died, restarting")
                    new_p = multiprocessing.Process(
                        target=worker_process,
                        args=(i, DB_PATH_MAIN, ENV_FILE_PATH,
                              MIN_FORMULA_DEPTH_MAIN, PROCESSING_BATCH_SIZE_MAIN, API_SLEEP_TIME_MAIN)
                    )
                    new_p.daemon = True
                    new_p.start()
                    processes[i] = new_p
            
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Main process interrupted by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}", exc_info=True)
    finally:
        logger.info("Initiating shutdown sequence...")
        with TERMINATE.get_lock():
            TERMINATE.value = 1
        
        logger.info("Waiting for all worker processes to terminate gracefully...")
        
        termination_grace_period = 120
        start_wait_time = time.time()
        
        for p in processes:
            join_timeout = max(0, termination_grace_period - (time.time() - start_wait_time))
            p.join(timeout=join_timeout)

        for i, p in enumerate(processes):
            if p.is_alive():
                logger.warning(f"Worker process {i} (PID {p.pid}) did not terminate gracefully. Forcefully terminating.")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.error(f"Worker process {i} (PID {p.pid}) could not be forcefully terminated.")


        logger.info("All worker processes have been processed for termination.")
        
        try:
            final_db_manager = DatabaseManager(DB_PATH_MAIN, -1)
            domain_distribution = final_db_manager.get_domain_distribution()
            logger.info("Final domain distribution in the database:")
            if domain_distribution:
                for domain, count in domain_distribution.items():
                    logger.info(f"  Domain '{domain}': {count} entries")
            else:
                logger.info("  Could not retrieve domain distribution or it's empty.")
            final_db_manager.close_connection()
        except Exception as e:
            logger.error(f"Failed to retrieve or log final statistics: {e}", exc_info=True)
            
        logger.info("Program shutdown complete.")

if __name__ == "__main__":
    main()
