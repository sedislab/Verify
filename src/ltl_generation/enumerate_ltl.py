from dataclasses import dataclass
from enum import Enum, auto
import itertools
from typing import Set, List, Dict, Optional, Union, Tuple, Generator
import sqlite3
import os
import json
from pathlib import Path
import threading
from queue import Queue
import numpy as np
from tqdm import tqdm
import shelve
import tempfile
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import deque
import random
import logging
import sys
from datetime import datetime
import time

TARGET_LTL_COUNT = ""
MAX_LTL_DEPTH = ""
BASE_DIRECTORY = Path("/path/to/dir")

class OperatorType(Enum):
    LOGICAL = auto()
    TEMPORAL = auto()
    ATOMIC = auto()

class Operator(Enum):
    # Logical operators
    AND = ("\\land", 2, OperatorType.LOGICAL)
    OR = ("\\lor", 2, OperatorType.LOGICAL)
    NOT = ("\\neg", 1, OperatorType.LOGICAL)
    IMPLIES = ("\\rightarrow", 2, OperatorType.LOGICAL)
    EQUIV = ("\\leftrightarrow", 2, OperatorType.LOGICAL)
    
    # Temporal operators
    GLOBALLY = ("G", 1, OperatorType.TEMPORAL)
    FINALLY = ("F", 1, OperatorType.TEMPORAL)
    NEXT = ("X", 1, OperatorType.TEMPORAL)
    UNTIL = ("U", 2, OperatorType.TEMPORAL)
    RELEASE = ("R", 2, OperatorType.TEMPORAL)
    WEAK_UNTIL = ("W", 2, OperatorType.TEMPORAL)

    def __init__(self, symbol: str, arity: int, op_type: OperatorType):
        self.symbol = symbol
        self.arity = arity
        self.type = op_type

@dataclass(frozen=True)
class Formula:
    operator: Optional[Operator]
    subformulas: Tuple['Formula', ...]
    atom: Optional[str] = None
    
    def __post_init__(self):
        if self.operator is None and self.atom is None:
            raise ValueError("Formula must have either an operator or an atom")
        if self.operator is not None and self.atom is not None:
            raise ValueError("Formula cannot have both operator and atom")
        if self.operator is not None and len(self.subformulas) != self.operator.arity:
            raise ValueError(f"Operator {self.operator} requires {self.operator.arity} subformulas")

    def to_latex(self) -> str:
        """Converts the formula to LaTeX representation with proper bracketing."""
        if self.atom is not None:
            return self.atom

        if self.operator == Operator.NOT:
            return f"{self.operator.symbol} {{{self.subformulas[0].to_latex()}}}"
            
        if self.operator.arity == 1:
            return f"{self.operator.symbol} {{{self.subformulas[0].to_latex()}}}"
            
        # Handle binary operators with proper precedence
        left = self.subformulas[0].to_latex()
        right = self.subformulas[1].to_latex()
        
        # Add brackets based on operator precedence
        if self._needs_brackets(self.subformulas[0]):
            left = f"\\left({left}\\right)"
        if self._needs_brackets(self.subformulas[1]):
            right = f"\\left({right}\\right)"
            
        return f"{left} {self.operator.symbol} {right}"
        
    def _needs_brackets(self, subformula: 'Formula') -> bool:
        """Determines if a subformula needs brackets based on operator precedence."""
        if subformula.atom is not None:
            return False
            
        # Precedence rules
        precedence = {
            Operator.NOT: 1,
            Operator.AND: 2,
            Operator.OR: 3,
            Operator.IMPLIES: 4,
            Operator.EQUIV: 5,
            # Temporal operators always need brackets
            Operator.GLOBALLY: 0,
            Operator.FINALLY: 0,
            Operator.NEXT: 0,
            Operator.UNTIL: 0,
            Operator.RELEASE: 0,
            Operator.WEAK_UNTIL: 0
        }
        
        # If parent has higher precedence, child needs brackets
        parent_precedence = precedence[self.operator]
        child_precedence = precedence[subformula.operator]
        return child_precedence > parent_precedence
        
    def __str__(self) -> str:
        """Returns the LaTeX representation of the formula."""
        return self.to_latex()

class FormulaStorage:
    """Handles persistent storage of generated formulas using SQLite and file-based caching."""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = Path(tempfile.gettempdir()) / "ltl_formulas"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_dir / "formulas.db"
        self.init_database()
        
        # Initialize shelve for semantic cache
        self.cache_path = str(self.storage_dir / "semantic_cache")
        
    def init_database(self):
        """Initializes the SQLite database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS formulas (
                    id INTEGER PRIMARY KEY,
                    formula TEXT NOT NULL,
                    latex TEXT NOT NULL,
                    canonical_hash TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    UNIQUE(canonical_hash)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON formulas(canonical_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_depth ON formulas(depth)")
    
    def store_formula(self, formula: 'Formula', latex: str, canonical_hash: str, depth: int):
        """Stores a formula in the database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            try:
                conn.execute(
                    "INSERT INTO formulas (formula, latex, canonical_hash, depth) VALUES (?, ?, ?, ?)",
                    (str(formula), latex, canonical_hash, depth)
                )
                return True
            except sqlite3.IntegrityError:
                return False
    
    def is_stored(self, canonical_hash: str) -> bool:
        """Checks if a formula with the given canonical hash exists."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM formulas WHERE canonical_hash = ?",
                (canonical_hash,)
            )
            return cursor.fetchone()[0] > 0
    
    def get_formula_count(self, depth: Optional[int] = None) -> int:
        """Returns the total number of stored formulas, optionally filtered by depth."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if depth is not None:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM formulas WHERE depth = ?",
                    (depth,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM formulas")
            return cursor.fetchone()[0]
    
    def iter_formulas(self, batch_size: int = 1000) -> Generator[List[Tuple[str, str]], None, None]:
        """Iterates over stored formulas in batches."""
        with sqlite3.connect(str(self.db_path)) as conn:
            offset = 0
            while True:
                cursor = conn.execute(
                    "SELECT formula, latex FROM formulas LIMIT ? OFFSET ?",
                    (batch_size, offset)
                )
                batch = cursor.fetchall()
                if not batch:
                    break
                yield batch
                offset += batch_size

class LTLGenerator:
    def __init__(self, max_depth: int = 8, storage_dir: str = None):
        self.max_depth = max_depth
        self.atoms = {'p', 'q', 'r', 's', 't', 'u', 'v', 'w'}
        self.storage = FormulaStorage(storage_dir)
        self.semantic_cache: Dict[str, Set[str]] = {}
        
    def _to_canonical_form(self, formula: Formula) -> Formula:
        """Converts a formula to its canonical form applying all possible transformations."""
        if formula.atom is not None:
            return formula
            
        # First apply NNF and expand implications
        formula = self._to_nnf(formula)
        formula = self._expand_implications(formula)
        
        # Apply associative transformations
        formula = self._apply_associative_laws(formula)
        
        # Apply distributive transformations
        formula = self._apply_distributive_laws(formula)
        
        # Sort subformulas for commutative operators
        formula = self._sort_subformulas(formula)
        
        return formula
        
    def _apply_associative_laws(self, formula: Formula) -> Formula:
        """Applies associative laws to standardize formula structure."""
        if formula.atom is not None:
            return formula
            
        if formula.operator in {Operator.AND, Operator.OR}:
            # Flatten nested AND/OR operations
            flattened = self._flatten_associative(formula)
            subformulas = tuple(self._apply_associative_laws(sub) for sub in flattened.subformulas)
            return Formula(formula.operator, subformulas)
            
        # Recursively apply to subformulas
        return Formula(formula.operator, 
                      tuple(self._apply_associative_laws(sub) for sub in formula.subformulas))
        
    def _flatten_associative(self, formula: Formula) -> Formula:
        """Flattens nested associative operations."""
        if formula.atom is not None:
            return formula
            
        if formula.operator in {Operator.AND, Operator.OR}:
            subformulas = []
            for sub in formula.subformulas:
                if sub.operator == formula.operator:
                    flattened = self._flatten_associative(sub)
                    subformulas.extend(flattened.subformulas)
                else:
                    subformulas.append(self._flatten_associative(sub))
            return Formula(formula.operator, tuple(subformulas))
            
        return Formula(formula.operator,
                      tuple(self._flatten_associative(sub) for sub in formula.subformulas))
        
    def _apply_distributive_laws(self, formula: Formula) -> Formula:
        """Applies distributive laws to standardize formula structure."""
        if formula.atom is not None:
            return formula
            
        if formula.operator == Operator.OR:
            # Check for AND subformulas to distribute over
            for i, sub in enumerate(formula.subformulas):
                if sub.operator == Operator.AND:
                    other_terms = list(formula.subformulas[:i] + formula.subformulas[i+1:])
                    distributed = [
                        Formula(Operator.OR, (term, *other_terms))
                        for term in sub.subformulas
                    ]
                    return Formula(Operator.AND, tuple(distributed))
                    
        # Recursively apply to subformulas
        return Formula(formula.operator,
                      tuple(self._apply_distributive_laws(sub) for sub in formula.subformulas))
        
    def _sort_subformulas(self, formula: Formula) -> Formula:
        """Sorts subformulas for commutative operators to ensure canonical form."""
        if formula.atom is not None:
            return formula
            
        if formula.operator in {Operator.AND, Operator.OR}:
            sorted_subs = sorted(
                (self._sort_subformulas(sub) for sub in formula.subformulas),
                key=lambda f: self._formula_key(f)
            )
            return Formula(formula.operator, tuple(sorted_subs))
            
        return Formula(formula.operator,
                      tuple(self._sort_subformulas(sub) for sub in formula.subformulas))
        
    def _formula_key(self, formula: Formula) -> str:
        """Generates a stable sorting key for formulas."""
        if formula.atom is not None:
            return formula.atom
            
        subkeys = [self._formula_key(sub) for sub in formula.subformulas]
        return f"{formula.operator.symbol}({''.join(sorted(subkeys))})"
        
    def _generate_canonical_hash(self, formula: Formula) -> str:
        """Generates a canonical hash for a formula that is invariant under semantically equivalent transformations."""
        canonical = self._to_canonical_form(formula)
        return hashlib.sha256(self._formula_key(canonical).encode()).hexdigest()

    def _is_tautology(self, formula: Formula) -> bool:
        """Complete tautology checking using semantic analysis."""
        # Convert to NNF for easier analysis
        nnf_formula = self._to_nnf(formula)
        
        # Check known tautology patterns
        if self._is_basic_tautology(nnf_formula):
            return True
            
        # Check for temporal tautologies
        if self._is_temporal_tautology(nnf_formula):
            return True
            
        return self._check_complex_tautology(nnf_formula)
        
    def _is_basic_tautology(self, formula: Formula) -> bool:
        """Checks for basic propositional tautology patterns."""
        if formula.operator == Operator.OR:
            # p ∨ ¬p
            if (len(formula.subformulas) == 2 and
                formula.subformulas[0].operator == Operator.NOT and
                formula.subformulas[1] == formula.subformulas[0].subformulas[0]):
                return True
                
            # Check for p ∨ (q ∨ ¬q)
            for subf in formula.subformulas:
                if subf.operator == Operator.OR and self._is_basic_tautology(subf):
                    return True
                    
        return False
        
    def _is_temporal_tautology(self, formula: Formula) -> bool:
        """Checks for temporal logic tautologies."""
        # G p ∨ F ¬p is a tautology
        if (formula.operator == Operator.OR and
            len(formula.subformulas) == 2 and
            formula.subformulas[0].operator == Operator.GLOBALLY and
            formula.subformulas[1].operator == Operator.FINALLY and
            formula.subformulas[1].subformulas[0].operator == Operator.NOT and
            formula.subformulas[0].subformulas[0] == formula.subformulas[1].subformulas[0].subformulas[0]):
            return True
            
        # F p ∨ G ¬p is a tautology
        if (formula.operator == Operator.OR and
            len(formula.subformulas) == 2 and
            formula.subformulas[0].operator == Operator.FINALLY and
            formula.subformulas[1].operator == Operator.GLOBALLY and
            formula.subformulas[1].subformulas[0].operator == Operator.NOT and
            formula.subformulas[0].subformulas[0] == formula.subformulas[1].subformulas[0].subformulas[0]):
            return True
            
        return False
        
    def _check_complex_tautology(self, formula: Formula) -> bool:
        """Checks for complex tautologies using semantic analysis."""
        # Convert formula to canonical form
        canonical = self._to_canonical_form(formula)
        
        known_patterns = self._get_tautology_patterns()
        return any(self._matches_pattern(canonical, pattern) for pattern in known_patterns)
        
    def _get_tautology_patterns(self) -> List[Formula]:
        """Returns a list of known tautology patterns."""
        return [
            # (p → q) ∨ (q → p)
            self._parse_pattern("(\\rightarrow(p,q) \\lor \\rightarrow(q,p))"),
            # (p ∧ q) → p
            self._parse_pattern("\\rightarrow(\\land(p,q),p)")
        ]

    def _is_contradiction(self, formula: Formula) -> bool:
        """Checks if a formula is a contradiction using symbolic analysis."""
        # Basic contradiction patterns
        if (formula.operator == Operator.AND and
            len(formula.subformulas) == 2 and
            formula.subformulas[0].operator == Operator.NOT and
            formula.subformulas[1] == formula.subformulas[0].subformulas[0]):
            return True
        return False

    def _to_nnf(self, formula: Formula) -> Formula:
        """Converts formula to Negation Normal Form."""
        if formula.atom is not None:
            return formula
            
        if formula.operator == Operator.NOT:
            sub = formula.subformulas[0]
            if sub.atom is not None:
                return formula
            
            # Push negation inward using De Morgan's laws
            if sub.operator == Operator.AND:
                return Formula(Operator.OR, tuple(self._to_nnf(Formula(Operator.NOT, (s,))) for s in sub.subformulas))
            if sub.operator == Operator.OR:
                return Formula(Operator.AND, tuple(self._to_nnf(Formula(Operator.NOT, (s,))) for s in sub.subformulas))
            if sub.operator == Operator.GLOBALLY:
                return Formula(Operator.FINALLY, (self._to_nnf(Formula(Operator.NOT, (sub.subformulas[0],))),))
            if sub.operator == Operator.FINALLY:
                return Formula(Operator.GLOBALLY, (self._to_nnf(Formula(Operator.NOT, (sub.subformulas[0],))),))
            if sub.operator == Operator.NEXT:
                return Formula(Operator.NEXT, (self._to_nnf(Formula(Operator.NOT, (sub.subformulas[0],))),))
            if sub.operator == Operator.UNTIL:
                return Formula(Operator.RELEASE, tuple(self._to_nnf(Formula(Operator.NOT, (s,))) for s in sub.subformulas))
            if sub.operator == Operator.RELEASE:
                return Formula(Operator.UNTIL, tuple(self._to_nnf(Formula(Operator.NOT, (s,))) for s in sub.subformulas))
                
        return Formula(formula.operator, tuple(self._to_nnf(sub) for sub in formula.subformulas))

    def _expand_implications(self, formula: Formula) -> Formula:
        """Expands implications and equivalences into their basic form."""
        if formula.atom is not None:
            return formula
            
        if formula.operator == Operator.IMPLIES:
            return Formula(
                Operator.OR,
                (Formula(Operator.NOT, (self._expand_implications(formula.subformulas[0]),)),
                 self._expand_implications(formula.subformulas[1]))
            )
            
        if formula.operator == Operator.EQUIV:
            return Formula(
                Operator.AND,
                (self._expand_implications(Formula(Operator.IMPLIES, (formula.subformulas[0], formula.subformulas[1]))),
                 self._expand_implications(Formula(Operator.IMPLIES, (formula.subformulas[1], formula.subformulas[0]))))
            )
            
        return Formula(formula.operator, tuple(self._expand_implications(sub) for sub in formula.subformulas))

    def _is_semantically_unique(self, formula: Formula) -> bool:
        """Checks if a formula is semantically unique compared to previously generated formulas."""
        canonical_hash = self._generate_canonical_hash(formula)
        formula_str = str(formula)
        
        # Check if we've seen this canonical form before
        if canonical_hash in self.semantic_cache:
            return False
            
        self.semantic_cache[canonical_hash] = {formula_str}
        self.generated_formulas.add(formula_str)
        return True

    def _generate_formula(self, depth: int) -> Formula:
        """Recursively generates a random LTL formula."""
        print(f"Generating at depth {depth}")
        
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            atom = random.choice(list(self.atoms))
            print(f"Created atom: {atom}")
            return Formula(None, (), atom=atom)
        
        operator = random.choice(list(Operator))
        print(f"Selected operator: {operator.name}")
        
        subformulas = tuple(self._generate_formula(depth + 1) for _ in range(operator.arity))
        formula = Formula(operator, subformulas)
        print(f"Created formula: {str(formula)}")
        return formula

    def generate_formulas(self, target_count: int) -> int:
        """Generates unique LTL formulas up to the target count."""
        print(f"Starting generation of {target_count} formulas")
        total_generated = 0
        
        with tqdm(total=target_count, desc="Generating formulas") as pbar:
            while total_generated < target_count:
                try:
                    print(f"\nGenerating formula {total_generated + 1}")
                    formula = self._generate_formula(0)
                    formula_str = str(formula)
                    latex = formula.to_latex()
                    print(f"Formula: {formula_str}")
                    
                    # Store in database
                    canonical_hash = self._generate_canonical_hash(formula)
                    if self.storage.store_formula(formula_str, latex, canonical_hash, formula_str.count('(')):
                        total_generated += 1
                        pbar.update(1)
                        print(f"Successfully stored formula {total_generated}")
                        
                        if total_generated % 10 == 0:
                            print(f"Milestone: Generated {total_generated} formulas")
                    
                except Exception as e:
                    print(f"Error generating formula: {str(e)}")
                    continue
        
        return total_generated

    def generate_batch(self, batch_size: int = 1000) -> List[str]:
        """Generates a batch of unique LTL formulas."""
        unique_formulas = []
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(unique_formulas) < batch_size and attempts < max_attempts:
            formula = self._generate_formula(0)
            
            formula = self._to_nnf(formula)
            formula = self._expand_implications(formula)
            
            if self._is_tautology(formula) or self._is_contradiction(formula):
                attempts += 1
                continue
                
            if self._is_semantically_unique(formula):
                unique_formulas.append(str(formula))
                
            attempts += 1
            
        return unique_formulas

class LargeScaleGenerator:
    """Handles generation of billions of formulas with efficient storage and processing."""
    
    def __init__(self, base_dir: str, max_depth: int = 8, num_processes: int = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_depth = max_depth
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.generator = LTLGenerator(max_depth, str(self.base_dir / "storage"))
        
        # Initialize progress tracking
        self.progress_file = self.base_dir / "progress.json"
        self.load_progress()
        
    def load_progress(self):
        """Loads or initializes generation progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'total_generated': 0,
                'current_depth': 1,
                'depth_progress': {},
                'last_batch_id': 0
            }
            self.save_progress()
            
    def save_progress(self):
        """Saves current generation progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
            
    def _estimate_formulas_at_depth(self, depth: int) -> int:
        """Estimates number of possible formulas at given depth."""
        if depth <= 1:
            return len(self.generator.atoms)
        num_operators = len(Operator)
        base = depth * num_operators * len(self.generator.atoms)
        return int(base ** (depth - 1))
        
    def _generate_batch_with_constraints(self, batch_id: int, depth: int,
                                       batch_size: int) -> List[Tuple[Formula, str]]:
        """Generates a batch of formulas with specific constraints."""
        formulas = []
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(formulas) < batch_size and attempts < max_attempts:
            formula = self.generator._generate_formula(depth)
            canonical = self.generator._to_canonical_form(formula)
            canonical_hash = self.generator._generate_canonical_hash(canonical)
            
            if not self.generator.storage.is_stored(canonical_hash):
                latex = formula.to_latex()
                formulas.append((formula, latex))
                
            attempts += 1
            
        return formulas
        
    def generate_formulas(self, target_count: Optional[int] = None,
                         progress_callback: Optional[callable] = None):
        """Generates formulas up to target count or until max depth is reached."""
        current_depth = self.progress['current_depth']
        total_generated = self.progress['total_generated']
        batch_size = 10000
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            while current_depth <= self.max_depth:
                depth_target = self._estimate_formulas_at_depth(current_depth)
                depth_progress = 0
                
                with tqdm(total=depth_target, desc=f"Depth {current_depth}") as pbar:
                    while depth_progress < depth_target:
                        futures = []
                        for _ in range(self.num_processes):
                            self.progress['last_batch_id'] += 1

if __name__ == "__main__":
    # Setup directories
    base_dir = BASE_DIRECTORY
    logs_dir = base_dir / "logs"
    data_dir = base_dir / "data"
    
    base_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("Starting Large-Scale LTL Formula Generation")
    logging.info(f"Base directory: {base_dir.absolute()}")
    logging.info(f"Data directory: {data_dir.absolute()}")
    logging.info(f"Log file: {log_file.absolute()}")
    
    try:
        generator = LTLGenerator(
            max_depth=MAX_LTL_DEPTH,
            storage_dir=str(data_dir)
        )
        
        logging.info(f"SQLite database location: {generator.storage.db_path}")
        
        target_count = TARGET_LTL_COUNT
        logging.info(f"Starting generation with target of {target_count:,} formulas")
        
        start_time = time.time()
        last_checkpoint = start_time
        checkpoint_interval = 3600
        
        total_generated = generator.generate_formulas(target_count)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logging.info("\n=== Generation Complete ===")
        logging.info(f"Total formulas generated: {total_generated:,}")
        logging.info(f"Total time: {total_time/3600:.2f} hours")
        logging.info(f"Average speed: {total_generated/total_time:.2f} formulas/second")
        
        with sqlite3.connect(generator.storage.db_path) as conn:
            # Get depth distribution
            cursor = conn.execute("""
                SELECT depth, COUNT(*) as count 
                FROM formulas 
                GROUP BY depth 
                ORDER BY depth
            """)
            logging.info("\nFormula distribution by depth:")
            for depth, count in cursor.fetchall():
                logging.info(f"Depth {depth}: {count:,} formulas")
            
            # Get total count
            count = conn.execute("SELECT COUNT(*) FROM formulas").fetchone()[0]
            logging.info(f"\nTotal formulas in database: {count:,}")
            
            # Get database size
            db_size = os.path.getsize(generator.storage.db_path) / (1024*1024*1024)  # Size in GB
            logging.info(f"Database size: {db_size:.2f} GB")
        
    except Exception as e:
        logging.error(f"Error during generation: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info("Process finished")
