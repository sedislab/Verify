import sqlite3
import re
import os
import logging
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import spot

DATA_DIRECTORY = ""
LOGS_DIRECTORY = ""
BATCH_SIZE = 
RUN_TESTS = False
TEST_SPECIFIC_FORMULA = None
VALIDATE_DB_ONLY = False
TEST_SYSTEM = False

def setup_logging(logs_dir):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"spot_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def add_spot_column(db_path):
    """Add spot_formulas column to the formulas table if it doesn't exist"""
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("PRAGMA table_info(formulas)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'spot_formulas' not in columns:
            conn.execute("ALTER TABLE formulas ADD COLUMN spot_formulas TEXT")
            logging.info("Added spot_formulas column to the database")
        else:
            logging.info("spot_formulas column already exists")

class FormulaParser:
    """Enhanced parser for converting LaTeX LTL formulas to Spot format"""
    
    def __init__(self):
        # Map of LaTeX operators to Spot operators
        self.operator_map = {
            "\\land": "&",
            "\\lor": "|",
            "\\neg": "!",
            "\\rightarrow": "->",
            "\\leftrightarrow": "<->",
            # Temporal operators remain the same
            "G": "G",
            "F": "F",
            "X": "X",
            "U": "U",
            "R": "R",
            "W": "W",
            "M": "M"
        }
        
        self.precedence = {
            "!": 1,     # NOT
            "G": 1,     # GLOBALLY
            "F": 1,     # FINALLY
            "X": 1,     # NEXT
            "&": 2,     # AND
            "|": 3,     # OR
            "U": 4,     # UNTIL
            "R": 4,     # RELEASE
            "W": 4,     # WEAK UNTIL
            "M": 4,     # STRONG RELEASE
            "->": 5,    # IMPLIES
            "<->": 6    # EQUIVALENT
        }
    
    def parse(self, latex_formula):
        """Main parsing function to convert LaTeX formula to Spot format"""
        try:
            normalized = self._normalize_formula(latex_formula)
            return self._parse_recursive(normalized)
        except Exception as e:
            logging.warning(f"Error in parser: {str(e)}. Formula: {latex_formula}")
            return self._basic_conversion(latex_formula)
    
    def _normalize_formula(self, formula):
        """Normalize the LaTeX formula for easier parsing"""
        for op in ["\\land", "\\lor", "\\rightarrow", "\\leftrightarrow", "U", "R", "W", "M"]:
            pattern = f"(?<![\\\\s]){re.escape(op)}(?![\\\\s])"
            formula = re.sub(pattern, f" {op} ", formula)
        
        formula = re.sub(r'(\})\s*([GFXURWM])\s*(\{)', r'\1 \2 \3', formula)
        
        return formula.strip()
    
    def _parse_recursive(self, formula):
        """Recursively parse the formula handling nesting correctly"""
        formula = formula.strip()
        
        if not formula:
            return ""
        
        if re.match(r'^[a-zA-Z0-9_]+$', formula):
            return formula
        
        if formula.startswith("\\left(") and formula.endswith("\\right)"):
            inner = formula[6:-6].strip()
            return f"({self._parse_recursive(inner)})"
        
        if formula.startswith("(") and formula.endswith(")"):
            inner = formula[1:-1].strip()
            inner_parsed = self._parse_recursive(inner)
            return f"({inner_parsed})"
        
        for op in ["\\neg", "G", "F", "X"]:
            brace_pattern = f"^{re.escape(op)}\\s*\\{{(.+?)\\}}$"
            paren_pattern = f"^{re.escape(op)}\\s*\\((.+?)\\)$"
            
            brace_match = re.match(brace_pattern, formula, re.DOTALL)
            if brace_match:
                subformula = brace_match.group(1).strip()
                spot_op = self.operator_map.get(op, op)
                sub_result = self._parse_recursive(subformula)
                
                if op == "\\neg":
                    if self._needs_parentheses(sub_result, "!"):
                        return f"!({sub_result})"
                    return f"!{sub_result}"
                else:
                    return f"{spot_op}({sub_result})"
            
            paren_match = re.match(paren_pattern, formula, re.DOTALL)
            if paren_match:
                subformula = paren_match.group(1).strip()
                spot_op = self.operator_map.get(op, op)
                sub_result = self._parse_recursive(subformula)
                
                if op == "\\neg":
                    if self._needs_parentheses(sub_result, "!"):
                        return f"!({sub_result})"
                    return f"!{sub_result}"
                else:
                    return f"{spot_op}({sub_result})"
        
        main_op_info = self._find_main_operator(formula)
        if main_op_info:
            op, start_idx = main_op_info
            spot_op = self.operator_map.get(op, op)
            
            left = formula[:start_idx].strip()
            right = formula[start_idx + len(op):].strip()
            
            left_result = self._parse_recursive(left)
            right_result = self._parse_recursive(right)
            
            if self._needs_parentheses(left_result, spot_op, is_left=True):
                left_result = f"({left_result})"
            
            if self._needs_parentheses(right_result, spot_op, is_left=False):
                right_result = f"({right_result})"
            
            return f"{left_result} {spot_op} {right_result}"
        
        neg_match = re.match(r'^\\neg\s+(.+)$', formula)
        if neg_match:
            subformula = neg_match.group(1).strip()
            sub_result = self._parse_recursive(subformula)
            
            return f"!({sub_result})"
        
        tokens = self._tokenize_formula(formula)
        
        if tokens:
            return self._parse_tokens(tokens)
        
        return self._basic_conversion(formula)
    
    def _tokenize_formula(self, formula):
        """Split formula into tokens, preserving nested structures"""
        tokens = []
        i = 0
        current_token = ""
        brace_level = 0
        paren_level = 0
        
        while i < len(formula):
            char = formula[i]
            
            if char == '{':
                brace_level += 1
                current_token += char
            elif char == '}':
                brace_level -= 1
                current_token += char
            elif char == '(':
                paren_level += 1
                current_token += char
            elif char == ')':
                paren_level -= 1
                current_token += char
            elif char == '\\':
                for op in ["\\land", "\\lor", "\\neg", "\\rightarrow", "\\leftrightarrow"]:
                    if formula[i:i+len(op)] == op and brace_level == 0 and paren_level == 0:
                        if current_token:
                            tokens.append(current_token.strip())
                        tokens.append(op)
                        current_token = ""
                        i += len(op)
                        break
                else:
                    current_token += char
                    i += 1
            elif char in "GFXURWM" and brace_level == 0 and paren_level == 0:
                if current_token:
                    tokens.append(current_token.strip())
                tokens.append(char)
                current_token = ""
                i += 1
            elif char.isspace() and brace_level == 0 and paren_level == 0:
                if current_token:
                    tokens.append(current_token.strip())
                    current_token = ""
                i += 1
            else:
                current_token += char
                i += 1
        
        if current_token:
            tokens.append(current_token.strip())
        
        return tokens
    
    def _parse_tokens(self, tokens):
        """Parse a list of tokens into a Spot formula"""
        if len(tokens) == 1:
            return self._parse_recursive(tokens[0])
        
        for op in sorted(self.operator_map.keys(), key=lambda x: self._get_precedence(self.operator_map.get(x, x))):
            if op in tokens:
                idx = tokens.index(op)
                left_tokens = tokens[:idx]
                right_tokens = tokens[idx+1:]
                
                left_result = self._parse_tokens(left_tokens) if left_tokens else ""
                right_result = self._parse_tokens(right_tokens) if right_tokens else ""
                
                spot_op = self.operator_map.get(op, op)
                
                if left_result and self._needs_parentheses(left_result, spot_op, is_left=True):
                    left_result = f"({left_result})"
                
                if right_result and self._needs_parentheses(right_result, spot_op, is_left=False):
                    right_result = f"({right_result})"
                
                if op in ["\\neg", "G", "F", "X"]:
                    return f"{spot_op}({right_result})"
                
                return f"{left_result} {spot_op} {right_result}"
        
        return " ".join(tokens)
    
    def _find_main_operator(self, formula):
        """Find the main binary operator at the top level with correct precedence"""
        # Track nesting levels
        brace_level = 0
        paren_level = 0
        latex_paren_level = 0
        
        operators = []
        
        i = 0
        while i < len(formula):
            if formula[i] == '{':
                brace_level += 1
            elif formula[i] == '}':
                brace_level -= 1
            elif formula[i] == '(':
                paren_level += 1
            elif formula[i] == ')':
                paren_level -= 1
            elif i + 5 < len(formula) and formula[i:i+6] == "\\left(":
                latex_paren_level += 1
                i += 5
            elif i + 6 < len(formula) and formula[i:i+7] == "\\right)":
                latex_paren_level -= 1
                i += 6
                
            if brace_level == 0 and paren_level == 0 and latex_paren_level == 0:
                for op in ["\\land", "\\lor", "\\rightarrow", "\\leftrightarrow", "U", "R", "W", "M"]:
                    if i + len(op) <= len(formula) and formula[i:i+len(op)] == op:
                        is_valid = False
                        if len(op) == 1:
                            prev_char = formula[i-1] if i > 0 else None
                            next_char = formula[i+1] if i+1 < len(formula) else None
                            is_valid = (prev_char is None or prev_char.isspace() or prev_char in '})}') and \
                                       (next_char is None or next_char.isspace() or next_char in '{([')
                        else:
                            is_valid = True
                        
                        if is_valid:
                            operators.append((op, i))
                        i += len(op) - 1
                        break
            
            i += 1
        
        if operators:
            sorted_ops = sorted(operators, key=lambda x: self._get_precedence(self.operator_map.get(x[0], x[0])))
            return sorted_ops[-1]
        
        return None
    
    def _get_precedence(self, op):
        """Get the precedence value of an operator (lower = higher precedence)"""
        return self.precedence.get(op, 10)
    
    def _needs_parentheses(self, sub_formula, parent_op, is_left=True):
        """Determine if a subformula needs parentheses based on operator precedence"""
        if not sub_formula or ' ' not in sub_formula:
            return False
        
        if parent_op in ["!", "G", "F", "X"]:
            return True
        
        parts = sub_formula.split(' ', 2)
        if len(parts) < 3:
            return False
        
        child_op = parts[1]
        
        parent_precedence = self._get_precedence(parent_op)
        child_precedence = self._get_precedence(child_op)
        
        if child_precedence > parent_precedence:
            return True
        
        if child_precedence == parent_precedence:
            if parent_op in ["&", "|"] and child_op == parent_op:
                return False
            
            if not is_left:
                return True
        
        return False
    
    def _basic_conversion(self, formula):
        """Basic fallback conversion that handles simple replacements"""
        result = formula
        
        for latex_op, spot_op in self.operator_map.items():
            result = result.replace(latex_op, spot_op)
        
        result = result.replace("\\left(", "(").replace("\\right)", ")")
        while '{' in result:
            result = re.sub(r'\{([^{}]*)\}', r'\1', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

def validate_with_spot(formula_str):
    """
    Validate a formula string with Spot
    """
    try:
        formula = spot.formula(formula_str)
        
        return True, str(formula)
    except Exception as e:
        return False, str(e)

def test_formula_conversion(latex_formula):
    """Test conversion of a specific formula and print the result with Spot validation"""
    parser = FormulaParser()
    spot_formula = parser.parse(latex_formula)
    
    is_valid, spot_interpretation = validate_with_spot(spot_formula)
    
    print(f"LaTeX: {latex_formula}")
    print(f"Spot: {spot_formula}")
    print(f"Valid: {'VALID' if is_valid else 'NOT VALID'}")
    if is_valid:
        print(f"Interpreted: {spot_interpretation}")
    else:
        print(f"Error: {spot_interpretation}")
    print("-" * 50)
    return spot_formula, is_valid, spot_interpretation

def test_complex_examples():
    """Test the parser with known complex examples, including Spot validation"""
    examples = [
        "u \\land \\neg {G {\\neg {\\neg {w \\leftrightarrow s}}} U u} \\lor s",
        "G {p} \\land F {\\neg {p}}",
        "p \\lor \\neg {p}",
        "p \\rightarrow (q \\lor r)",
        "\\neg {\\neg {p}} \\leftrightarrow p",
        "G {F {p}} \\rightarrow F {G {p}}",
        "p U (q U r)",
        "(p U q) U r",
        "\\neg {G {p}} \\leftrightarrow F {\\neg {p}}",
        "(G (p \\rightarrow F q)) \\land (G (r \\rightarrow F s))",
        "G (p \\rightarrow (q U r)) \\lor (s U (\\neg {t}))",
        "\\neg{(p U q) R (r M s)}",
        "G (p \\rightarrow (F q \\land X (r \\lor s)))",
        "F G ((p \\lor q) \\land (r -> s))"
    ]
    
    logging.info("\n=== Testing Complex Examples with Spot Validation ===")
    parser = FormulaParser()
    
    total_count = len(examples)
    valid_count = 0
    
    for i, example in enumerate(examples):
        try:
            spot_formula = parser.parse(example)
            
            is_valid, spot_interpretation = validate_with_spot(spot_formula)
            
            if is_valid:
                valid_count += 1
                status = "VALID"
            else:
                status = "NOT VALID"
                
            logging.info(f"Example {i+1}: {status}")
            logging.info(f"LaTeX: {example}")
            logging.info(f"Spot: {spot_formula}")
            if is_valid:
                logging.info(f"Interpreted: {spot_interpretation}")
            else:
                logging.info(f"Error: {spot_interpretation}")
                
        except Exception as e:
            logging.error(f"Failed to parse example {i+1}: {example}")
            logging.error(f"Error: {str(e)}")
    
    logging.info(f"\nResults: {valid_count}/{total_count} formulas successfully validated")
    logging.info("===========================\n")
    
    return valid_count == total_count

def add_columns(db_path):
    """Add spot_formulas and canonical_form columns to the formulas table if they don't exist"""
    with sqlite3.connect(str(db_path)) as conn:
        # Check if columns exist
        cursor = conn.execute("PRAGMA table_info(formulas)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'spot_formulas' not in columns:
            conn.execute("ALTER TABLE formulas ADD COLUMN spot_formulas TEXT")
            logging.info("Added spot_formulas column to the database")
        else:
            logging.info("spot_formulas column already exists")
            
        if 'canonical_form' not in columns:
            conn.execute("ALTER TABLE formulas ADD COLUMN canonical_form TEXT")
            logging.info("Added canonical_form column to the database")
        else:
            logging.info("canonical_form column already exists")

def update_database(db_path, batch_size=1000):
    """Update all formulas in the database with both Spot syntax and canonical form"""
    parser = FormulaParser()
    start_time = time.time()
    processed_formulas = 0
    conversion_failures = 0
    validation_failures = 0
    
    with sqlite3.connect(str(db_path)) as conn:
        remaining_formulas = conn.execute(
            "SELECT COUNT(*) FROM formulas WHERE spot_formulas IS NULL OR canonical_form IS NULL"
        ).fetchone()[0]
        
        logging.info(f"Remaining formulas to process: {remaining_formulas:,}")
        
        with tqdm(total=remaining_formulas, desc="Converting formulas") as pbar:
            while True:
                cursor = conn.execute(
                    "SELECT id, latex FROM formulas WHERE spot_formulas IS NULL OR canonical_form IS NULL LIMIT ?",
                    (batch_size,)
                )
                batch = cursor.fetchall()
                if not batch:
                    break
                
                updates = []
                for formula_id, latex in batch:
                    try:
                        spot_formula = parser.parse(latex)
                        
                        is_valid, spot_interpretation = validate_with_spot(spot_formula)
                        
                        if is_valid:
                            updates.append((spot_formula, spot_interpretation, formula_id))
                        else:
                            logging.warning(f"Validation failed for formula {formula_id}: {spot_interpretation}")
                            validation_failures += 1
                            
                            basic_formula = parser._basic_conversion(latex)
                            is_valid_basic, spot_interpretation_basic = validate_with_spot(basic_formula)
                            
                            if is_valid_basic:
                                logging.info(f"Basic conversion succeeded for formula {formula_id}")
                                updates.append((basic_formula, spot_interpretation_basic, formula_id))
                            else:
                                logging.warning(f"Basic conversion also failed for formula {formula_id}")
                                updates.append((f"ERROR: {spot_formula}", "ERROR", formula_id))
                                
                    except Exception as e:
                        logging.error(f"Error converting formula {formula_id}: {str(e)}")
                        conversion_failures += 1
                
                if updates:
                    conn.executemany(
                        "UPDATE formulas SET spot_formulas = ?, canonical_form = ? WHERE id = ?",
                        updates
                    )
                    conn.commit()
                
                processed_count = len(batch)
                processed_formulas += processed_count
                pbar.update(processed_count)
                
                if processed_formulas <= batch_size:
                    display_examples(updates[:5])
                
                if processed_formulas % (batch_size * 10) == 0:
                    remaining = conn.execute(
                        "SELECT COUNT(*) FROM formulas WHERE spot_formulas IS NULL OR canonical_form IS NULL"
                    ).fetchone()[0]
                    logging.info(f"Remaining formulas to process: {remaining:,}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info("\n=== Conversion Summary ===")
    logging.info(f"Total formulas processed: {processed_formulas:,}")
    logging.info(f"Conversion failures: {conversion_failures:,}")
    logging.info(f"Validation failures: {validation_failures:,}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info(f"Average speed: {processed_formulas/duration:.2f} formulas/second")
    logging.info(f"Success rate: {(processed_formulas - conversion_failures - validation_failures) / processed_formulas * 100:.2f}%")
    
    remaining = conn.execute(
        "SELECT COUNT(*) FROM formulas WHERE spot_formulas IS NULL OR canonical_form IS NULL"
    ).fetchone()[0]
    
    if remaining > 0:
        logging.warning(f"There are still {remaining:,} formulas that need to be processed!")
    else:
        logging.info("All formulas have been successfully processed!")
        
def display_examples(updates, count=5):
    """Display some example conversions for verification"""
    logging.info("\n=== Example Conversions ===")
    for i, (spot_formula, canonical_form, _) in enumerate(updates[:count]):
        logging.info(f"Example {i+1}:")
        logging.info(f"Spot syntax: {spot_formula}")
        logging.info(f"Canonical: {canonical_form}")
    logging.info("===========================\n")

def validate_existing_formulas(db_path, batch_size=1000):
    """Validate existing formulas in the database and update canonical forms"""
    start_time = time.time()
    total_formulas = 0
    valid_formulas = 0
    invalid_formulas = 0
    
    with sqlite3.connect(str(db_path)) as conn:
        total_formulas = conn.execute(
            "SELECT COUNT(*) FROM formulas WHERE spot_formulas IS NOT NULL"
        ).fetchone()[0]
        
        if total_formulas == 0:
            logging.info("No spot_formulas found in the database to validate")
            return
            
        logging.info(f"Validating {total_formulas:,} existing spot_formulas")
        
        offset = 0
        with tqdm(total=total_formulas, desc="Validating formulas") as pbar:
            while True:
                cursor = conn.execute(
                    "SELECT id, spot_formulas FROM formulas WHERE spot_formulas IS NOT NULL LIMIT ? OFFSET ?",
                    (batch_size, offset)
                )
                batch = cursor.fetchall()
                if not batch:
                    break
                
                updates = []
                for formula_id, spot_formula in batch:
                    if spot_formula.startswith("ERROR:"):
                        invalid_formulas += 1
                        continue
                        
                    is_valid, spot_interpretation = validate_with_spot(spot_formula)
                    
                    if is_valid:
                        valid_formulas += 1
                        updates.append((spot_formula, spot_interpretation, formula_id))
                    else:
                        invalid_formulas += 1
                        logging.warning(f"Invalid formula {formula_id}: {spot_formula}")
                        logging.warning(f"  Error: {spot_interpretation}")
                        updates.append((f"ERROR: {spot_formula}", "ERROR", formula_id))
                
                if updates:
                    conn.executemany(
                        "UPDATE formulas SET spot_formulas = ?, canonical_form = ? WHERE id = ?",
                        updates
                    )
                    conn.commit()
                
                pbar.update(len(batch))
                offset += batch_size
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info("\n=== Validation Summary ===")
    logging.info(f"Total formulas validated: {total_formulas:,}")
    logging.info(f"Valid formulas: {valid_formulas:,}")
    logging.info(f"Invalid formulas: {invalid_formulas:,}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info(f"Average speed: {total_formulas/duration:.2f} formulas/second")
    logging.info(f"Success rate: {valid_formulas / total_formulas * 100:.2f}%")

def main():
    log_file = setup_logging(LOGS_DIRECTORY)
    logging.info(f"Log file created at: {log_file}")

    if TEST_SYSTEM:
        latex_example = "u \\land \\neg {G {\\neg {\\neg {w \\leftrightarrow s}}} U u} \\lor s"
        spot_formula = FormulaParser().parse(latex_example)
        is_valid, canonical_form = validate_with_spot(spot_formula)

        print(f"Test formula conversion:")
        print(f"LaTeX: {latex_example}")
        print(f"Spot syntax: {spot_formula}")
        print(f"Canonical: {canonical_form}")
        print(f"Valid: {'VALID' if is_valid else 'NOT VALID'}")

        if not is_valid:
            logging.warning(f"Warning: Test formula validation failed. Conversion may have issues.")

            if input("Continue with database conversion anyway? (y/n): ").lower() != 'y':
                logging.info("Conversion aborted by user")
                return 1

    if RUN_TESTS:
        test_complex_examples()
        return 0

    if TEST_SPECIFIC_FORMULA:
        latex_formula = TEST_SPECIFIC_FORMULA
        spot_formula = FormulaParser().parse(latex_formula)
        is_valid, canonical_form = validate_with_spot(spot_formula)

        print(f"LaTeX formula: {latex_formula}")
        print(f"Spot syntax: {spot_formula}")
        print(f"Valid: {'VALID' if is_valid else 'NOT VALID'}")
        print(f"Canonical form: {canonical_form}")
        return 0

    data_dir = Path(DATA_DIRECTORY)
    db_path = data_dir / "formulas.db"

    if not db_path.exists():
        logging.error(f"Error: Database not found at {db_path}")
        return 1

    logging.info(f"Database path: {db_path}")

    try:
        add_columns(db_path)

        if VALIDATE_DB_ONLY:
            validate_existing_formulas(db_path, BATCH_SIZE)
        else:
            update_database(db_path, BATCH_SIZE)

        logging.info("Conversion completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Error during conversion: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())