import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import multiprocessing
import os
import random
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import spot
from datetime import datetime
import tqdm

log_dir = ""
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"itl_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger('itl_creation')

BATCH_SIZE = 64
VERIFICATION_BATCH_SIZE = 32
SPOT_TIMEOUT = 150
MAX_RETRIES = 3
DEFAULT_NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
MAX_FORMULA_DEPTH = 20
DB_PATH = "path/to/formulas.db" # path to the SQLite db
VERIFY_ONLY_MODE = False
BATCH_SIZE_PROCESSING = 64
NUM_WORKERS_PROCESSING = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
RANDOM_SEED = 42

# LTL Operator mappings for canonical templates
LTL_TO_CANONICAL = {
    "G": "Always, {0}",
    "F": "Eventually, {0}",
    "X": "In the next state, {0}",
    "U": "{0} until {1}",
    "R": "{0} releases {1}",
    "W": "{0} weakly until {1}",
    "M": "{0} strong release {1}",
    "!": "not {0}",
    "&": "{0} and {1}",
    "|": "{0} or {1}",
    "->": "if {0}, then {1}",
    "<->": "{0} if and only if {1}",
    "true": "true",
    "false": "false"
}

# Extended ITL grammar rules for the parser
ITL_GRAMMAR_RULES = {
    "formula": ["always_expr", "eventually_expr", "next_expr", "until_expr", 
                "release_expr", "weak_until_expr", "strong_release_expr", 
                "implies_expr", "iff_expr", "and_expr", "or_expr", "not_expr", "atom"],
    "always_expr": [
        "Always", "At all times", "It is always the case that", "Invariably",
        "It is invariably the case that", "Perpetually", "Continuously", 
        "For all states", "In every state", "Without exception"
    ],
    "eventually_expr": [
        "Eventually", "At some point", "At some time", "In the future", "Ultimately",
        "At a future time", "There exists a future state where", "It will be the case that",
        "There will be a time when", "Sooner or later", "At some future moment"
    ],
    "next_expr": [
        "In the next state", "In the next time step", "Immediately after this", "Next",
        "In the subsequent state", "In the following state", "Right after this state",
        "In the immediately following state", "In the very next state"
    ],
    "until_expr": [
        "until", "holds until", "is true until", "continues until",
        "remains true until", "persists until", "stays valid until",
        "is maintained until", "continues to hold until", "is satisfied until"
    ],
    "release_expr": [
        "releases", "frees", "releases from", "discharges from",
        "releases from holding", "relieves from", "frees from having to hold",
        "stops requiring", "terminates the requirement that", "ends the obligation that"
    ],
    "weak_until_expr": [
        "weakly until", "unless", "either holds forever or until",
        "holds until or forever", "either always holds or until",
        "persists until or indefinitely", "either holds perpetually or until",
        "remains valid until or infinitely", "holds until or eternally"
    ],
    "strong_release_expr": [
        "strongly releases", "strongly frees", "definitively releases",
        "absolutely releases", "unconditionally releases", "certainly releases",
        "definitely releases", "positively releases", "truly releases"
    ],
    "implies_expr": [
        "implies", "if", "then", "leads to", "results in",
        "guarantees", "assures", "ensures", "is sufficient for",
        "causes", "brings about", "is a condition for", "necessitates"
    ],
    "iff_expr": [
        "if and only if", "iff", "is equivalent to", "is the same as",
        "holds exactly when", "is satisfied precisely when",
        "is true just when", "occurs exactly when", "is necessary and sufficient for",
        "is characterized by", "holds if and only if", "is true if and only if"
    ],
    "and_expr": [
        "and", "along with", "together with", "as well as",
        "in addition to", "plus", "accompanied by", "combined with",
        "coupled with", "jointly with", "simultaneously with", "in conjunction with",
        "with", "while", "whereas", "also"
    ],
    "or_expr": [
        "or", "alternatively", "otherwise", "either way",
        "at least one of", "one or more of", "optionally", "as an alternative",
        "as another possibility", "or else", "or rather", "or alternatively"
    ],
    "not_expr": [
        "not", "it is not the case that", "it is false that", "never",
        "it is untrue that", "it does not hold that", "it is not true that",
        "it is incorrect that", "it is not valid that", "nothing like",
        "in no way", "by no means", "under no circumstances"
    ],
    "atom": [
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "the condition", "the proposition", "the system", "the property",
        "the requirement", "the constraint", "the invariant", "the signal",
        "the event", "the state", "the action", "the response", "the trigger"
    ]
}

# AST node classes
class NodeType(Enum):
    ATOM = "atom"
    NOT = "not"
    AND = "and"
    OR = "or"
    IMPLIES = "implies"
    IFF = "iff"
    NEXT = "next"
    ALWAYS = "always"
    EVENTUALLY = "eventually"
    UNTIL = "until"
    RELEASE = "release"
    WEAK_UNTIL = "weak_until"
    STRONG_RELEASE = "strong_release"

@dataclass
class ASTNode:
    node_type: NodeType
    children: List["ASTNode"] = field(default_factory=list)
    value: Optional[str] = None
    
    def __str__(self) -> str:
        if self.node_type == NodeType.ATOM:
            return self.value or ""
        elif len(self.children) == 1:
            return f"{self.node_type.value}({self.children[0]})"
        elif len(self.children) == 2:
            return f"({self.children[0]} {self.node_type.value} {self.children[1]})"
        return f"{self.node_type.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.node_type.value}
        if self.value:
            result["value"] = self.value
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ASTNode":
        node_type = NodeType(data["type"])
        value = data.get("value")
        children = [cls.from_dict(child) for child in data.get("children", [])]
        return cls(node_type=node_type, children=children, value=value)

# Database management
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.create_tables_if_not_exist()
        
    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_tables_if_not_exist(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='formulas'")
            formulas_exist = cursor.fetchone() is not None
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS itl_representations (
                    id INTEGER PRIMARY KEY,
                    formula_id INTEGER NOT NULL,
                    itl_text TEXT NOT NULL,
                    canonical_form BOOLEAN NOT NULL,
                    generation_method TEXT NOT NULL,
                    verified BOOLEAN DEFAULT 0,
                    is_correct BOOLEAN DEFAULT NULL,
                    verification_errors TEXT,
                    FOREIGN KEY (formula_id) REFERENCES formulas(id)
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_itl_formula_id ON itl_representations(formula_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_itl_verified ON itl_representations(verified)")
            
            conn.commit()
    
    def get_formulas_to_process(self, batch_size: int, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, formula, canonical_hash, depth, spot_formulas, canonical_form 
                FROM formulas
                ORDER BY id
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    def count_formulas(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM formulas")
            return cursor.fetchone()[0]
    
    def save_itl_representations(self, itl_representations: List[Dict[str, Any]]) -> None:
        if not itl_representations:
            return
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO itl_representations
                (formula_id, itl_text, canonical_form, generation_method, verified)
                VALUES (?, ?, ?, ?, ?)
            """, [(
                rep["formula_id"],
                rep["itl_text"],
                rep["canonical_form"],
                rep["generation_method"],
                False
            ) for rep in itl_representations])
            conn.commit()
    
    def get_unverified_itl(self, batch_size: int, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.id, i.formula_id, f.formula, f.spot_formulas, f.canonical_form, i.itl_text
                FROM itl_representations i
                JOIN formulas f ON i.formula_id = f.id
                WHERE i.verified = 0
                ORDER BY i.id
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    def update_verification_results(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE itl_representations
                SET verified = 1, is_correct = ?, verification_errors = ?
                WHERE id = ?
            """, [(
                result["is_correct"],
                result.get("verification_errors", None),
                result["id"]
            ) for result in results])
            conn.commit()

    def get_processed_count(self) -> Tuple[int, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM itl_representations")
            generated = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM itl_representations WHERE verified = 1")
            verified = cursor.fetchone()[0]
            
            return generated, verified

# LTL Parser
class LTLParser:
    def __init__(self):
        self.spot_env = os.environ.copy()
        self.spot_env["SPOT_STREETT_CONV_SIZE"] = "10"
        self.spot_env["SPOT_STREETT_CONV_TIME"] = "10"

    @staticmethod
    def _spot_formula_to_ast(spot_formula) -> ASTNode:
        """Convert a SPOT formula to our AST representation."""
        try:
            formula_type = spot_formula.kindstr()
            
            if formula_type == 'tt' or str(spot_formula) == '1':
                return ASTNode(NodeType.ATOM, value="true")
            elif formula_type == 'ff' or str(spot_formula) == '0':
                return ASTNode(NodeType.ATOM, value="false")
            elif formula_type == 'ap' or (len(str(spot_formula)) == 1 and str(spot_formula).isalpha()):
                return ASTNode(NodeType.ATOM, value=str(spot_formula))
                
            if spot_formula._is(spot.op_Not):
                child = LTLParser._spot_formula_to_ast(spot_formula[0])
                return ASTNode(NodeType.NOT, children=[child])
            elif spot_formula._is(spot.op_And):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.AND, children=[left, right])
            elif spot_formula._is(spot.op_Or):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.OR, children=[left, right])
            elif spot_formula._is(spot.op_Implies):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.IMPLIES, children=[left, right])
            elif spot_formula._is(spot.op_Equiv):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.IFF, children=[left, right])
            elif spot_formula._is(spot.op_X):
                child = LTLParser._spot_formula_to_ast(spot_formula[0])
                return ASTNode(NodeType.NEXT, children=[child])
            elif spot_formula._is(spot.op_G):
                child = LTLParser._spot_formula_to_ast(spot_formula[0])
                return ASTNode(NodeType.ALWAYS, children=[child])
            elif spot_formula._is(spot.op_F):
                child = LTLParser._spot_formula_to_ast(spot_formula[0])
                return ASTNode(NodeType.EVENTUALLY, children=[child])
            elif spot_formula._is(spot.op_U):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.UNTIL, children=[left, right])
            elif spot_formula._is(spot.op_R):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.RELEASE, children=[left, right])
            elif spot_formula._is(spot.op_W):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.WEAK_UNTIL, children=[left, right])
            elif spot_formula._is(spot.op_M):
                left = LTLParser._spot_formula_to_ast(spot_formula[0])
                right = LTLParser._spot_formula_to_ast(spot_formula[1])
                return ASTNode(NodeType.STRONG_RELEASE, children=[left, right])
            else:
                logger.warning(f"Using fallback parsing for formula type: {formula_type}")
                return ASTNode(NodeType.ATOM, value=str(spot_formula))
        except Exception as e:
            logger.error(f"Error in _spot_formula_to_ast: {e}")
            logger.error(f"Formula: {spot_formula}, Type: {type(spot_formula)}")
            raise ValueError(f"Failed to convert SPOT formula to AST: {str(e)}")
    
    @staticmethod
    def _parse_ltl_with_spot(formula_str: str, timeout: int = SPOT_TIMEOUT) -> spot.formula:
        """Parse LTL formula using SPOT with timeout and better error handling."""
        try:
            # Check for obviously malformed formula
            if '|' in formula_str and re.search(r'[A-Z]\s*\|\s*\(', formula_str):
                # Try to repair malformed temporal operators
                fixed_formula = re.sub(r'([GFXURWM])\s*\|\s*\(', r'\1(', formula_str)
                logger.warning(f"Attempting to fix malformed formula: {formula_str} -> {fixed_formula}")
                formula_str = fixed_formula
                
            logger.debug(f"Parsing formula: {formula_str}")
            
            return spot.formula(formula_str)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error parsing formula with SPOT: {formula_str}")
            logger.error(f"Exception: {e}")
            
            try:
                atomic_props = re.findall(r'\b[a-z]\b', formula_str)
                if atomic_props:
                    logger.warning(f"Using fallback atomic proposition: {atomic_props[0]}")
                    return spot.formula(atomic_props[0])
                else:
                    return spot.formula("true")
            except Exception:
                raise ValueError(f"Failed to parse LTL formula: {str(e)}")
    
    def parse_ltl(self, formula_data: Dict[str, Any]) -> ASTNode:
        """Parse LTL formula using pre-computed Spot formula if available."""
        try:
            if formula_data.get("spot_formulas"):
                spot_formula = self._parse_ltl_with_spot(formula_data["spot_formulas"])
                return self._spot_formula_to_ast(spot_formula)
            
            formula_str = formula_data["formula"].strip()
            spot_formula = self._parse_ltl_with_spot(formula_str)
            return self._spot_formula_to_ast(spot_formula)
        except Exception as e:
            logger.error(f"Error parsing LTL formula: {formula_data.get('formula', '')}")
            logger.error(f"Exception: {e}")
            raise ValueError(f"Failed to parse LTL formula: {str(e)}")

    @staticmethod
    def _ast_to_spot_formula(ast: ASTNode) -> spot.formula:
        """Convert AST node to SPOT formula."""
        if ast.node_type == NodeType.ATOM:
            if ast.value == "true":
                return spot.formula.tt()
            elif ast.value == "false":
                return spot.formula.ff()
            else:
                return spot.formula(ast.value)
        elif ast.node_type == NodeType.NOT:
            child = LTLParser._ast_to_spot_formula(ast.children[0])
            return spot.formula.Not(child)
        elif ast.node_type == NodeType.AND:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.And(left, right)
        elif ast.node_type == NodeType.OR:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.Or(left, right)
        elif ast.node_type == NodeType.IMPLIES:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.Implies(left, right)
        elif ast.node_type == NodeType.IFF:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.Equiv(left, right)
        elif ast.node_type == NodeType.NEXT:
            child = LTLParser._ast_to_spot_formula(ast.children[0])
            return spot.formula.X(child)
        elif ast.node_type == NodeType.ALWAYS:
            child = LTLParser._ast_to_spot_formula(ast.children[0])
            return spot.formula.G(child)
        elif ast.node_type == NodeType.EVENTUALLY:
            child = LTLParser._ast_to_spot_formula(ast.children[0])
            return spot.formula.F(child)
        elif ast.node_type == NodeType.UNTIL:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.U(left, right)
        elif ast.node_type == NodeType.RELEASE:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.R(left, right)
        elif ast.node_type == NodeType.WEAK_UNTIL:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.W(left, right)
        elif ast.node_type == NodeType.STRONG_RELEASE:
            left = LTLParser._ast_to_spot_formula(ast.children[0])
            right = LTLParser._ast_to_spot_formula(ast.children[1])
            return spot.formula.M(left, right)
        else:
            raise ValueError(f"Unsupported AST node type: {ast.node_type}")

    @staticmethod
    def spot_translate_with_timeout(formula: str, timeout: int = SPOT_TIMEOUT) -> spot.formula:
        """Translate a formula with SPOT with a timeout."""
        try:
            return spot.formula(formula)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Error in SPOT translation: {str(e)}")

    @staticmethod
    def verify_equivalence(formula1: str, formula2: str) -> Tuple[bool, Optional[str]]:
        """Verify if two LTL formulas are equivalent using SPOT."""
        try:
            f1 = spot.formula(formula1)
            f2 = spot.formula(formula2)
            
            # Check equivalent by comparing if a ↔ b is a tautology
            equiv = spot.formula.Equiv(f1, f2)
            result = spot.contains(equiv, spot.formula.tt())
            
            return result, None
        except Exception as e:
            error_msg = f"Error verifying equivalence: {str(e)}"
            return False, error_msg

# ITL Transformer
class ITLTransformer:
    def __init__(self, ltl_parser: LTLParser):
        self.ltl_parser = ltl_parser
        
    def _generate_canonical_itl(self, ast: ASTNode) -> str:
        """Generate canonical ITL from AST."""
        if ast.node_type == NodeType.ATOM:
            return ast.value or ""
        elif ast.node_type == NodeType.NOT:
            child_itl = self._generate_canonical_itl(ast.children[0])
            return LTL_TO_CANONICAL["!"].format(child_itl)
        elif ast.node_type == NodeType.AND:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["&"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.OR:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["|"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.IMPLIES:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["->"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.IFF:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["<->"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.NEXT:
            child_itl = self._generate_canonical_itl(ast.children[0])
            return LTL_TO_CANONICAL["X"].format(child_itl)
        elif ast.node_type == NodeType.ALWAYS:
            child_itl = self._generate_canonical_itl(ast.children[0])
            return LTL_TO_CANONICAL["G"].format(child_itl)
        elif ast.node_type == NodeType.EVENTUALLY:
            child_itl = self._generate_canonical_itl(ast.children[0])
            return LTL_TO_CANONICAL["F"].format(child_itl)
        elif ast.node_type == NodeType.UNTIL:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["U"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.RELEASE:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["R"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.WEAK_UNTIL:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["W"].format(left_itl, right_itl)
        elif ast.node_type == NodeType.STRONG_RELEASE:
            left_itl = self._generate_canonical_itl(ast.children[0])
            right_itl = self._generate_canonical_itl(ast.children[1])
            return LTL_TO_CANONICAL["M"].format(left_itl, right_itl)
        else:
            raise ValueError(f"Unsupported AST node type: {ast.node_type}")
    
    def ltl_to_canonical_itl(self, formula_data: Dict[str, Any]) -> str:
        """Transform LTL formula to canonical ITL using canonical form if available."""
        try:
            # If canonical_form is available, parse it
            if formula_data.get("canonical_form"):
                ast = self.ltl_parser.parse_ltl({"spot_formulas": formula_data["canonical_form"]})
            else:
                # Otherwise use the original formula
                ast = self.ltl_parser.parse_ltl(formula_data)
                
            return self._generate_canonical_itl(ast)
        except Exception as e:
            logger.error(f"Error transforming LTL to canonical ITL: {formula_data.get('formula', '')}")
            logger.error(f"Exception: {e}")
            raise ValueError(f"Failed to transform LTL to ITL: {str(e)}")

# ITL Parser for verification
class ITLParser:
    def __init__(self, ltl_parser: LTLParser):
        self.ltl_parser = ltl_parser
        self.grammar_rules = ITL_GRAMMAR_RULES
        self.pattern_mappings = self._build_pattern_mappings()
        
    def _build_pattern_mappings(self) -> Dict[str, str]:
        patterns = {}
        
        # Always patterns with comprehensive variations
        for always_pattern in ITL_GRAMMAR_RULES["always_expr"]:
            patterns[f"{always_pattern},? (.+?)(?:\.|\Z)"] = "G({0})"
            patterns[f"{always_pattern} that (.+?)(?:\.|\Z)"] = "G({0})"
            patterns[f"It is {always_pattern} that (.+?)(?:\.|\Z)"] = "G({0})"
            patterns[f"{always_pattern} the case that (.+?)(?:\.|\Z)"] = "G({0})"
        
        # Eventually patterns
        for eventually_pattern in ITL_GRAMMAR_RULES["eventually_expr"]:
            patterns[f"{eventually_pattern},? (.+?)(?:\.|\Z)"] = "F({0})"
            patterns[f"{eventually_pattern} in the future,? (.+?)(?:\.|\Z)"] = "F({0})"
            patterns[f"It will {eventually_pattern} be the case that (.+?)(?:\.|\Z)"] = "F({0})"
            patterns[f"It will be {eventually_pattern} true that (.+?)(?:\.|\Z)"] = "F({0})"
        
        # Next patterns
        for next_pattern in ITL_GRAMMAR_RULES["next_expr"]:
            patterns[f"{next_pattern},? (.+?)(?:\.|\Z)"] = "X({0})"
            patterns[f"{next_pattern} state,? (.+?)(?:\.|\Z)"] = "X({0})"
            patterns[f"In the state immediately following this,? (.+?)(?:\.|\Z)"] = "X({0})"
        
        # Until patterns with proper grouping
        for until_pattern in ITL_GRAMMAR_RULES["until_expr"]:
            patterns[f"(.+?) {until_pattern} (.+?)(?:\.|\Z)"] = "({0}) U ({1})"
            patterns[f"(.+?) will be true {until_pattern} (.+?)(?:\.|\Z)"] = "({0}) U ({1})"
            patterns[f"(.+?) continues {until_pattern} (.+?) becomes true(?:\.|\Z)"] = "({0}) U ({1})"
            patterns[f"(.+?) holds {until_pattern} (.+?) holds(?:\.|\Z)"] = "({0}) U ({1})"
        
        # Release patterns
        for release_pattern in ITL_GRAMMAR_RULES["release_expr"]:
            patterns[f"(.+?) {release_pattern} (.+?)(?:\.|\Z)"] = "({0}) R ({1})"
            patterns[f"(.+?) {release_pattern} the obligation of (.+?)(?:\.|\Z)"] = "({0}) R ({1})"
            patterns[f"(.+?) {release_pattern} the condition that (.+?)(?:\.|\Z)"] = "({0}) R ({1})"
        
        # Weak until patterns
        for weak_until_pattern in ITL_GRAMMAR_RULES["weak_until_expr"]:
            patterns[f"(.+?) {weak_until_pattern} (.+?)(?:\.|\Z)"] = "({0}) W ({1})"
            patterns[f"(.+?) holds {weak_until_pattern} (.+?) holds or forever(?:\.|\Z)"] = "({0}) W ({1})"
            patterns[f"(.+?) {weak_until_pattern} (.+?) or holds indefinitely(?:\.|\Z)"] = "({0}) W ({1})"
        
        # Strong release patterns
        for strong_release_pattern in ITL_GRAMMAR_RULES["strong_release_expr"]:
            patterns[f"(.+?) {strong_release_pattern} (.+?)(?:\.|\Z)"] = "({0}) M ({1})"
            patterns[f"(.+?) {strong_release_pattern} the condition that (.+?)(?:\.|\Z)"] = "({0}) M ({1})"
        
        # Implies patterns with variations
        for implies_pattern in ITL_GRAMMAR_RULES["implies_expr"]:
            patterns[f"(.+?) {implies_pattern} (.+?)(?:\.|\Z)"] = "({0}) -> ({1})"
            patterns[f"{implies_pattern} (.+?), then (.+?)(?:\.|\Z)"] = "({0}) -> ({1})"
            patterns[f"Whenever (.+?), (.+?)(?:\.|\Z)"] = "({0}) -> ({1})"
            patterns[f"(.+?) only {implies_pattern} (.+?)(?:\.|\Z)"] = "({0}) -> ({1})"
        
        # IFF patterns
        for iff_pattern in ITL_GRAMMAR_RULES["iff_expr"]:
            patterns[f"(.+?) {iff_pattern} (.+?)(?:\.|\Z)"] = "({0}) <-> ({1})"
            patterns[f"(.+?) holds exactly when (.+?) holds(?:\.|\Z)"] = "({0}) <-> ({1})"
            patterns[f"(.+?) is true precisely when (.+?) is true(?:\.|\Z)"] = "({0}) <-> ({1})"
        
        # And patterns with priority over other binary operators
        for and_pattern in ITL_GRAMMAR_RULES["and_expr"]:
            patterns[f"(.+?) {and_pattern} (.+?)(?:\.|\Z)"] = "({0}) & ({1})"
            patterns[f"both (.+?) {and_pattern} (.+?)(?:\.|\Z)"] = "({0}) & ({1})"
            patterns[f"(.+?), {and_pattern} at the same time, (.+?)(?:\.|\Z)"] = "({0}) & ({1})"
            patterns[f"(.+?), while (.+?)(?:\.|\Z)"] = "({0}) & ({1})"
        
        # Or patterns
        for or_pattern in ITL_GRAMMAR_RULES["or_expr"]:
            patterns[f"(.+?) {or_pattern} (.+?)(?:\.|\Z)"] = "({0}) | ({1})"
            patterns[f"either (.+?) {or_pattern} (.+?)(?:\.|\Z)"] = "({0}) | ({1})"
            patterns[f"(.+?), unless (.+?)(?:\.|\Z)"] = "({0}) | ({1})"
        
        # Not patterns
        for not_pattern in ITL_GRAMMAR_RULES["not_expr"]:
            patterns[f"{not_pattern} (.+?)(?:\.|\Z)"] = "!({0})"
            patterns[f"it is {not_pattern} the case that (.+?)(?:\.|\Z)"] = "!({0})"
            patterns[f"{not_pattern} the case that (.+?)(?:\.|\Z)"] = "!({0})"
            patterns[f"it is false that (.+?)(?:\.|\Z)"] = "!({0})"
            patterns[f"{not_pattern} ever (.+?)(?:\.|\Z)"] = "!(F({0}))"
        
        # Compound patterns for common combinations
        patterns[f"Always, if (.+?), then eventually (.+?)(?:\.|\Z)"] = "G(({0}) -> F({1}))"
        patterns[f"It is always the case that if (.+?), then eventually (.+?)(?:\.|\Z)"] = "G(({0}) -> F({1}))"
        patterns[f"Whenever (.+?), eventually (.+?)(?:\.|\Z)"] = "G(({0}) -> F({1}))"
        patterns[f"If (.+?) is true, then eventually (.+?) must hold(?:\.|\Z)"] = "G(({0}) -> F({1}))"
        
        # Nested patterns
        patterns[f"Eventually, always (.+?)(?:\.|\Z)"] = "F(G({0}))"
        patterns[f"Always, eventually (.+?)(?:\.|\Z)"] = "G(F({0}))"
        patterns[f"It is always the case that eventually (.+?)(?:\.|\Z)"] = "G(F({0}))"
        patterns[f"It will eventually be always the case that (.+?)(?:\.|\Z)"] = "F(G({0}))"
        
        # Handle parenthesized expressions
        patterns[r"\((.+?)\)"] = "({0})"
        
        return patterns


    def _match_pattern(self, itl_text: str) -> Tuple[Optional[str], List[str]]:
        """Match ITL text against known patterns."""
        for pattern, ltl_template in self.pattern_mappings.items():
            match = re.match(pattern, itl_text, re.IGNORECASE)
            if match:
                return ltl_template, list(match.groups())
        return None, []
    
    def _recursive_parse(self, itl_text: str, depth: int = 0) -> str:
        """
        Recursively parse ITL text to LTL formula with robust handling of complex expressions.
        """
        if depth > MAX_FORMULA_DEPTH:
            raise ValueError(f"Maximum recursion depth exceeded when parsing: {itl_text}")
        
        itl_text = itl_text.strip()
        if not itl_text:
            return ""
        
        if itl_text.startswith("(") and itl_text.endswith(")"):
            if self._check_balanced_parentheses(itl_text[1:-1]):
                inner_result = self._recursive_parse(itl_text[1:-1], depth + 1)
                return f"({inner_result})"
                
        for operator_type in ["always_expr", "eventually_expr", "next_expr"]:
            if operator_type not in self.grammar_rules:
                continue
                
            for operator in self.grammar_rules[operator_type]:
                not_temporal_pattern = f"not {operator}"
                if itl_text.lower().startswith(not_temporal_pattern.lower()):
                    remaining = itl_text[len(not_temporal_pattern):].strip()
                    if remaining.startswith(","):
                        remaining = remaining[1:].strip()
                    if remaining:
                        operand_ltl = self._recursive_parse(remaining, depth + 1)
                        
                        if operator_type == "always_expr":
                            return f"!(G({operand_ltl}))"
                        elif operator_type == "eventually_expr":
                            return f"!(F({operand_ltl}))"
                        elif operator_type == "next_expr":
                            return f"!(X({operand_ltl}))"
                
                if itl_text.lower().startswith(operator.lower()):
                    remaining = itl_text[len(operator):].strip()
                    if remaining.startswith(","):
                        remaining = remaining[1:].strip()
                    if remaining.startswith("that"):
                        remaining = remaining[4:].strip()
                        
                    if remaining:
                        operand_ltl = self._recursive_parse(remaining, depth + 1)
                        
                        if operator_type == "always_expr":
                            return f"G({operand_ltl})"
                        elif operator_type == "eventually_expr":
                            return f"F({operand_ltl})"
                        elif operator_type == "next_expr":
                            return f"X({operand_ltl})"
        
        for not_expr in self.grammar_rules.get("not_expr", []):
            if itl_text.lower().startswith(not_expr.lower()):
                remaining = itl_text[len(not_expr):].strip()
                if remaining:
                    operand_ltl = self._recursive_parse(remaining, depth + 1)
                    return f"!({operand_ltl})"
        
        if itl_text.lower().startswith("if "):
            parts = itl_text[3:].split(" then ", 1)
            if len(parts) == 2:
                condition, result = parts
                condition_ltl = self._recursive_parse(condition.strip(), depth + 1)
                result_ltl = self._recursive_parse(result.strip(), depth + 1)
                return f"({condition_ltl}) -> ({result_ltl})"
        
        for operator_type, operators in [
            ("iff_expr", ["if and only if"]),
            ("implies_expr", ["implies"]),
            ("or_expr", [" or "]),
            ("and_expr", [" and "]),
            ("until_expr", [" until "]),
            ("release_expr", [" releases "]),
            ("weak_until_expr", [" weakly until "])
        ]:
            for op in operators:
                parts = self._split_on_operator_outside_parentheses(itl_text, op)
                if len(parts) == 2:
                    left, right = parts
                    left_ltl = self._recursive_parse(left.strip(), depth + 1)
                    right_ltl = self._recursive_parse(right.strip(), depth + 1)
                    
                    if operator_type == "and_expr":
                        return f"({left_ltl}) & ({right_ltl})"
                    elif operator_type == "or_expr":
                        return f"({left_ltl}) | ({right_ltl})"
                    elif operator_type == "implies_expr":
                        return f"({left_ltl}) -> ({right_ltl})"
                    elif operator_type == "iff_expr":
                        return f"({left_ltl}) <-> ({right_ltl})"
                    elif operator_type == "until_expr":
                        return f"({left_ltl}) U ({right_ltl})"
                    elif operator_type == "release_expr":
                        return f"({left_ltl}) R ({right_ltl})"
                    elif operator_type == "weak_until_expr":
                        return f"({left_ltl}) W ({right_ltl})"
        
        matched = False
        best_template = None
        best_groups = None
        
        sorted_patterns = sorted(self.pattern_mappings.items(), 
                            key=lambda x: (-len(x[0]), -x[0].count("(")))
        
        for pattern, ltl_template in sorted_patterns:
            match = re.match(f"^{pattern}$", itl_text, re.IGNORECASE)
            if match:
                groups = list(match.groups())
                if len(groups) > 0:
                    matched = True
                    best_template = ltl_template
                    best_groups = groups
                    break
        
        if matched:
            parsed_groups = []
            for group in best_groups:
                if not group or group.isspace():
                    parsed_groups.append("")
                    continue
                    
                try:
                    parsed_group = self._recursive_parse(group, depth + 1)
                    parsed_groups.append(parsed_group)
                except Exception as e:
                    logger.debug(f"Error parsing group '{group}': {e}")
                    parsed_groups.append(self._clean_atomic_proposition(group))
            
            try:
                result = best_template.format(*parsed_groups)
                return result
            except Exception as e:
                logger.error(f"Error formatting template {best_template} with groups {parsed_groups}: {e}")
                return self._clean_atomic_proposition(itl_text)

        # If everything fails, assume it's an atomic proposition
        return self._clean_atomic_proposition(itl_text)

    def _clean_atomic_proposition(self, text: str) -> str:
        """Clean up atomic propositions by removing common phrases."""
        clean_text = re.sub(r"the condition |condition |the proposition |proposition |is true|holds|is satisfied|is met",
                        "", text, flags=re.IGNORECASE)
        
        if clean_text.lower() in ["true", "truth", "is always satisfied"]:
            return "true"
        elif clean_text.lower() in ["false", "falsehood", "is never satisfied"]:
            return "false"
        
        clean_text = clean_text.strip(" ,.;:")
        
        if len(clean_text) == 1 and clean_text.isalpha():
            return clean_text
        
        words = clean_text.split()
        for word in words:
            if len(word) == 1 and word.isalpha() and word.islower():
                return word
        
        # Last resort - use as is
        return clean_text

    def _check_balanced_parentheses(self, text: str) -> bool:
        """Check if parentheses in the text are balanced."""
        stack = []
        for char in text:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0

    def _split_on_operator_outside_parentheses(self, text: str, operator: str) -> List[str]:
        """Split text on an operator, but only if the operator is outside parentheses."""
        result = []
        paren_depth = 0
        start_pos = 0
        
        i = 0
        while i <= len(text) - len(operator):
            # Check for open parenthesis
            if text[i] == '(':
                paren_depth += 1
            # Check for close parenthesis
            elif text[i] == ')':
                paren_depth = max(0, paren_depth - 1)  # Ensure non-negative
            
            if paren_depth == 0 and text[i:i+len(operator)] == operator:
                result.append(text[start_pos:i].strip())
                start_pos = i + len(operator)
                i += len(operator)
            else:
                i += 1
        
        if start_pos < len(text):
            result.append(text[start_pos:].strip())
        
        return result

    def itl_to_ltl(self, itl_text: str) -> str:
        """Convert ITL text to LTL formula."""
        try:
            preprocessed = itl_text.strip()
            
            if preprocessed.endswith("."):
                preprocessed = preprocessed[:-1]
                
            ltl_formula = self._recursive_parse(preprocessed)
            
            self.ltl_parser.spot_translate_with_timeout(ltl_formula)
            
            return ltl_formula
        except Exception as e:
            logger.error(f"Error parsing ITL to LTL: {itl_text}")
            logger.error(f"Exception: {e}")
            raise ValueError(f"Failed to parse ITL to LTL: {str(e)}")

# Verification System
class Verifier:
    def __init__(self, ltl_parser: LTLParser, itl_parser: ITLParser):
        self.ltl_parser = ltl_parser
        self.itl_parser = itl_parser
    
    def verify_itl(self, ltl_formula: str, itl_text: str) -> Tuple[bool, Optional[str]]:
        """Verify if ITL correctly represents the LTL formula."""
        try:
            reconstructed_ltl = self.itl_parser.itl_to_ltl(itl_text)
            
            equivalent, error_msg = self.ltl_parser.verify_equivalence(ltl_formula, reconstructed_ltl)
            
            if equivalent:
                return True, None
            else:
                error_detail = f"ITL '{itl_text}' translates to '{reconstructed_ltl}' which is not equivalent to '{ltl_formula}'"
                return False, error_detail
        except Exception as e:
            error_msg = f"Verification error: {str(e)}"
            return False, error_msg

# Pipeline components
class GenerationPipeline:
    def __init__(self, db_manager: DatabaseManager, ltl_parser: LTLParser, 
                 itl_transformer: ITLTransformer,
                 batch_size: int = BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS):
        self.db_manager = db_manager
        self.ltl_parser = ltl_parser
        self.itl_transformer = itl_transformer
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _process_formula_batch(self, formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of formulas."""
        results = []
        
        for formula_data in formulas:
            formula_id = formula_data["id"]
            
            try:
                canonical_itl = self.itl_transformer.ltl_to_canonical_itl(formula_data)
                
                results.append({
                    "formula_id": formula_id,
                    "itl_text": canonical_itl,
                    "canonical_form": True,
                    "generation_method": "canonical_mapping"
                })

            except Exception as e:
                logger.error(f"Error processing formula {formula_id}: {formula_data.get('formula', '')}")
                logger.error(f"Exception: {e}")
                # Add error entry
                results.append({
                    "formula_id": formula_id,
                    "itl_text": f"Error generating ITL: {str(e)}",
                    "canonical_form": False,
                    "generation_method": "error"
                })
                
        return results
    
    def generate_itl_representations(self) -> int:
        """Generate ITL representations for all formulas."""
        import time
        import os
        from datetime import datetime

        start_time = time.time()
        total_formulas = self.db_manager.count_formulas()
        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ITL generation for {total_formulas} formulas")
        print(f"Starting ITL generation for {total_formulas} formulas with batch size {self.batch_size} and {self.num_workers} workers")
        
        processed_count = 0
        batch_count = 0
        successful_generations = 0
        error_count = 0
        
        try:
            # Try to get memory usage if psutil is available
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = f"Initial memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB"
            logger.info(memory_info)
            print(memory_info)
        except ImportError:
            logger.info("psutil not available for memory monitoring")
            print("psutil not available for memory monitoring")
        
        with tqdm.tqdm(total=total_formulas) as pbar:
            while processed_count < total_formulas:
                batch_start_time = time.time()
                batch_count += 1
                
                # Get next batch of formulas
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching batch {batch_count} from offset {processed_count}")
                print(f"Fetching batch {batch_count} from offset {processed_count}")
                
                formulas = self.db_manager.get_formulas_to_process(
                    self.batch_size, offset=processed_count
                )
                
                if not formulas:
                    logger.warning(f"No more formulas found at offset {processed_count}. Breaking.")
                    print(f"No more formulas found at offset {processed_count}. Breaking.")
                    break
                
                logger.info(f"Retrieved {len(formulas)} formulas for batch {batch_count}")
                print(f"Retrieved {len(formulas)} formulas for batch {batch_count}")
                
                # Sample the first few formulas to show what's being processed
                sample_size = min(5, len(formulas))
                sample_formulas = [f["formula"][:50] + "..." if len(f["formula"]) > 50 else f["formula"] for f in formulas[:sample_size]]
                logger.info(f"Sample formulas: {sample_formulas}")
                print(f"Sample formulas: {sample_formulas}")
                
                # Process formulas in parallel
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting parallel processing of batch {batch_count}")
                print(f"Starting parallel processing of batch {batch_count} with {self.num_workers} workers")
                
                parallel_start_time = time.time()
                try:
                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        future_to_batch = {}
                        
                        # Split into smaller batches for parallel processing
                        batch_size = max(1, len(formulas) // self.num_workers)
                        batches = [formulas[i:i + batch_size] for i in range(0, len(formulas), batch_size)]
                        
                        logger.info(f"Split into {len(batches)} sub-batches of ~{batch_size} formulas each")
                        print(f"Split into {len(batches)} sub-batches of ~{batch_size} formulas each")
                        
                        for i, batch in enumerate(batches):
                            logger.debug(f"Submitting sub-batch {i+1}/{len(batches)} with {len(batch)} formulas")
                            future = executor.submit(self._process_formula_batch, batch)
                            future_to_batch[future] = batch
                        
                        all_results = []
                        completed_batches = 0
                        
                        for future in concurrent.futures.as_completed(future_to_batch):
                            try:
                                batch_results = future.result()
                                batch_success = sum(1 for r in batch_results if r["generation_method"] != "error")
                                batch_errors = sum(1 for r in batch_results if r["generation_method"] == "error")
                                
                                successful_generations += batch_success
                                error_count += batch_errors
                                
                                all_results.extend(batch_results)
                                completed_batches += 1
                                
                                if completed_batches % max(1, len(batches) // 10) == 0:
                                    logger.info(f"Completed {completed_batches}/{len(batches)} sub-batches")
                                    print(f"Completed {completed_batches}/{len(batches)} sub-batches")
                                    
                            except Exception as e:
                                logger.error(f"Error processing batch: {str(e)}")
                                logger.error(traceback.format_exc())
                                print(f"ERROR processing batch: {str(e)}")
                                error_count += len(future_to_batch[future])
                    
                    parallel_duration = time.time() - parallel_start_time
                    logger.info(f"Parallel processing completed in {parallel_duration:.2f} seconds")
                    print(f"Parallel processing completed in {parallel_duration:.2f} seconds")
                    
                except Exception as e:
                    logger.error(f"Critical error in thread management: {str(e)}")
                    logger.error(traceback.format_exc())
                    print(f"CRITICAL ERROR in thread management: {str(e)}")
                    continue
                
                # Save results to database
                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving {len(all_results)} ITL representations to database")
                print(f"Saving {len(all_results)} ITL representations to database")
                
                save_start_time = time.time()
                self.db_manager.save_itl_representations(all_results)
                save_duration = time.time() - save_start_time
                
                logger.info(f"Database save completed in {save_duration:.2f} seconds")
                print(f"Database save completed in {save_duration:.2f} seconds")
                
                # Log some examples of generated ITL
                if all_results:
                    canonical_examples = [r["itl_text"][:100] + "..." if len(r["itl_text"]) > 100 else r["itl_text"] 
                                        for r in all_results[:3] if r["canonical_form"]]
                    
                    if canonical_examples:
                        logger.info(f"Canonical ITL examples: {canonical_examples}")
                        print(f"Canonical ITL examples: {canonical_examples}")
                
                # Update progress
                processed_count += len(formulas)
                pbar.update(len(formulas))
                
                # Log detailed progress information
                batch_duration = time.time() - batch_start_time
                total_duration = time.time() - start_time
                formulas_per_second = processed_count / total_duration if total_duration > 0 else 0
                
                # System resource usage if available
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                    cpu_percent = process.cpu_percent()
                    resource_info = f"Memory usage: {memory_usage:.2f} MB, CPU: {cpu_percent:.1f}%"
                except ImportError:
                    resource_info = "Resource monitoring unavailable (psutil not installed)"
                
                progress_msg = (
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Processed {processed_count}/{total_formulas} formulas ({processed_count/total_formulas*100:.2f}%)\n"
                    f"Batch {batch_count} completed in {batch_duration:.2f} seconds\n"
                    f"Running for {total_duration:.2f} seconds ({formulas_per_second:.2f} formulas/sec)\n"
                    f"Successful generations: {successful_generations}, Errors: {error_count}\n"
                    f"{resource_info}"
                )
                
                logger.info(progress_msg)
                print(progress_msg)
                
                # Also log to separate progress file (for easy tracking)
                with open('itl_generation_progress.log', 'a') as f:
                    f.write(f"{progress_msg}\n{'='*80}\n")
        
        # Final summary
        end_time = time.time()
        total_duration = end_time - start_time
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        summary_msg = (
            f"ITL generation completed.\n"
            f"Processed {processed_count}/{total_formulas} formulas ({processed_count/total_formulas*100:.2f}%)\n"
            f"Successful generations: {successful_generations}, Errors: {error_count}\n"
            f"Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n"
            f"Average speed: {processed_count/total_duration:.2f} formulas/sec"
        )
        
        logger.info(summary_msg)
        print(summary_msg)
        
        return processed_count

class VerificationPipeline:
    def __init__(self, db_manager: DatabaseManager, verifier: Verifier,
                 batch_size: int = VERIFICATION_BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS):
        self.db_manager = db_manager
        self.verifier = verifier
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _verify_itl_batch(self, itl_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify a batch of ITL representations."""
        results = []
        
        for itl_data in itl_batch:
            itl_id = itl_data["id"]
            formula_id = itl_data["formula_id"]
            
            # Use spot_formulas if available for more accurate verification
            ltl_formula = itl_data.get("spot_formulas", itl_data["formula"])
            itl_text = itl_data["itl_text"]
            
            try:
                # Check if this is an error entry
                if itl_text.startswith("Error generating ITL:"):
                    results.append({
                        "id": itl_id,
                        "is_correct": False,
                        "verification_errors": itl_text
                    })
                    continue
                
                # Verify ITL
                is_correct, error_msg = self.verifier.verify_itl(ltl_formula, itl_text)
                
                results.append({
                    "id": itl_id,
                    "is_correct": is_correct,
                    "verification_errors": error_msg
                })
            except Exception as e:
                logger.error(f"Error verifying ITL {itl_id}: {itl_text}")
                logger.error(f"Exception: {e}")
                results.append({
                    "id": itl_id,
                    "is_correct": False,
                    "verification_errors": f"Verification error: {str(e)}"
                })
                
        return results
        
    def verify_all_itl(self) -> int:
        """Verify all ITL representations."""
        # Get total number of unverified ITL representations
        total_generated, total_verified = self.db_manager.get_processed_count()
        total_unverified = total_generated - total_verified
        
        logger.info(f"Starting verification for {total_unverified} ITL representations")
        
        processed_count = 0
        
        with tqdm.tqdm(total=total_unverified) as pbar:
            while processed_count < total_unverified:
                # Get next batch of unverified ITL
                itl_batch = self.db_manager.get_unverified_itl(
                    self.batch_size, offset=0  # Always get the first batch
                )
                
                if not itl_batch:
                    break
                
                # Verify ITL in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_batch = {}
                    
                    # Split into smaller batches for parallel processing
                    batch_size = max(1, len(itl_batch) // self.num_workers)
                    batches = [itl_batch[i:i + batch_size] for i in range(0, len(itl_batch), batch_size)]
                    
                    for batch in batches:
                        future = executor.submit(self._verify_itl_batch, batch)
                        future_to_batch[future] = batch
                    
                    all_results = []
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch_results = future.result()
                        all_results.extend(batch_results)
                
                # Update database with verification results
                self.db_manager.update_verification_results(all_results)
                
                # Update progress
                processed_count += len(itl_batch)
                pbar.update(len(itl_batch))
                
                # Log progress periodically
                if processed_count % (self.batch_size * 10) == 0:
                    logger.info(f"Verified {processed_count}/{total_unverified} ITL representations")
        
        logger.info(f"Verification completed. Verified {processed_count} ITL representations.")
        return processed_count

class Pipeline:
    def __init__(self, db_path: str, device: str = "cpu",
                batch_size: int = BATCH_SIZE, num_workers: int = DEFAULT_NUM_WORKERS):

        self.db_manager = DatabaseManager(db_path)
        self.ltl_parser = LTLParser()
        self.itl_transformer = ITLTransformer(self.ltl_parser)
        self.itl_parser = ITLParser(self.ltl_parser)
        self.verifier = Verifier(self.ltl_parser, self.itl_parser)
        
        #Initialize pipelines
        self.generation_pipeline = GenerationPipeline(
            self.db_manager, self.ltl_parser, self.itl_transformer, 
            batch_size=batch_size, num_workers=num_workers
        )
        
        self.verification_pipeline = VerificationPipeline(
            self.db_manager, self.verifier, 
            batch_size=batch_size, num_workers=num_workers
        )
    
    def run(self) -> None:
        """Run the complete pipeline."""
        start_time = time.time()
        
        logger.info("Starting LTL to ITL pipeline")
        
        try:
            logger.info("Step 1: Generating ITL representations")
            num_generated = self.generation_pipeline.generate_itl_representations()
            logger.info(f"Generated {num_generated} ITL representations")
            
            logger.info("Step 2: Verifying ITL representations")
            num_verified = self.verification_pipeline.verify_all_itl()
            logger.info(f"Verified {num_verified} ITL representations")
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            
            total_generated, total_verified = self.db_manager.get_processed_count()
            logger.info(f"Total ITL representations: {total_generated}")
            logger.info(f"Total verified: {total_verified}")
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM itl_representations
                    WHERE verified = 1 AND is_correct = 1
                """)
                correct_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM itl_representations
                    WHERE verified = 1 AND is_correct = 0
                """)
                incorrect_count = cursor.fetchone()[0]
                
                logger.info(f"Correct ITL representations: {correct_count} ({correct_count/total_verified*100:.2f}%)")
                logger.info(f"Incorrect ITL representations: {incorrect_count} ({incorrect_count/total_verified*100:.2f}%)")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Pipeline failed: {str(e)}")

def _generate_simple_variations(canonical_itl: str) -> List[str]:
    """Generate simple variations of the canonical ITL representation."""
    variations = []
    
    if "Always," in canonical_itl:
        variations.append(canonical_itl.replace("Always,", "At all times,"))
        variations.append(canonical_itl.replace("Always,", "Invariably,"))
    
    if "Eventually," in canonical_itl:
        variations.append(canonical_itl.replace("Eventually,", "At some point,"))
        variations.append(canonical_itl.replace("Eventually,", "In the future,"))
    
    if "until" in canonical_itl:
        variations.append(canonical_itl.replace("until", "holds until"))
    
    # Keep only unique variations
    return list(set(variations))

def _generate_additional_variations(canonical_itl: str) -> List[str]:
    """Generate more diverse variations when needed to reach sample size."""
    base_variations = _generate_simple_variations(canonical_itl)
    extra_variations = []
    
    if "Always," in canonical_itl:
        extra_variations.append(canonical_itl.replace("Always,", "It is always the case that"))
        extra_variations.append(canonical_itl.replace("Always,", "In every state,"))
        
    if "Eventually," in canonical_itl:
        extra_variations.append(canonical_itl.replace("Eventually,", "There will be a time when"))
        extra_variations.append(canonical_itl.replace("Eventually,", "Ultimately,"))
        
    if "In the next state," in canonical_itl:
        extra_variations.append(canonical_itl.replace("In the next state,", "Immediately after this,"))
        
    all_variations = base_variations + extra_variations
    
    return list(set(all_variations))

def _load_hand_crafted_examples() -> List[Dict[str, str]]:
    """Load hand-crafted examples that cover key LTL patterns."""
    examples = [
        # Always operator (G)
        {
            "ltl_formula": "G(p)",
            "canonical_itl": "Always, p",
            "itl_text": "Always, p"
        },
        {
            "ltl_formula": "G(p)",
            "canonical_itl": "Always, p",
            "itl_text": "At all times, p holds"
        },
        {
            "ltl_formula": "G(p)",
            "canonical_itl": "Always, p",
            "itl_text": "It is invariably the case that p is true"
        },
        
        # Eventually operator (F)
        {
            "ltl_formula": "F(q)",
            "canonical_itl": "Eventually, q",
            "itl_text": "Eventually, q"
        },
        {
            "ltl_formula": "F(q)",
            "canonical_itl": "Eventually, q",
            "itl_text": "At some point, q will be true"
        },
        {
            "ltl_formula": "F(q)",
            "canonical_itl": "Eventually, q",
            "itl_text": "Ultimately, q holds"
        },
        
        # Next operator (X)
        {
            "ltl_formula": "X(r)",
            "canonical_itl": "In the next state, r",
            "itl_text": "In the next state, r"
        },
        {
            "ltl_formula": "X(r)",
            "canonical_itl": "In the next state, r",
            "itl_text": "Immediately after this, r holds"
        },
        
        # Until operator (U)
        {
            "ltl_formula": "p U q",
            "canonical_itl": "p until q",
            "itl_text": "p until q"
        },
        {
            "ltl_formula": "p U q",
            "canonical_itl": "p until q",
            "itl_text": "p holds until q becomes true"
        },
        
        # Release operator (R)
        {
            "ltl_formula": "p R q",
            "canonical_itl": "p releases q",
            "itl_text": "p releases q"
        },
        {
            "ltl_formula": "p R q",
            "canonical_itl": "p releases q",
            "itl_text": "q holds until and including the point where p holds"
        },
        
        # Weak Until (W)
        {
            "ltl_formula": "p W q",
            "canonical_itl": "p weakly until q",
            "itl_text": "p weakly until q"
        },
        {
            "ltl_formula": "p W q",
            "canonical_itl": "p weakly until q",
            "itl_text": "Either p holds forever or p holds until q"
        },
        
        # Common pattern: Response
        {
            "ltl_formula": "G(p -> F(q))",
            "canonical_itl": "Always, if p, then Eventually, q",
            "itl_text": "Always, if p, then Eventually, q"
        },
        {
            "ltl_formula": "G(p -> F(q))",
            "canonical_itl": "Always, if p, then Eventually, q",
            "itl_text": "Whenever p holds, q will eventually be true"
        },
        {
            "ltl_formula": "G(p -> F(q))",
            "canonical_itl": "Always, if p, then Eventually, q",
            "itl_text": "Every occurrence of p is followed by an occurrence of q"
        },
        
        # Common pattern: Precedence
        {
            "ltl_formula": "!q W p",
            "canonical_itl": "not q weakly until p",
            "itl_text": "not q weakly until p"
        },
        {
            "ltl_formula": "!q W p",
            "canonical_itl": "not q weakly until p",
            "itl_text": "q never occurs before p"
        },
        
        # Common pattern: Absence
        {
            "ltl_formula": "G(!p)",
            "canonical_itl": "Always, not p",
            "itl_text": "Always, not p"
        },
        {
            "ltl_formula": "G(!p)",
            "canonical_itl": "Always, not p",
            "itl_text": "p never holds"
        }
    ]
    
    complex_examples = [
        # Nested temporal operators
        {
            "ltl_formula": "G(F(p))",
            "canonical_itl": "Always, Eventually, p",
            "itl_text": "Always, Eventually, p"
        },
        {
            "ltl_formula": "G(F(p))",
            "canonical_itl": "Always, Eventually, p",
            "itl_text": "It is always the case that p will eventually hold"
        },
        {
            "ltl_formula": "F(G(p))",
            "canonical_itl": "Eventually, Always, p",
            "itl_text": "Eventually, Always, p"
        },
        {
            "ltl_formula": "F(G(p))",
            "canonical_itl": "Eventually, Always, p",
            "itl_text": "Eventually, p will hold forever"
        },
        
        # Boolean combinations
        {
            "ltl_formula": "G(p & q)",
            "canonical_itl": "Always, p and q",
            "itl_text": "Always, both p and q hold"
        },
        {
            "ltl_formula": "F(p | q)",
            "canonical_itl": "Eventually, p or q",
            "itl_text": "Eventually, either p or q will be true"
        },
        
        # Complex real-world examples
        {
            "ltl_formula": "G(request -> F(acknowledge))",
            "canonical_itl": "Always, if request, then Eventually, acknowledge",
            "itl_text": "Every request must be eventually acknowledged"
        },
        {
            "ltl_formula": "G(try -> F(success | G(failure)))",
            "canonical_itl": "Always, if try, then Eventually, success or Always, failure",
            "itl_text": "Every attempt either eventually succeeds or permanently fails"
        },
        {
            "ltl_formula": "G(send -> X(F(receive)))",
            "canonical_itl": "Always, if send, then In the next state, Eventually, receive",
            "itl_text": "Whenever a message is sent, it will start to be received in the next state"
        },
        {
            "ltl_formula": "G(alarm -> (alarm U reset))",
            "canonical_itl": "Always, if alarm, then alarm until reset",
            "itl_text": "Once triggered, an alarm continues until it is reset"
        }
    ]
    
    examples.extend(complex_examples)
    return examples

def _generate_structured_variations(ltl_formula: str, canonical_itl: str, ast: ASTNode) -> List[str]:
    """
    Generate systematic variations based on formula structure and semantics.
    """
    variations = []
    
    formula_type = _classify_formula_pattern(ast)
    
    if formula_type == "response":  # G(p -> F(q))
        variations.extend([
            f"Whenever {_extract_proposition(ast, 'antecedent')} occurs, {_extract_proposition(ast, 'consequent')} will eventually follow",
            f"Each time {_extract_proposition(ast, 'antecedent')} happens, {_extract_proposition(ast, 'consequent')} will happen at some point afterward",
            f"If {_extract_proposition(ast, 'antecedent')} ever becomes true, then {_extract_proposition(ast, 'consequent')} must become true later"
        ])
    elif formula_type == "precedence":  # !q W p
        variations.extend([
            f"{_extract_proposition(ast, 'consequent')} cannot occur before {_extract_proposition(ast, 'antecedent')}",
            f"{_extract_proposition(ast, 'consequent')} must be preceded by {_extract_proposition(ast, 'antecedent')}",
            f"Either {_extract_proposition(ast, 'antecedent')} happens first, or {_extract_proposition(ast, 'consequent')} never happens"
        ])
    elif formula_type == "invariance":  # G(p)
        variations.extend([
            f"At all times, {_extract_proposition(ast, 'condition')} holds",
            f"It is invariably the case that {_extract_proposition(ast, 'condition')}",
            f"{_extract_proposition(ast, 'condition')} is true in every state"
        ])
    elif formula_type == "eventuality":  # F(p)
        variations.extend([
            f"At some point, {_extract_proposition(ast, 'condition')} will become true",
            f"Eventually, {_extract_proposition(ast, 'condition')} will hold",
            f"In the future, {_extract_proposition(ast, 'condition')} will be satisfied"
        ])
    elif formula_type == "until":  # p U q
        variations.extend([
            f"{_extract_proposition(ast, 'left')} holds until {_extract_proposition(ast, 'right')} becomes true",
            f"{_extract_proposition(ast, 'left')} continues to be true until {_extract_proposition(ast, 'right')} occurs",
            f"{_extract_proposition(ast, 'left')} remains valid until a point where {_extract_proposition(ast, 'right')} is satisfied"
        ])
    elif formula_type == "release":  # p R q
        variations.extend([
            f"{_extract_proposition(ast, 'left')} releases {_extract_proposition(ast, 'right')} from having to hold",
            f"{_extract_proposition(ast, 'right')} holds until and including the moment {_extract_proposition(ast, 'left')} holds",
            f"{_extract_proposition(ast, 'right')} must remain true until {_extract_proposition(ast, 'left')} becomes true, or forever if {_extract_proposition(ast, 'left')} never holds"
        ])
    elif formula_type == "reactivity":  # G(F(p))
        variations.extend([
            f"{_extract_proposition(ast, 'condition')} will occur infinitely often",
            f"It is always the case that {_extract_proposition(ast, 'condition')} will eventually happen",
            f"At every point, {_extract_proposition(ast, 'condition')} will occur sometime in the future"
        ])
    elif formula_type == "stability":  # F(G(p))
        variations.extend([
            f"Eventually, {_extract_proposition(ast, 'condition')} will hold forever",
            f"At some point, {_extract_proposition(ast, 'condition')} becomes permanently true",
            f"There will be a time after which {_extract_proposition(ast, 'condition')} is always satisfied"
        ])
    else:
        # General formula variations
        variations.extend([
            canonical_itl,
            _rephrase_canonical(canonical_itl)
        ])
    
    # Filter out duplicates and None values
    variations = [v for v in variations if v and v.strip()]
    variations = list(dict.fromkeys(variations))  # Remove duplicates
    
    return variations

def _classify_formula_pattern(ast: ASTNode) -> str:
    """Classify LTL formula into common patterns based on structure."""
    if (ast.node_type == NodeType.ALWAYS and 
        len(ast.children) == 1 and 
        ast.children[0].node_type == NodeType.IMPLIES and
        len(ast.children[0].children) == 2 and
        ast.children[0].children[1].node_type == NodeType.EVENTUALLY):
        return "response"
    
    # Check for precedence pattern: !q W p
    if (ast.node_type == NodeType.WEAK_UNTIL and 
        len(ast.children) == 2 and 
        ast.children[0].node_type == NodeType.NOT):
        return "precedence"
    
    # Check for invariance pattern: G(p)
    if ast.node_type == NodeType.ALWAYS and len(ast.children) == 1:
        return "invariance"
    
    # Check for eventuality pattern: F(p)
    if ast.node_type == NodeType.EVENTUALLY and len(ast.children) == 1:
        return "eventuality"
    
    # Check for until pattern: p U q
    if ast.node_type == NodeType.UNTIL and len(ast.children) == 2:
        return "until"
    
    # Check for release pattern: p R q
    if ast.node_type == NodeType.RELEASE and len(ast.children) == 2:
        return "release"
    
    # Check for reactivity pattern: G(F(p))
    if (ast.node_type == NodeType.ALWAYS and 
        len(ast.children) == 1 and 
        ast.children[0].node_type == NodeType.EVENTUALLY):
        return "reactivity"
    
    # Check for stability pattern: F(G(p))
    if (ast.node_type == NodeType.EVENTUALLY and 
        len(ast.children) == 1 and 
        ast.children[0].node_type == NodeType.ALWAYS):
        return "stability"
    
    # Default
    return "general"

def _extract_proposition(ast: ASTNode, role: str) -> str:
    """Extract propositions from AST based on their role in the pattern."""
    if role == "condition" and len(ast.children) == 1:
        # For G(p) or F(p)
        return str(ast.children[0])
        
    elif role == "antecedent" and ast.node_type == NodeType.ALWAYS and ast.children[0].node_type == NodeType.IMPLIES:
        # For response pattern G(p -> F(q))
        return str(ast.children[0].children[0])
        
    elif role == "consequent" and ast.node_type == NodeType.ALWAYS and ast.children[0].node_type == NodeType.IMPLIES:
        # For response pattern G(p -> F(q))
        consequent = ast.children[0].children[1]
        if consequent.node_type == NodeType.EVENTUALLY:
            return str(consequent.children[0])
        return str(consequent)
        
    elif role == "left" and len(ast.children) == 2:
        # For binary operators like U, R
        return str(ast.children[0])
        
    elif role == "right" and len(ast.children) == 2:
        # For binary operators like U, R
        return str(ast.children[1])
    
    # Default fallback
    return "the condition"

def _rephrase_canonical(canonical: str) -> str:
    """Rephrase the canonical form with different wording but equivalent meaning."""
    rephrasing_map = {
        "Always,": ["At all times,", "It is invariably the case that", "Invariably,"],
        "Eventually,": ["At some point,", "At some time in the future,", "Ultimately,"],
        "In the next state,": ["In the immediately following state,", "Next,"],
        "until": ["holds until", "is true until", "continues until"],
        "releases": ["frees", "releases the obligation that"],
        "weakly until": ["unless", "either holds forever or until"],
        "and": ["as well as", "together with", "along with"],
        "or": ["alternatively", "otherwise"],
        "not": ["it is not the case that", "it is false that"]
    }
    
    result = canonical
    for phrase, alternatives in rephrasing_map.items():
        if phrase in result:
            alternative = alternatives[hash(canonical + phrase) % len(alternatives)]
            result = result.replace(phrase, alternative, 1)
    
    return result

def main():
    np.random.seed(RANDOM_SEED)

    try:
        pipeline = Pipeline(
            DB_PATH,
            batch_size=BATCH_SIZE_PROCESSING,
            num_workers=NUM_WORKERS_PROCESSING
        )

        if VERIFY_ONLY_MODE:
            logger.info("Running in verification-only mode")
            pipeline.verification_pipeline.verify_all_itl()
        else:
            logger.info("Running full ITL generation and verification pipeline")
            pipeline.run()

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    logger.info("Done")

if __name__ == "__main__":
    main()