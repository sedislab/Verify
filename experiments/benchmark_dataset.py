import os
import spot
import sys
import json
import time
import math
import shutil
import sqlite3
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
import zss
import evaluate
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    T5ForConditionalGeneration, BartForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, 
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except ImportError:
    print("Warning: NLTK not found. METEOR scores will not be available.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('verify_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEBUG_MODE = False
DEBUG_DATA_LIMIT = 200

if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPUs available")
    for i in range(n_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name} with {gpu_mem:.2f} GB memory")
else:
    n_gpus = 0
    logger.warning("No GPU found, using CPU")

RANDOM_SEED = 42

CONFIG = {
    'db_path': '',
    'db_copy_path': '',
    'model_dir': '',
    'results_dir': '',
    'human_eval_samples': 100,
    'test_size': 0.1,
    'val_size': 0.1,
    'batch_size': 16 if n_gpus > 0 else 4,
    'eval_batch_size': 4 if n_gpus > 0 else 2,
    'max_source_length': 256,
    'max_target_length': 256,
    'num_beams': 1, #5
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'num_train_epochs': 3, #3
    'warmup_ratio': 0.1,
    'logging_steps': 100,
    'eval_steps': 500, #3
    'save_steps': 1000,
    'gradient_accumulation_steps': 2,
    'fp16': False,
    'bf16':True,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 3,
    'early_stopping_threshold': 0.01,
    'checkpointing': True,
    'checkpoint_dir': '',
    'use_deepspeed': True
}

# Model configurations
MODEL_CONFIGS = {
    # Seq2Seq models
    't5-base': {
        'name': 't5-base',
        'type': 'seq2seq',
        'class': T5ForConditionalGeneration,
        'tokenizer': 'T5Tokenizer'
    },
    't5-large': {
        'name': 't5-large',
        'type': 'seq2seq',
        'class': T5ForConditionalGeneration,
        'tokenizer': 'T5Tokenizer'
    },
    'bart-base': {
        'name': 'facebook/bart-base',
        'type': 'seq2seq',
        'class': BartForConditionalGeneration,
        'tokenizer': 'BartTokenizer'
    },
    'bart-large': {
        'name': 'facebook/bart-large',
        'type': 'seq2seq',
        'class': BartForConditionalGeneration,
        'tokenizer': 'BartTokenizer'
    },
    
    # LLMs
    'llama-3-8b': {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'type': 'causal',
        'class': AutoModelForCausalLM,
        'tokenizer': 'LlamaTokenizer'
    },
    'mistral-7b': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.2',
        'type': 'causal',
        'class': AutoModelForCausalLM,
        'tokenizer': 'AutoTokenizer'
    },
    
    # Code-focused LLMs
    'codellama-7b': {
        'name': 'codellama/CodeLlama-7b-Instruct-hf',
        'type': 'causal',
        'class': AutoModelForCausalLM,
        'tokenizer': 'LlamaTokenizer'
    },
    'deepseek-coder': {
        'name': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'type': 'causal',
        'class': AutoModelForCausalLM,
        'tokenizer': 'AutoTokenizer'
    }
}

# Task definitions
TASK_CONFIGS = {
    'ltl_to_nl': {
        'name': 'ltl_to_nl',
        'human_name': 'LTL to Natural Language',
        'input_format': 'translate LTL to NL: domain: {domain} activity: {activity} ltl: {formula}',
        'output_format': '{translation}',
        'metrics': ['bertscore', 'rouge-l', 'bleu', 'meteor'],
        'primary_metrics': ['bertscore', 'rouge-l']
    },
    'itl_to_nl': {
        'name': 'itl_to_nl',
        'human_name': 'ITL to Natural Language',
        'input_format': 'translate ITL to NL: domain: {domain} activity: {activity} itl: {itl_text}',
        'output_format': '{translation}',
        'metrics': ['bertscore', 'rouge-l', 'bleu', 'meteor'],
        'primary_metrics': ['bertscore', 'rouge-l']
    },
    'nl_to_ltl': {
        'name': 'nl_to_ltl',
        'human_name': 'Natural Language to LTL',
        'input_format': 'translate NL to LTL: domain: {domain} activity: {activity} nl: {translation}',
        'output_format': '{formula}',
        'metrics': ['exact_match', 'normalized_exact_match', 'semantic_equivalence', 'tree_edit_distance', 'syntactic_correctness'],
        'primary_metrics': ['semantic_equivalence']
    },
    'nl_to_itl': {
        'name': 'nl_to_itl',
        'human_name': 'Natural Language to ITL',
        'input_format': 'translate NL to ITL: domain: {domain} activity: {activity} nl: {translation}',
        'output_format': '{itl_text}',
        'metrics': ['exact_match', 'tree_edit_distance', 'syntactic_correctness'],
        'primary_metrics': ['exact_match']
    },
    'ltl_to_itl': {
        'name': 'ltl_to_itl',
        'human_name': 'LTL to ITL',
        'input_format': 'translate LTL to ITL: ltl: {formula}',
        'output_format': '{itl_text}',
        'metrics': ['exact_match', 'tree_edit_distance', 'syntactic_correctness'],
        'primary_metrics': ['exact_match']
    },
    'itl_to_ltl': {
        'name': 'itl_to_ltl',
        'human_name': 'ITL to LTL',
        'input_format': 'translate ITL to LTL: itl: {itl_text}',
        'output_format': '{formula}',
        'metrics': ['exact_match', 'semantic_equivalence', 'tree_edit_distance', 'syntactic_correctness'],
        'primary_metrics': ['exact_match', 'semantic_equivalence']
    }
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'value_of_itl': {
        'name': 'value_of_itl',
        'human_name': 'The Value of ITL',
        'description': 'Compare direct LTL->NL vs two-stage LTL->ITL->NL',
        'tasks': ['ltl_to_nl', 'ltl_to_itl', 'itl_to_nl'],
        'variants': ['direct', 'two_stage']
    },
    'impact_of_context': {
        'name': 'impact_of_context',
        'human_name': 'The Impact of Context',
        'description': 'Compare performance with and without domain/activity context',
        'tasks': ['ltl_to_nl', 'itl_to_nl'],
        'variants': ['with_context', 'without_context']
    },
    'domain_generalization': {
        'name': 'domain_generalization',
        'human_name': 'Domain Generalization & Adaptation',
        'description': 'Assess cross-domain performance and few-shot adaptation',
        'tasks': ['ltl_to_nl', 'nl_to_ltl'],
        'variants': ['lodo', 'few_shot'],
        'few_shot_k': [1, 5, 10, 25, 50]
    },
    'sensitivity_to_complexity': {
        'name': 'sensitivity_to_complexity',
        'human_name': 'Sensitivity to Logical Complexity',
        'description': 'Assess performance across varying LTL complexity levels',
        'tasks': ['ltl_to_nl', 'nl_to_ltl'],
        'complexity_measures': ['operators', 'depth', 'propositions']
    }
}

def setup_directory_structure():
    """Create necessary directories if they don't exist."""
    directories = [
        os.path.dirname(CONFIG['db_copy_path']),
        CONFIG['model_dir'],
        CONFIG['results_dir'],
        os.path.join(CONFIG['results_dir'], 'tasks'),
        os.path.join(CONFIG['results_dir'], 'experiments'),
        os.path.join(CONFIG['results_dir'], 'human_eval'),
        CONFIG['checkpoint_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory confirmed: {directory}")

def copy_database():
    """Make a copy of the database if it doesn't already exist."""
    if not os.path.exists(CONFIG['db_copy_path']):
        logger.info(f"Creating a copy of the database at {CONFIG['db_copy_path']}")
        os.makedirs(os.path.dirname(CONFIG['db_copy_path']), exist_ok=True)
        shutil.copy2(CONFIG['db_path'], CONFIG['db_copy_path'])
        logger.info("Database copy completed successfully")
    else:
        logger.info(f"Database copy already exists at {CONFIG['db_copy_path']}")

def load_data_from_db():
    """Load and join data from all three tables."""
    logger.info("Loading data from database...")
    
    conn = sqlite3.connect(CONFIG['db_copy_path'])
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM formulas")
    formula_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM itl_representations")
    itl_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM nl_translations")
    nl_count = cursor.fetchone()[0]
    
    logger.info(f"Database contains: {formula_count} formulas, {itl_count} ITL representations, {nl_count} NL translations")
    
    cursor.execute("PRAGMA table_info(itl_representations)")
    columns = cursor.fetchall()
    has_is_correct = any(col[1] == 'is_correct' for col in columns)
    
    if has_is_correct:
        query = """
        SELECT f.id AS formula_id, f.formula, f.canonical_form AS ltl_canonical,
               f.depth, i.id AS itl_id, i.itl_text, i.canonical_form AS itl_canonical,
               n.id AS nl_id, n.domain, n.activity, n.translation
        FROM formulas f
        JOIN itl_representations i ON f.id = i.formula_id
        JOIN nl_translations n ON f.id = n.formula_id AND i.id = n.itl_id
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
    
    logger.info(f"Loaded {len(rows)} valid entries with formula, ITL, and NL data")
    
    data = [dict(row) for row in rows]

    if DEBUG_MODE:
        logger.warning(f"DEBUG MODE: Limiting data to {DEBUG_DATA_LIMIT} samples.")
        if len(data) > DEBUG_DATA_LIMIT:
            data = random.sample(data, DEBUG_DATA_LIMIT)
        else:
            logger.warning(f"Dataset smaller than DEBUG_DATA_LIMIT ({len(data)}), using all loaded data.")
    
    cursor.execute("SELECT domain FROM domain_stats")
    domains = [row[0] for row in cursor.fetchall()]
    logger.info(f"Found {len(domains)} unique domains: {domains}")
    
    conn.close()
    return data, domains

def split_data(data, test_size=0.1, val_size=0.1, stratify_by='domain'):
    """Split data into train/val/test sets, stratified by domain if specified."""
    logger.info(f"Splitting data into train/val/test sets (stratified by {stratify_by if stratify_by else 'None'})")
    
    if stratify_by:
        stratify = [item[stratify_by] for item in data]
    else:
        stratify = None
    
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify
    )
    
    if stratify_by:
        stratify = [item[stratify_by] for item in train_val]
    else:
        stratify = None
        
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=RANDOM_SEED, stratify=stratify
    )
    
    logger.info(f"Data split: {len(train)} train, {len(val)} validation, {len(test)} test samples")
    
    return train, val, test

class VerifyDataset(Dataset):
    """Dataset class for Verify benchmark."""
    
    def __init__(self, data, task_config, tokenizer, max_source_length, max_target_length, include_context=True):
        self.data = data
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.include_context = include_context
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.include_context:
            source_text = self.task_config['input_format'].format(**item)
        else:
            if 'ltl_to_nl' in self.task_config['name'].lower():
                source_text = f"translate LTL to NL: ltl: {item['formula']}"
            elif 'itl_to_nl' in self.task_config['name'].lower():
                source_text = f"translate ITL to NL: itl: {item['itl_text']}"
            else:
                source_text = self.task_config['input_format'].format(**item)
        
        target_text = self.task_config['output_format'].format(**item)
        
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = source_encoding["input_ids"][0]
        attention_mask = source_encoding["attention_mask"][0]
        labels = target_encoding["input_ids"][0]
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        depth = item.get('depth', 0)
        operators = item.get('operators', 0)
        propositions = item.get('propositions', 0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_text": source_text,
            "target_text": target_text,
            "item_id": item.get('formula_id', idx),
            "depth": depth,
            "operators": operators,
            "propositions": propositions
        }

def get_model_and_tokenizer(model_name, model_path_dir):
    """Download and prepare model and tokenizer."""
    model_config = MODEL_CONFIGS[model_name]
    original_name = model_config['name']
    model_dir = os.path.join(model_path_dir, model_name)
    
    logger.info(f"Loading model {original_name}")
    
    try:
        if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
            logger.info(f"Loading model and tokenizer from local cache: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            if model_config['type'] == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            elif model_config['type'] == 'causal':
                model = AutoModelForCausalLM.from_pretrained(model_dir)
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
        else:
            logger.info(f"Downloading model {original_name} from Hugging Face")
            tokenizer = AutoTokenizer.from_pretrained(original_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set padding token to EOS token for {model_name}")
            
            if model_config['type'] == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(original_name)
            elif model_config['type'] == 'causal':
                model = AutoModelForCausalLM.from_pretrained(original_name)
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
            
            os.makedirs(model_dir, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            logger.info(f"Saved model and tokenizer to {model_dir}")
        
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def setup_distributed_training():
    """Setup multi-GPU training without requiring distributed launch."""
    if n_gpus <= 1:
        logger.info("Single GPU or CPU training")
        return False
    
    logger.info(f"Setting up multi-GPU training on {n_gpus} GPUs")
    torch.cuda.set_device(0)
    
    logger.info(f"Multi-GPU setup complete. Using {n_gpus} GPUs with DataParallel")
    return True

def train_and_evaluate_model(model, tokenizer, train_data, val_data, test_data, task_config, 
                             output_dir, model_name, include_context=True, domains_to_exclude=None):
    """Train and evaluate a model on the given task with multi-GPU support."""
    logger.info(f"Starting training for task: {task_config['name']}")
    logger.info(f"Model: {model_name}, Include context: {include_context}")
    
    if domains_to_exclude:
        logger.info(f"Excluding domains: {domains_to_exclude}")
        train_data = [item for item in train_data if item['domain'] not in domains_to_exclude]
    
    train_dataset = VerifyDataset(
        train_data, task_config, tokenizer, 
        CONFIG['max_source_length'], CONFIG['max_target_length'],
        include_context=include_context
    )
    
    val_dataset = VerifyDataset(
        val_data, task_config, tokenizer, 
        CONFIG['max_source_length'], CONFIG['max_target_length'],
        include_context=include_context
    )
    
    test_dataset = VerifyDataset(
        test_data, task_config, tokenizer, 
        CONFIG['max_source_length'], CONFIG['max_target_length'],
        include_context=include_context
    )
    
    try:
        logger.info(f"Verifying dataset integrity with sample item")
        sample_item = train_dataset[0]
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key not in sample_item or sample_item[key] is None or len(sample_item[key]) == 0:
                raise ValueError(f"Dataset item missing required key: {key}")
        logger.info(f"Dataset verification successful")
    except Exception as e:
        logger.error(f"Dataset verification failed: {str(e)}")
        raise
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if not isinstance(preds, np.ndarray):
            preds = preds.cpu().numpy() if hasattr(preds, 'cpu') else np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)

        preds = np.where((preds >= 0) & (preds < tokenizer.vocab_size), preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = ["[DECODING_INIT_ERROR]"] * len(preds)
        decoded_labels = ["[DECODING_INIT_ERROR]"] * len(labels)
        results = {}
        primary_metric_key_eval = f"eval_{task_config['primary_metrics'][0]}"

        try:
            try:
                decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            except Exception as e_pred:
                logger.error(f"Error during tokenizer.batch_decode(preds): {type(e_pred).__name__} - {e_pred}", exc_info=True)
                decoded_preds = [f"[DECODING_ERROR: {type(e_pred).__name__}]"] * len(preds)

            try:
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            except Exception as e_label:
                logger.error(f"Error during tokenizer.batch_decode(labels): {type(e_label).__name__} - {e_label}", exc_info=True)
                decoded_labels = [f"[LABEL_DECODING_ERROR: {type(e_label).__name__}]"] * len(labels)

            decoded_preds = [str(pred).strip() for pred in decoded_preds]
            decoded_labels = [str(label).strip() for label in decoded_labels]

            if any("[DECODING_ERROR:" in s for s in decoded_preds) or any("[LABEL_DECODING_ERROR:" in s for s in decoded_labels):
                logger.warning("Skipping metric calculation due to decoding errors.")
                results = {primary_metric_key_eval: 0.0}
            else:
                results_calc = calculate_metrics(decoded_preds, decoded_labels, task_config)
                results = {}
                for k, v in results_calc.items():
                    eval_key = f"eval_{k}"
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        results[eval_key] = 0.0
                        if eval_key == primary_metric_key_eval:
                            logger.warning(f"Primary metric '{eval_key}' failed to compute, using 0.0.")
                    else:
                        results[eval_key] = v

                if primary_metric_key_eval not in results:
                    logger.warning(f"Primary metric key '{primary_metric_key_eval}' was missing after calculation, adding as 0.0.")
                    results[primary_metric_key_eval] = 0.0

        except Exception as e:
            logger.error(f"Unexpected error in compute_metrics outer block: {type(e).__name__} - {str(e)}", exc_info=True)
            if primary_metric_key_eval not in results:
                results[primary_metric_key_eval] = 0.0

        if not results:
            logger.warning("Compute_metrics returning empty results dict, ensuring primary metric key exists as 0.0.")
            results[primary_metric_key_eval] = 0.0

        return results

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
        return_tensors="pt"
    )
    
    num_update_steps_per_epoch = max(1, len(train_dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']))
    total_train_steps = CONFIG['num_train_epochs'] * num_update_steps_per_epoch
    warmup_steps = int(total_train_steps * CONFIG['warmup_ratio'])
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['eval_batch_size'],
        weight_decay=CONFIG['weight_decay'],
        num_train_epochs=CONFIG['num_train_epochs'],
        warmup_steps=warmup_steps,
        logging_steps=CONFIG['logging_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        fp16=CONFIG['fp16'],
        bf16=CONFIG['bf16'],
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model=task_config['primary_metrics'][0],
        greater_is_better=True if task_config['primary_metrics'][0] in ['bertscore', 'bleu', 'meteor', 'rouge-l'] else False,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=CONFIG['max_target_length'],
        generation_num_beams=CONFIG['num_beams'],
        dataloader_num_workers=0,
        group_by_length=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_grad_norm=CONFIG['max_grad_norm'],
        report_to="none",
        deepspeed="configs/deepspeed_config.json"
    )
    
    training_args_dict = {
        "output_dir": output_dir,
        "eval_strategy": "steps",
        "learning_rate": CONFIG['learning_rate'],
        "per_device_train_batch_size": CONFIG['batch_size'],
        "per_device_eval_batch_size": CONFIG['eval_batch_size'],
        "weight_decay": CONFIG['weight_decay'],
        "num_train_epochs": CONFIG['num_train_epochs'],
        "warmup_steps": warmup_steps,
        "logging_steps": CONFIG['logging_steps'],
        "eval_steps": CONFIG['eval_steps'],
        "gradient_accumulation_steps": CONFIG['gradient_accumulation_steps'],
        "fp16": CONFIG['fp16'],
        "bf16": CONFIG['bf16'],
        "gradient_checkpointing": True,
        "predict_with_generate": True,
        "generation_max_length": CONFIG['max_target_length'],
        "generation_num_beams": CONFIG['num_beams'],
        "dataloader_num_workers": 0,
        "group_by_length": True,
        "gradient_checkpointing_kwargs": {'use_reentrant': False},
        "max_grad_norm": CONFIG['max_grad_norm'],
        "report_to": "none",
    }

    if DEBUG_MODE:
        logger.warning("DEBUG MODE: Setting TrainingArguments for minimal run.")
        training_args_dict["max_steps"] = 80
        training_args_dict["save_strategy"] = "no"
        training_args_dict["load_best_model_at_end"] = False
        training_args_dict["eval_steps"] = min(CONFIG['eval_steps'], training_args_dict["max_steps"] // 2)
    else:
        training_args_dict["save_strategy"] = "steps"
        training_args_dict["save_steps"] = CONFIG['save_steps']
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = f"eval_{task_config['primary_metrics'][0]}"
        training_args_dict["greater_is_better"] = True if task_config['primary_metrics'][0] in ['bertscore', 'bleu', 'meteor', 'rouge-l', 'semantic_equivalence', 'syntactic_correctness', 'exact_match', 'normalized_exact_match'] else False # Added logic metrics
        training_args_dict["save_total_limit"] = 3
        training_args_dict["report_to"] = "none"

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    checkpoint_dir_base = CONFIG['checkpoint_dir']
    task_name_sanitized = task_config['name'].replace(' ', '_').replace('/', '_')
    model_name_sanitized = model_name.replace('/', '__')
    checkpoint_path = os.path.join(checkpoint_dir_base, f"{task_name_sanitized}_{model_name_sanitized}")

    resume_from_checkpoint = None
    can_resume = (
        CONFIG.get('checkpointing', False) and
        os.path.isdir(checkpoint_path) and
        os.path.exists(os.path.join(checkpoint_path, "trainer_state.json"))
    )

    if can_resume:
        logger.info(f"Attempting to resume training from checkpoint: {checkpoint_path}")
        resume_from_checkpoint = checkpoint_path
    elif CONFIG.get('checkpointing', False) and os.path.exists(checkpoint_path):
         logger.warning(f"Checkpoint directory found but seems incomplete or incompatible ({checkpoint_path}). Will start fresh training.")
         resume_from_checkpoint = None
    else:
         logger.info("No valid checkpoint found or checkpointing disabled. Starting training from scratch.")
         resume_from_checkpoint = None

    try:
        logger.info(f"Starting model training (resuming from: {resume_from_checkpoint})...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except ValueError as e:
        if "You should supply an encoding" in str(e):
            logger.error(f"Data encoding error during training: {str(e)}")
            logger.error("This often indicates an issue with dataset tokenization or collation. Please check data preprocessing steps.")
            raise
        elif "out of memory" in str(e).lower():
            logger.error(f"OOM ValueError during training: {str(e)}")
            logger.error("Try reducing batch size ('batch_size'), sequence length ('max_source_length'/'max_target_length'), enabling gradient checkpointing, or using a smaller model / DDP / DeepSpeed.")
            raise
        else:
            logger.error(f"Unhandled ValueError during training: {str(e)}", exc_info=True)
            raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM RuntimeError during training: {str(e)}")
            logger.error("Try reducing batch size ('batch_size'), sequence length ('max_source_length'/'max_target_length'), enabling gradient checkpointing, or using a smaller model / DDP / DeepSpeed.")
            raise
        else:
            logger.error(f"Unhandled RuntimeError during training: {str(e)}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"Generic error during training: {type(e).__name__} - {str(e)}", exc_info=True)
        raise

    logger.info("Training completed or training loop exited.")
    eval_model_instance = None
    if DEBUG_MODE:
        logger.warning("DEBUG MODE: Evaluating final model state directly (best model not loaded).")
        current_model = trainer.model
        if hasattr(current_model, "module"):
                eval_model_instance = current_model.module
        else:
                eval_model_instance = current_model
        if eval_model_instance:
            eval_model_instance.to(device)
    else:
        best_model_output_dir = os.path.join(output_dir, "best_model")
        try:
            if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint:
                logger.info(f"Trainer identified best checkpoint: {trainer.state.best_model_checkpoint}")
                trainer.save_model(best_model_output_dir)
                logger.info(f"Best model saved to {best_model_output_dir}")
            elif os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
                logger.warning("No best model checkpoint recorded by trainer (or load_best_model_at_end=False). Saving final model state as 'best_model'.")
                trainer.save_model(best_model_output_dir)
                logger.info(f"Final model state saved to {best_model_output_dir}")
            else:
                logger.error(f"Training seemed to finish, but no model found to save in {output_dir} and no best checkpoint recorded.")

        except Exception as e:
            logger.error(f"Error saving best/final model: {str(e)}", exc_info=True)

        logger.info("Evaluating on test set using the saved best model...")
        if 'device' not in locals():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if os.path.exists(best_model_output_dir):
            try:
                logger.info(f"Loading best model for final evaluation from: {best_model_output_dir}")
                original_model_class = type(trainer.model.module if hasattr(trainer.model, "module") else trainer.model)
                eval_model_instance = original_model_class.from_pretrained(best_model_output_dir)
                eval_model_instance.to(device)
                logger.info("Best model loaded successfully for evaluation.")
            except Exception as e:
                logger.error(f"Error loading best model from {best_model_output_dir}: {str(e)}. Cannot proceed with evaluation.", exc_info=True)
                return {"error": f"Failed to load best model from {best_model_output_dir}", "metrics": {}}
        else:
            logger.error(f"Best model directory {best_model_output_dir} not found. Cannot perform final evaluation.")
            return {"error": f"Best model directory not found at {best_model_output_dir}", "metrics": {}}

    test_results = {"error": "No model instance available for evaluation", "metrics": {}}
    if eval_model_instance is not None:
        logger.info("Running generation and evaluation on the test set...")
        try:
            test_results = generate_and_evaluate(
                eval_model_instance, tokenizer, test_dataset,
                task_config, CONFIG['num_beams'], CONFIG['max_target_length']
            )
        except Exception as e:
            logger.error(f"Error during final generation and evaluation: {str(e)}", exc_info=True)
            test_results = {"error": f"Error during final evaluation: {str(e)}", "metrics": {}}
    else:
        logger.error("Could not obtain a model instance for final evaluation.")

    if 'metrics' in test_results and test_results.get('metrics'):
        logger.info(f"Test set metrics: {test_results['metrics']}")
    elif 'error' in test_results:
            logger.error(f"Evaluation failed or skipped: {test_results['error']}")
    else:
            logger.warning("Evaluation finished but produced no metrics or errors.")

    return test_results

def generate_and_evaluate(model, tokenizer, dataset, task_config, num_beams, max_length):
    """Generate outputs for a dataset and evaluate them."""
    logger.info(f"Generating predictions for task: {task_config['name']}")
    
    model.eval()
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = []
    references = []
    source_texts = []
    item_ids = []
    complexity_info = []
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['eval_batch_size'], 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            batch_size = input_ids.size(0)
            current_beam_size = num_beams
            
            while True:
                try:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=current_beam_size,
                        early_stopping=True,
                    )
                    
                    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and (batch_size > 1 or current_beam_size > 1):
                        if batch_size > 1:
                            batch_size = batch_size // 2
                            logger.warning(f"OOM error, reducing batch size to {batch_size}")
                            input_ids = input_ids[:batch_size]
                            attention_mask = attention_mask[:batch_size]
                        elif current_beam_size > 1:
                            current_beam_size = max(1, current_beam_size // 2)
                            logger.warning(f"OOM error at batch size 1, reducing beam size to {current_beam_size}")
                        else:
                            logger.error(f"OOM error at minimum batch and beam size, cannot generate")
                            raise
                    else:
                        logger.error(f"Error during generation: {str(e)}")
                        raise
            predictions.extend(decoded_preds)
            
            batch_references = batch["target_text"][:len(decoded_preds)]
            batch_source_texts = batch["source_text"][:len(decoded_preds)]
            batch_item_ids = batch["item_id"][:len(decoded_preds)]
            
            for i in range(len(decoded_preds)):
                complexity_info.append({
                    "depth": batch["depth"][i].item() if "depth" in batch else 0,
                    "operators": batch["operators"][i].item() if "operators" in batch else 0,
                    "propositions": batch["propositions"][i].item() if "propositions" in batch else 0
                })
            
            references.extend(batch_references)
            source_texts.extend(batch_source_texts)
            item_ids.extend(batch_item_ids.tolist())
            
            if batch_size < len(batch["input_ids"]):
                remaining = len(batch["input_ids"]) - batch_size
                logger.info(f"Processing remaining {remaining} examples one by one")
                
                for i in range(batch_size, len(batch["input_ids"])):
                    try:
                        single_input_ids = batch["input_ids"][i:i+1].to(device)
                        single_attention_mask = batch["attention_mask"][i:i+1].to(device)
                        
                        single_output = model.generate(
                            input_ids=single_input_ids,
                            attention_mask=single_attention_mask,
                            max_length=max_length,
                            num_beams=1,
                            early_stopping=True,
                        )
                        
                        single_pred = tokenizer.decode(single_output[0], skip_special_tokens=True)
                        predictions.append(single_pred)
                        
                        references.append(batch["target_text"][i])
                        source_texts.append(batch["source_text"][i])
                        item_ids.append(batch["item_id"][i].item())
                        
                        complexity_info.append({
                            "depth": batch["depth"][i].item() if "depth" in batch else 0,
                            "operators": batch["operators"][i].item() if "operators" in batch else 0,
                            "propositions": batch["propositions"][i].item() if "propositions" in batch else 0
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing individual example {i}: {str(e)}")
                        predictions.append("[Generation failed]")
                        references.append(batch["target_text"][i])
                        source_texts.append(batch["source_text"][i])
                        item_ids.append(batch["item_id"][i].item())
                        
                        complexity_info.append({
                            "depth": batch["depth"][i].item() if "depth" in batch else 0,
                            "operators": batch["operators"][i].item() if "operators" in batch else 0,
                            "propositions": batch["propositions"][i].item() if "propositions" in batch else 0
                        })
    
    metrics = calculate_metrics(predictions, references, task_config)
    
    prediction_data = {
        "predictions": predictions,
        "references": references,
        "source_texts": source_texts,
        "item_ids": item_ids,
        "complexity_info": complexity_info,
        "metrics": metrics
    }
    
    return prediction_data

def _spot_ast_to_zss_node(spot_formula_node):
    if not spot or not zss:
        raise ImportError("spot or zss library not available for AST conversion.")

    children = []
    if spot_formula_node.is_operator():
        for child in spot_formula_node:
             children.append(_spot_ast_to_zss_node(child))

    node_label = str(spot_formula_node.kind())
    if spot_formula_node.is_literal():
        prop_name = str(spot_formula_node)
        node_label = prop_name
    elif spot_formula_node.is_value():
         node_label = str(spot_formula_node)

    return zss.Node(node_label, children)

def _get_node_count(zss_node):
     count = 1
     for child in zss_node.children:
         count += _get_node_count(child)
     return count

def calculate_metrics(predictions, references, task_config):
    logger.info(f"Calculating metrics for task: {task_config['name']}")
    metrics_results = {}

    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references)
                   if isinstance(pred, str) and pred != "[Generation failed]"]

    if not valid_pairs:
        logger.warning("No valid prediction pairs found for metric calculation")
        for metric_name in task_config.get('metrics', []):
             metrics_results[metric_name] = 0.0 if metric_name not in ['semantic_equivalence', 'syntactic_correctness', 'tree_edit_distance'] else None
        return metrics_results

    valid_preds, valid_refs = zip(*valid_pairs)
    valid_preds = list(valid_preds)
    valid_refs = list(valid_refs)

    task_name_lower = task_config.get('name', '').lower()
    is_ltl_task = 'ltl' in task_name_lower
    is_itl_task = 'itl' in task_name_lower
    is_nl_output_task = 'nl' in task_name_lower
    for metric_name in task_config.get('metrics', []):
        metric_result = None
        try:
            if metric_name == 'bertscore':
                logger.info(f"Processing metric: {metric_name}")
                metric_result = None
                if not is_nl_output_task:
                    logger.info(f"Metric '{metric_name}' skipped: Not an NL output task.")
                elif valid_preds and valid_refs:
                    logger.info(f"Attempting '{metric_name}' calculation on {len(valid_preds)} valid pairs.")
                    try:
                        sample_idx = min(2, len(valid_preds))
                        logger.info(f"Logging first {sample_idx} pair(s) for '{metric_name}':")
                        for i in range(sample_idx):
                             pred_sample = valid_preds[i][:150] + ('...' if len(valid_preds[i]) > 150 else '')
                             ref_sample = valid_refs[i][:150] + ('...' if len(valid_refs[i]) > 150 else '')
                             logger.info(f"  Sample {i+1} Pred (len {len(valid_preds[i])}): '{pred_sample}'")
                             logger.info(f"  Sample {i+1} Ref (len {len(valid_refs[i])}): '{ref_sample}'")

                        logger.info(f"Loading metric '{metric_name}'...")
                        bertscore_eval = evaluate.load('bertscore')
                        logger.info(f"Metric '{metric_name}' loaded successfully.")
                        logger.info(f"Calling {metric_name}.compute...")
                        results = bertscore_eval.compute(predictions=valid_preds, references=valid_refs, lang='en')
                        logger.info(f"{metric_name}.compute finished.")

                        if results and 'f1' in results and results['f1'] is not None:
                            f1_scores = results['f1']
                            if isinstance(f1_scores, list) and len(f1_scores) > 0:
                                metric_result = np.mean(f1_scores)
                                logger.info(f"Metric '{metric_name}' calculated successfully (average F1): {metric_result}")
                            elif isinstance(f1_scores, (float, np.number)):
                                metric_result = float(f1_scores)
                                logger.info(f"Metric '{metric_name}' calculated successfully (single F1): {metric_result}")
                            else:
                                 logger.warning(f"Metric '{metric_name}' F1 score is not a valid list or number: {f1_scores}")
                                 metric_result = None
                        else:
                            logger.warning(f"Metric '{metric_name}' compute results missing 'f1' key, key value is None, or results empty. Results: {results}")
                            metric_result = None

                    except Exception as e:
                        logger.error(f"Error computing metric '{metric_name}': {type(e).__name__} - {e}", exc_info=True)
                        metric_result = None
                else:
                    logger.info(f"Metric '{metric_name}' set to 0.0: No valid predictions/references provided.")
                    metric_result = 0.0
            elif metric_name == 'rouge-l':
                 if not is_nl_output_task:
                     metric_result = None
                 elif valid_preds and valid_refs:
                    rouge_eval = evaluate.load('rouge')
                    results = rouge_eval.compute(predictions=valid_preds, references=valid_refs, use_aggregator=True)
                    metric_result = results['rougeL']
                 else:
                     metric_result = 0.0
            elif metric_name == 'bleu':
                if not is_nl_output_task:
                     metric_result = None
                elif valid_preds and valid_refs:
                    bleu_eval = evaluate.load('bleu')
                    list_of_list_refs = [[ref] for ref in valid_refs]
                    results = bleu_eval.compute(predictions=valid_preds, references=list_of_list_refs)
                    metric_result = results['bleu']
                else:
                    metric_result = 0.0
            elif metric_name == 'meteor':
                 if not is_nl_output_task:
                     metric_result = None
                 elif nltk is None:
                     logger.warning("NLTK not available, skipping METEOR")
                     metric_result = None
                 elif valid_preds and valid_refs:
                    meteor_eval = evaluate.load('meteor')
                    results = meteor_eval.compute(predictions=valid_preds, references=valid_refs)
                    metric_result = results['meteor']
                 else:
                     metric_result = 0.0
            elif metric_name == 'exact_match':
                if valid_preds:
                    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(valid_preds, valid_refs)]
                    metric_result = sum(exact_matches) / len(exact_matches)
                else:
                    metric_result = 0.0
            elif metric_name == 'normalized_exact_match':
                if valid_preds:
                    normalized_preds = [' '.join(pred.strip().lower().split()) for pred in valid_preds]
                    normalized_refs = [' '.join(ref.strip().lower().split()) for ref in valid_refs]
                    norm_exact_matches = [pred == ref for pred, ref in zip(normalized_preds, normalized_refs)]
                    metric_result = sum(norm_exact_matches) / len(norm_exact_matches)
                else:
                     metric_result = 0.0

            elif metric_name == 'semantic_equivalence':
                if not is_ltl_task:
                    metric_result = None
                elif not spot:
                    logger.warning("Spot library not available. Skipping semantic equivalence check.")
                    metric_result = None
                else:
                    equivalence_count = 0
                    valid_comparison_count = 0
                    for pred, ref in zip(valid_preds, valid_refs):
                        try:
                            pred_formula = spot.formula(pred.strip())
                            ref_formula = spot.formula(ref.strip())
                            if pred_formula.equivalent_to(ref_formula):
                                equivalence_count += 1
                            valid_comparison_count += 1
                        except Exception:
                            continue
                    if valid_comparison_count > 0:
                        metric_result = equivalence_count / valid_comparison_count
                    else:
                         metric_result = 0.0 if valid_preds else None

            elif metric_name == 'tree_edit_distance':
                 if is_ltl_task:
                     distances = []
                     valid_comparison_count = 0
                     for pred, ref in zip(valid_preds, valid_refs):
                         try:
                             pred_f = spot.formula(pred.strip())
                             ref_f = spot.formula(ref.strip())

                             pred_ast = _spot_ast_to_zss_node(pred_f)
                             ref_ast = _spot_ast_to_zss_node(ref_f)

                             dist = zss.simple_distance(pred_ast, ref_ast)

                             pred_nodes = _get_node_count(pred_ast)
                             ref_nodes = _get_node_count(ref_ast)
                             max_nodes = max(pred_nodes, ref_nodes)

                             normalized_dist = dist / max_nodes if max_nodes > 0 else 0
                             distances.append(normalized_dist)
                             valid_comparison_count += 1
                         except ImportError:
                             raise
                         except Exception:
                             continue

                     if valid_comparison_count > 0:
                         metric_result = np.mean(distances)
                     else:
                         metric_result = None
                 elif is_itl_task:
                     logger.warning("Tree Edit Distance not implemented for ITL (requires ITL parser).")
                     metric_result = None
                 elif is_nl_output_task:
                      metric_result = None
                 else:
                     if not spot: logger.warning("Spot library not available for LTL Tree Edit Distance.")
                     if not zss: logger.warning("ZSS library not available for LTL Tree Edit Distance.")
                     metric_result = None

            elif metric_name == 'syntactic_correctness':
                 if is_ltl_task and spot:
                     correct_count = 0
                     total_count = len(valid_preds)
                     if total_count > 0:
                         for pred in valid_preds:
                             try:
                                 spot.formula(pred.strip())
                                 correct_count += 1
                             except Exception:
                                 continue
                         metric_result = correct_count / total_count
                     else:
                         metric_result = 0.0
                 elif is_ltl_task and not spot:
                     logger.warning("Spot library not available. Skipping LTL syntactic correctness.")
                     metric_result = None
                 elif is_itl_task:
                     logger.warning("Syntactic correctness check not implemented for ITL.")
                     metric_result = None
                 else:
                     metric_result = None

        except Exception as e:
             logger.error(f"Unexpected error calculating metric '{metric_name}': {type(e).__name__} - {str(e)}", exc_info=True)
             metric_result = None

        metrics_results[metric_name] = None if metric_result is None or (isinstance(metric_result, float) and math.isnan(metric_result)) else metric_result


    logger.info(f"Metrics calculated: {metrics_results}")
    return metrics_results

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def create_human_evaluation_data(predictions, references, source_texts, task_name, num_samples=100):
    """Create data for human evaluation of Tasks 1 and 2."""
    logger.info(f"Creating human evaluation data for task: {task_name}")
    
    valid_data = [(pred, ref, src) for pred, ref, src in zip(predictions, references, source_texts) 
                 if pred != "[Generation failed]"]
    
    if not valid_data:
        logger.warning("No valid predictions for human evaluation")
        return {"error": "No valid predictions for human evaluation"}
    
    valid_preds, valid_refs, valid_srcs = zip(*valid_data)
    
    if len(valid_preds) < num_samples:
        num_samples = len(valid_preds)
        logger.warning(f"Reduced human evaluation samples to {num_samples} due to available data")
    
    indices = random.sample(range(len(valid_preds)), num_samples)
    
    human_eval_data = {
        "task": task_name,
        "evaluation_instructions": """
        Please evaluate each generated translation on three dimensions using a scale of 1-5:
        
        1. Fluency (1-5): How natural and fluent is the language?
           1: Incomprehensible or completely unnatural
           2: Difficult to understand with major grammatical errors
           3: Understandable but with noticeable errors or awkwardness
           4: Mostly natural with minor issues
           5: Perfectly fluent, natural language
        
        2. Adequacy (1-5): How well does the translation capture the meaning of the original formula?
           1: Completely inaccurate, misses core meaning
           2: Major errors in meaning
           3: Partial capture of meaning with some errors
           4: Mostly accurate with minor omissions
           5: Perfect representation of the formula's meaning
        
        3. Context Relevance (1-5): How well does the translation utilize the domain and activity context?
           1: Ignores context completely
           2: Minimal use of context
           3: Includes context but not fully integrated
           4: Good integration of context
           5: Perfect integration of domain and activity context
        
        For each sample, assign a score from 1-5 for each of the three dimensions.
        Record your scores for each sample in the 'human_scores' field.
        """,
        "samples": []
    }
    
    for idx in indices:
        human_eval_data["samples"].append({
            "source": valid_srcs[idx],
            "reference": valid_refs[idx],
            "prediction": valid_preds[idx],
            "human_scores": {
                "fluency": None,
                "adequacy": None,
                "context_relevance": None
            }
        })
    
    return human_eval_data

def run_task(task_name, model_name, train_data, val_data, test_data, include_context=True, domains_to_exclude=None):
    """Run a single task with the specified model."""
    logger.info(f"Running task {task_name} with model {model_name}")
    
    task_config = TASK_CONFIGS[task_name]
    logger.info(f"Task config: {task_config}")
    output_dir = os.path.join(CONFIG['results_dir'], 'tasks', f"{task_name}_{model_name}")
    
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, CONFIG['model_dir'])
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        return {
            "error": f"Model loading failed: {str(e)}",
            "metrics": {},
            "predictions": [],
            "references": [],
            "source_texts": [],
            "item_ids": []
        }

    logging.info(f"{model_name}")
    
    results = train_and_evaluate_model(
        model, tokenizer, train_data, val_data, test_data,
        task_config, output_dir, model_name, include_context, domains_to_exclude
    )
    
    if task_name in ['ltl_to_nl', 'itl_to_nl'] and len(results.get('predictions', [])) > 0:
        human_eval_data = create_human_evaluation_data(
            results['predictions'], results['references'], results['source_texts'],
            task_config['name'], CONFIG['human_eval_samples']
        )
        
        human_eval_path = os.path.join(CONFIG['results_dir'], 'human_eval', f"{task_name}_{model_name}_human_eval.json")
        with open(human_eval_path, 'w') as f:
            json.dump(human_eval_data, f, indent=2)
        logger.info(f"Human evaluation data saved to {human_eval_path}")
    
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, 'w') as f:
        serializable_results = {
            "predictions": results.get('predictions', []),
            "references": results.get('references', []),
            "source_texts": results.get('source_texts', []),
            "item_ids": [int(id) for id in results.get('item_ids', [])],
            "complexity_info": results.get('complexity_info', []),
            "metrics": results.get('metrics', {})
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results

def run_experiment_value_of_itl(train_data, val_data, test_data, model_name="t5-base"):
    """Run Experiment A: The Value of ITL."""
    logger.info("Running Experiment A: The Value of ITL")
    
    # Direct LTL -> NL
    direct_results = run_task('ltl_to_nl', model_name, train_data, val_data, test_data)
    
    # Two-stage approach
    # First stage: LTL -> ITL
    ltl_to_itl_results = run_task('ltl_to_itl', model_name, train_data, val_data, test_data)
    
    # For two-stage inference, we need to generate ITL for all test examples
    test_ltl_to_itl_predictions = ltl_to_itl_results.get('predictions', [])
    
    # Create new test data with generated ITL
    two_stage_test_data = []
    for i, item in enumerate(test_data):
        if i < len(test_ltl_to_itl_predictions):
            new_item = item.copy()
            new_item['itl_text'] = test_ltl_to_itl_predictions[i]
            two_stage_test_data.append(new_item)
    
    # Second stage: ITL -> NL
    itl_to_nl_results = run_task('itl_to_nl', model_name, train_data, val_data, two_stage_test_data)
    
    # Compare results
    comparison = {
        "experiment": "Value of ITL",
        "model": model_name,
        "direct_approach": direct_results.get('metrics', {}),
        "two_stage_approach": itl_to_nl_results.get('metrics', {}),
        "complexity_analysis": {}
    }
    
    if (len(direct_results.get('predictions', [])) > 0 and 
        len(itl_to_nl_results.get('predictions', [])) > 0):
        
        depth_bins = defaultdict(list)
        operator_bins = defaultdict(list)
        proposition_bins = defaultdict(list)
        
        min_length = min(len(direct_results['predictions']), len(itl_to_nl_results['predictions']))
        
        for i in range(min_length):
            if i < len(direct_results.get('complexity_info', [])) and i < len(itl_to_nl_results.get('complexity_info', [])):
                direct_info = direct_results['complexity_info'][i]
                
                depth_bin = f"depth_{direct_info['depth']}"
                depth_bins[depth_bin].append({
                    "direct_prediction": direct_results['predictions'][i],
                    "two_stage_prediction": itl_to_nl_results['predictions'][i],
                    "reference": direct_results['references'][i]
                })
                
                op_bin = f"operators_{5*(direct_info['operators']//5)}-{5*(direct_info['operators']//5)+4}"
                operator_bins[op_bin].append({
                    "direct_prediction": direct_results['predictions'][i],
                    "two_stage_prediction": itl_to_nl_results['predictions'][i],
                    "reference": direct_results['references'][i]
                })
                
                prop_bin = f"propositions_{direct_info['propositions']}"
                proposition_bins[prop_bin].append({
                    "direct_prediction": direct_results['predictions'][i],
                    "two_stage_prediction": itl_to_nl_results['predictions'][i],
                    "reference": direct_results['references'][i]
                })
        
        comparison["complexity_analysis"]["by_depth"] = {}
        for bin_key, bin_items in depth_bins.items():
            if len(bin_items) >= 5:
                direct_preds = [item['direct_prediction'] for item in bin_items]
                two_stage_preds = [item['two_stage_prediction'] for item in bin_items]
                refs = [item['reference'] for item in bin_items]
                
                bertscore = evaluate.load('bertscore')
                direct_results_score = bertscore.compute(predictions=direct_preds, references=refs, lang='en')
                two_stage_results_score = bertscore.compute(predictions=two_stage_preds, references=refs, lang='en')
                
                comparison["complexity_analysis"]["by_depth"][bin_key] = {
                    "count": len(bin_items),
                    "direct_bertscore": np.mean(direct_results_score['f1']),
                    "two_stage_bertscore": np.mean(two_stage_results_score['f1'])
                }
        
        comparison["complexity_analysis"]["by_operators"] = {}
        for bin_key, bin_items in operator_bins.items():
            if len(bin_items) >= 5:
                direct_preds = [item['direct_prediction'] for item in bin_items]
                two_stage_preds = [item['two_stage_prediction'] for item in bin_items]
                refs = [item['reference'] for item in bin_items]
                
                bertscore = evaluate.load('bertscore')
                direct_results_score = bertscore.compute(predictions=direct_preds, references=refs, lang='en')
                two_stage_results_score = bertscore.compute(predictions=two_stage_preds, references=refs, lang='en')
                
                comparison["complexity_analysis"]["by_operators"][bin_key] = {
                    "count": len(bin_items),
                    "direct_bertscore": np.mean(direct_results_score['f1']),
                    "two_stage_bertscore": np.mean(two_stage_results_score['f1'])
                }
        
        comparison["complexity_analysis"]["by_propositions"] = {}
        for bin_key, bin_items in proposition_bins.items():
            if len(bin_items) >= 5:
                direct_preds = [item['direct_prediction'] for item in bin_items]
                two_stage_preds = [item['two_stage_prediction'] for item in bin_items]
                refs = [item['reference'] for item in bin_items]
                
                bertscore = evaluate.load('bertscore')
                direct_results_score = bertscore.compute(predictions=direct_preds, references=refs, lang='en')
                two_stage_results_score = bertscore.compute(predictions=two_stage_preds, references=refs, lang='en')
                
                comparison["complexity_analysis"]["by_propositions"][bin_key] = {
                    "count": len(bin_items),
                    "direct_bertscore": np.mean(direct_results_score['f1']),
                    "two_stage_bertscore": np.mean(two_stage_results_score['f1'])
                }
    
    comparison_path = os.path.join(CONFIG['results_dir'], 'experiments', 'value_of_itl.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Value of ITL experiment results saved to {comparison_path}")
    return comparison

def run_experiment_impact_of_context(train_data, val_data, test_data, model_name="t5-base"):
    """Run Experiment B: The Impact of Context."""
    logger.info("Running Experiment B: The Impact of Context")
    
    with_context_ltl_results = run_task('ltl_to_nl', model_name, train_data, val_data, test_data, include_context=True)
    with_context_itl_results = run_task('itl_to_nl', model_name, train_data, val_data, test_data, include_context=True)
    
    without_context_ltl_results = run_task('ltl_to_nl', model_name, train_data, val_data, test_data, include_context=False)
    without_context_itl_results = run_task('itl_to_nl', model_name, train_data, val_data, test_data, include_context=False)
    
    comparison = {
        "experiment": "Impact of Context",
        "model": model_name,
        "ltl_to_nl": {
            "with_context": with_context_ltl_results.get('metrics', {}),
            "without_context": without_context_ltl_results.get('metrics', {}),
            "difference": {
                metric: with_context_ltl_results.get('metrics', {}).get(metric, 0) - 
                        without_context_ltl_results.get('metrics', {}).get(metric, 0)
                for metric in with_context_ltl_results.get('metrics', {})
                if (with_context_ltl_results.get('metrics', {}).get(metric) is not None and 
                    without_context_ltl_results.get('metrics', {}).get(metric) is not None)
            }
        },
        "itl_to_nl": {
            "with_context": with_context_itl_results.get('metrics', {}),
            "without_context": without_context_itl_results.get('metrics', {}),
            "difference": {
                metric: with_context_itl_results.get('metrics', {}).get(metric, 0) - 
                        without_context_itl_results.get('metrics', {}).get(metric, 0)
                for metric in with_context_itl_results.get('metrics', {})
                if (with_context_itl_results.get('metrics', {}).get(metric) is not None and 
                    without_context_itl_results.get('metrics', {}).get(metric) is not None)
            }
        }
    }
    
    if (len(with_context_ltl_results.get('predictions', [])) > 0 and 
        len(without_context_ltl_results.get('predictions', [])) > 0):
        
        domain_analysis = defaultdict(lambda: {"with_context": [], "without_context": []})
        
        for i, item in enumerate(test_data):
            if i < len(with_context_ltl_results['predictions']) and i < len(without_context_ltl_results['predictions']):
                domain = item['domain']
                
                domain_analysis[domain]["with_context"].append({
                    "prediction": with_context_ltl_results['predictions'][i],
                    "reference": with_context_ltl_results['references'][i]
                })
                
                domain_analysis[domain]["without_context"].append({
                    "prediction": without_context_ltl_results['predictions'][i],
                    "reference": without_context_ltl_results['references'][i]
                })
        
        domain_results = {}
        for domain, data in domain_analysis.items():
            if len(data["with_context"]) >= 10:
                with_context_preds = [item["prediction"] for item in data["with_context"]]
                with_context_refs = [item["reference"] for item in data["with_context"]]
                
                without_context_preds = [item["prediction"] for item in data["without_context"]]
                without_context_refs = [item["reference"] for item in data["without_context"]]
                
                bertscore = evaluate.load('bertscore')
                with_context_score = bertscore.compute(predictions=with_context_preds, references=with_context_refs, lang='en')
                without_context_score = bertscore.compute(predictions=without_context_preds, references=without_context_refs, lang='en')
                
                domain_results[domain] = {
                    "count": len(data["with_context"]),
                    "with_context_bertscore": np.mean(with_context_score['f1']),
                    "without_context_bertscore": np.mean(without_context_score['f1']),
                    "difference": np.mean(with_context_score['f1']) - np.mean(without_context_score['f1'])
                }
        
        comparison["domain_analysis"] = domain_results
    
    comparison_path = os.path.join(CONFIG['results_dir'], 'experiments', 'impact_of_context.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Impact of Context experiment results saved to {comparison_path}")
    return comparison

def run_experiment_domain_generalization(train_data, val_data, test_data, domains, model_name="t5-base"):
    """Run Experiment C: Domain Generalization & Adaptation."""
    logger.info("Running Experiment C: Domain Generalization & Adaptation")
    
    logger.info("Running Leave-One-Domain-Out experiments")
    
    lodo_results = {}
    for domain in domains:
        logger.info(f"Running LODO experiment for domain: {domain}")
        
        domain_test_data = [item for item in test_data if item['domain'] == domain]
        
        if not domain_test_data:
            logger.warning(f"No test data for domain {domain}, skipping")
            continue
        
        # Run LTL->NL task excluding the current domain
        ltl_to_nl_results = run_task(
            'ltl_to_nl', model_name, train_data, val_data, domain_test_data,
            domains_to_exclude=[domain]
        )
        
        # Run NL->LTL task excluding the current domain
        nl_to_ltl_results = run_task(
            'nl_to_ltl', model_name, train_data, val_data, domain_test_data,
            domains_to_exclude=[domain]
        )
        
        lodo_results[domain] = {
            "ltl_to_nl": ltl_to_nl_results.get('metrics', {}),
            "nl_to_ltl": nl_to_ltl_results.get('metrics', {}),
            "test_samples": len(domain_test_data)
        }
    
    logger.info("Running Few-Shot Adaptation experiments")
    
    if len(domains) >= 3:
        target_domains = domains[:3]  # Take first 3 domains for consistency
    else:
        target_domains = domains
    
    few_shot_results = {}
    for domain in target_domains:
        logger.info(f"Running few-shot adaptation for domain: {domain}")
        
        domain_train_data = [item for item in train_data if item['domain'] == domain]
        domain_test_data = [item for item in test_data if item['domain'] == domain]
        
        if len(domain_train_data) < max(EXPERIMENT_CONFIGS['domain_generalization']['few_shot_k']) or not domain_test_data:
            logger.warning(f"Insufficient data for domain {domain}, skipping")
            continue
        
        other_domains_train_data = [item for item in train_data if item['domain'] != domain]
        
        # Train base model for LTL->NL
        base_ltl_to_nl_results = run_task(
            'ltl_to_nl', model_name, other_domains_train_data, val_data, domain_test_data,
            domains_to_exclude=[domain]
        )
        
        # Few-shot adaptation for different k values
        domain_few_shot_results = {"k_values": {}}
        
        for k in EXPERIMENT_CONFIGS['domain_generalization']['few_shot_k']:
            if k > len(domain_train_data):
                logger.warning(f"Insufficient data for k={k}, skipping")
                continue
                
            # Sample k examples from domain training data
            few_shot_examples = random.sample(domain_train_data, k)
            
            # Fine-tune on few-shot examples
            few_shot_ltl_to_nl_results = run_task(
                'ltl_to_nl', model_name, few_shot_examples, val_data, domain_test_data
            )
            
            domain_few_shot_results["k_values"][k] = few_shot_ltl_to_nl_results.get('metrics', {})
        
        domain_few_shot_results["base_model"] = base_ltl_to_nl_results.get('metrics', {})
        few_shot_results[domain] = domain_few_shot_results
    
    experiment_results = {
        "experiment": "Domain Generalization & Adaptation",
        "model": model_name,
        "lodo_results": lodo_results,
        "few_shot_results": few_shot_results
    }
    
    results_path = os.path.join(CONFIG['results_dir'], 'experiments', 'domain_generalization.json')
    with open(results_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    logger.info(f"Domain Generalization experiment results saved to {results_path}")
    return experiment_results

def run_experiment_sensitivity_to_complexity(train_data, val_data, test_data, model_name="t5-base"):
    """Run Experiment D: Sensitivity to Logical Complexity."""
    logger.info("Running Experiment D: Sensitivity to Logical Complexity")
    
    ltl_to_nl_results = run_task('ltl_to_nl', model_name, train_data, val_data, test_data)
    nl_to_ltl_results = run_task('nl_to_ltl', model_name, train_data, val_data, test_data)
    
    complexity_bins = {
        "depth": defaultdict(lambda: {"ltl_to_nl": [], "nl_to_ltl": []}),
        "operators": defaultdict(lambda: {"ltl_to_nl": [], "nl_to_ltl": []}),
        "propositions": defaultdict(lambda: {"ltl_to_nl": [], "nl_to_ltl": []})
    }
    
    if (len(ltl_to_nl_results.get('predictions', [])) > 0 and 
        len(nl_to_ltl_results.get('predictions', [])) > 0):
        
        for i, item in enumerate(test_data):
            if (i < len(ltl_to_nl_results.get('predictions', [])) and 
                i < len(nl_to_ltl_results.get('predictions', [])) and
                i < len(ltl_to_nl_results.get('complexity_info', [])) and
                i < len(nl_to_ltl_results.get('complexity_info', []))):
                
                depth = ltl_to_nl_results['complexity_info'][i]['depth']
                operators = ltl_to_nl_results['complexity_info'][i]['operators']
                propositions = ltl_to_nl_results['complexity_info'][i]['propositions']
                
                depth_bin = f"depth_{depth}"
                op_bin = f"operators_{5*(operators//5)}-{5*(operators//5)+4}"
                prop_bin = f"propositions_{propositions}"
                
                complexity_bins["depth"][depth_bin]["ltl_to_nl"].append({
                    "prediction": ltl_to_nl_results['predictions'][i],
                    "reference": ltl_to_nl_results['references'][i]
                })
                
                complexity_bins["depth"][depth_bin]["nl_to_ltl"].append({
                    "prediction": nl_to_ltl_results['predictions'][i],
                    "reference": nl_to_ltl_results['references'][i]
                })
                
                complexity_bins["operators"][op_bin]["ltl_to_nl"].append({
                    "prediction": ltl_to_nl_results['predictions'][i],
                    "reference": ltl_to_nl_results['references'][i]
                })
                
                complexity_bins["operators"][op_bin]["nl_to_ltl"].append({
                    "prediction": nl_to_ltl_results['predictions'][i],
                    "reference": nl_to_ltl_results['references'][i]
                })
                
                complexity_bins["propositions"][prop_bin]["ltl_to_nl"].append({
                    "prediction": ltl_to_nl_results['predictions'][i],
                    "reference": ltl_to_nl_results['references'][i]
                })
                
                complexity_bins["propositions"][prop_bin]["nl_to_ltl"].append({
                    "prediction": nl_to_ltl_results['predictions'][i],
                    "reference": nl_to_ltl_results['references'][i]
                })
        
        complexity_results = {
            "depth": {},
            "operators": {},
            "propositions": {}
        }
        
        for bin_key, bin_data in complexity_bins["depth"].items():
            if len(bin_data["ltl_to_nl"]) >= 5:
                ltl_to_nl_preds = [item["prediction"] for item in bin_data["ltl_to_nl"]]
                ltl_to_nl_refs = [item["reference"] for item in bin_data["ltl_to_nl"]]
                
                bertscore = evaluate.load('bertscore')
                bertscore_results = bertscore.compute(predictions=ltl_to_nl_preds, references=ltl_to_nl_refs, lang='en')
                
                # NL->LTL metrics
                nl_to_ltl_preds = [item["prediction"] for item in bin_data["nl_to_ltl"]]
                nl_to_ltl_refs = [item["reference"] for item in bin_data["nl_to_ltl"]]
                
                # Calculate exact match
                exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(nl_to_ltl_preds, nl_to_ltl_refs)]
                
                # Calculate semantic equivalence
                semantic_equiv = None
                if spot:
                    equiv_count = 0
                    valid_count = 0
                    
                    for pred, ref in zip(nl_to_ltl_preds, nl_to_ltl_refs):
                        try:
                            pred_formula = spot.formula(pred.strip())
                            ref_formula = spot.formula(ref.strip())
                            
                            if pred_formula.equivalent_to(ref_formula):
                                equiv_count += 1
                            
                            valid_count += 1
                        except Exception:
                            continue
                    
                    semantic_equiv = equiv_count / valid_count if valid_count > 0 else 0.0
                
                complexity_results["depth"][bin_key] = {
                    "count": len(bin_data["ltl_to_nl"]),
                    "ltl_to_nl": {
                        "bertscore": np.mean(bertscore_results['f1'])
                    },
                    "nl_to_ltl": {
                        "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
                        "semantic_equivalence": semantic_equiv
                    }
                }
        
        for complexity_type in ["operators", "propositions"]:
            for bin_key, bin_data in complexity_bins[complexity_type].items():
                if len(bin_data["ltl_to_nl"]) >= 5:
                    ltl_to_nl_preds = [item["prediction"] for item in bin_data["ltl_to_nl"]]
                    ltl_to_nl_refs = [item["reference"] for item in bin_data["ltl_to_nl"]]
                    
                    bertscore = evaluate.load('bertscore')
                    bertscore_results = bertscore.compute(predictions=ltl_to_nl_preds, references=ltl_to_nl_refs, lang='en')
                    
                    nl_to_ltl_preds = [item["prediction"] for item in bin_data["nl_to_ltl"]]
                    nl_to_ltl_refs = [item["reference"] for item in bin_data["nl_to_ltl"]]
                    
                    exact_matches = [pred.strip() == ref.strip() for pred, ref in zip(nl_to_ltl_preds, nl_to_ltl_refs)]
                    
                    semantic_equiv = None
                    if spot:
                        equiv_count = 0
                        valid_count = 0
                        
                        for pred, ref in zip(nl_to_ltl_preds, nl_to_ltl_refs):
                            try:
                                pred_formula = spot.formula(pred.strip())
                                ref_formula = spot.formula(ref.strip())
                                
                                if pred_formula.equivalent_to(ref_formula):
                                    equiv_count += 1
                                
                                valid_count += 1
                            except Exception:
                                continue
                        
                        semantic_equiv = equiv_count / valid_count if valid_count > 0 else 0.0
                    
                    complexity_results[complexity_type][bin_key] = {
                        "count": len(bin_data["ltl_to_nl"]),
                        "ltl_to_nl": {
                            "bertscore": np.mean(bertscore_results['f1'])
                        },
                        "nl_to_ltl": {
                            "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
                            "semantic_equivalence": semantic_equiv
                        }
                    }
    
    final_results = {
        "experiment": "Sensitivity to Logical Complexity",
        "model": model_name,
        "complexity_analysis": complexity_results if 'complexity_results' in locals() else {}
    }
    
    results_path = os.path.join(CONFIG['results_dir'], 'experiments', 'complexity_sensitivity.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Complexity Sensitivity experiment results saved to {results_path}")
    return final_results

def run_all_experiments(train_data, val_data, test_data, domains, model_names=None):
    """Run all experiments with selected models."""
    if model_names is None:
        model_names = ['t5-base', 'bart-base']
    
    logger.info(f"Running all experiments with models: {model_names}")
    
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"Running experiments with model: {model_name}")
        
        try:
            # Experiment A: Value of ITL
            value_of_itl_results = run_experiment_value_of_itl(train_data, val_data, test_data, model_name)
            # Experiment B: Impact of Context
            impact_of_context_results = run_experiment_impact_of_context(train_data, val_data, test_data, model_name)
            # Experiment C: Domain Generalization
            domain_gen_results = run_experiment_domain_generalization(train_data, val_data, test_data, domains, model_name)
            # Experiment D: Sensitivity to Complexity
            complexity_results = run_experiment_sensitivity_to_complexity(train_data, val_data, test_data, model_name)
            
            all_results[model_name] = {
                "value_of_itl": value_of_itl_results,
                "impact_of_context": impact_of_context_results,
                "domain_generalization": domain_gen_results,
                "complexity_sensitivity": complexity_results
            }
        except Exception as e:
            logger.error(f"Error running experiments with model {model_name}: {str(e)}")
            all_results[model_name] = {"error": str(e)}
    
    all_results_path = os.path.join(CONFIG['results_dir'], 'all_experiments_results.json')
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All experiment results saved to {all_results_path}")
    return all_results

def main():
    """Main execution function."""
    start_time = time.time()
    logger.info("Starting Verify benchmark evaluation")
    setup_directory_structure()
    copy_database()
    data, domains = load_data_from_db()
    
    train_data, val_data, test_data = split_data(
        data, 
        test_size=CONFIG['test_size'], 
        val_size=CONFIG['val_size'], 
        stratify_by='domain'
    )
    
    if DEBUG_MODE:
        logger.warning("DEBUG MODE: Adjusting CONFIG parameters for speed.")
        CONFIG['batch_size'] = 2 
        CONFIG['eval_batch_size'] = 2
        CONFIG['max_source_length'] = 128
        CONFIG['max_target_length'] = 128
        CONFIG['num_train_epochs'] = 1
        CONFIG['logging_steps'] = 10
        CONFIG['eval_steps'] = 40
        CONFIG['save_steps'] = 80
        CONFIG['gradient_accumulation_steps'] = 1
        CONFIG['fp16'] = False
        CONFIG['checkpointing'] = False
        CONFIG['num_beams'] = 1
        CONFIG['human_eval_samples'] = 5

    use_multi_gpu = setup_distributed_training()
    
    if DEBUG_MODE:
        # Select only one, small model
        model_names = ['t5-base']
        logger.warning(f"DEBUG MODE: Running ONLY with model: {model_names[0]}")
        # Run only ONE core task to test the pipeline
        logger.warning("DEBUG MODE: Running ONLY 'ltl_to_nl' task.")
    else:
        model_names = ['t5-base', 'bart-base', 't5-large', 'bart-large', 'llama-3-8b', 'deepseek-coder', 'mistral-7b', 'codellama-7b']

        logger.info(f"Running with models: {model_names}")
        
        for model_name in model_names:
            logger.info(f"Running all tasks with model: {model_name}")
            run_task('ltl_to_nl', model_name, train_data, val_data, test_data)
            run_task('itl_to_nl', model_name, train_data, val_data, test_data)
            run_task('nl_to_ltl', model_name, train_data, val_data, test_data)
            run_task('nl_to_itl', model_name, train_data, val_data, test_data)
            run_task('ltl_to_itl', model_name, train_data, val_data, test_data)
            run_task('itl_to_ltl', model_name, train_data, val_data, test_data)
        
        # Run all experiments with the best performing models
        run_all_experiments(train_data, val_data, test_data, domains, model_names[:2])
        
    if n_gpus > 1 and dist.is_initialized():
        dist.destroy_process_group()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Verify benchmark evaluation completed in {elapsed_time/3600:.2f} hours")

if __name__ == "__main__":
    print("Running Verify benchmark evaluation...")
    main()