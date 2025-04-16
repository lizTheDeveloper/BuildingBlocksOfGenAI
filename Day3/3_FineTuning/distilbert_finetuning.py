"""
DistilBERT Fine-tuning Exercise
----------------------------
This file provides code for fine-tuning DistilBERT on sentiment analysis
using both single-task and multi-task approaches, and comparing their performance.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple

# Import helper modules
from models import DistilBERTMultiTaskModel
from trainers import MultiTaskDataCollator, MultiTaskTrainer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def introduction_to_exercise():
    """
    Introduction to the DistilBERT fine-tuning exercise
    """
    print("=" * 60)
    print("DISTILBERT FINE-TUNING EXERCISE")
    print("=" * 60)
    
    print("\nIn this exercise, we'll explore two approaches to fine-tuning DistilBERT:")
    print("1. Single-Task Fine-Tuning: Focused on sentiment analysis")
    print("2. Multi-Task Fine-Tuning: Balancing sentiment analysis with other tasks")
    
    print("\nWe'll then compare their performance, particularly examining:")
    print("- Task-specific performance")
    print("- Generalization capabilities")
    print("- Evidence of catastrophic forgetting")
    print("- Resource efficiency")
    
    print("\nDatasets Used:")
    print("- SST-2: Stanford Sentiment Treebank (Movie Reviews)")
    print("- CoLA: Corpus of Linguistic Acceptability (Grammar Judgments)")
    print("- MNLI: Multi-Genre Natural Language Inference (Entailment)")
    
    print("\nKey Steps:")
    print("1. Load and preprocess datasets")
    print("2. Set up single-task and multi-task models")
    print("3. Train both models")
    print("4. Evaluate and compare performance")
    print("5. Analyze results and draw conclusions")

def load_and_process_datasets():
    """
    Load and preprocess datasets for both single-task and multi-task training
    
    Returns:
        Dict containing all processed datasets
    """
    print("\n" + "=" * 60)
    print("LOADING AND PROCESSING DATASETS")
    print("=" * 60)
    
    # Define tasks and their properties
    tasks = {
        "sst2": {
            "name": "SST-2",
            "description": "Binary sentiment classification (positive/negative)",
            "dataset": "glue",
            "subset": "sst2",
            "num_labels": 2,
            "metric": "accuracy",
            "text_fields": ["sentence"],
            "label_field": "label"
        },
        "cola": {
            "name": "CoLA",
            "description": "Binary grammatical acceptability judgment",
            "dataset": "glue",
            "subset": "cola",
            "num_labels": 2,
            "metric": "matthews_correlation",
            "text_fields": ["sentence"],
            "label_field": "label"
        },
        "mnli": {
            "name": "MNLI",
            "description": "Multi-genre natural language inference (3 classes)",
            "dataset": "glue",
            "subset": "mnli",
            "num_labels": 3,
            "metric": "accuracy",
            "text_fields": ["premise", "hypothesis"],
            "label_field": "label"
        }
    }
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load datasets
    print("\nLoading datasets from Hugging Face Hub...")
    raw_datasets = {}
    for task_id, task_info in tasks.items():
        print(f"Loading {task_info['name']} dataset...")
        raw_datasets[task_id] = load_dataset(task_info["dataset"], task_info["subset"])
        
        # Print some dataset statistics
        train_size = len(raw_datasets[task_id]["train"])
        val_size = len(raw_datasets[task_id]["validation"])
        print(f"  {task_info['name']}: {train_size} training examples, {val_size} validation examples")
    
    # Define tokenization function
    def preprocess_function(examples, task_id):
        task = tasks[task_id]
        
        # Handle different text field formats
        if len(task["text_fields"]) == 1:
            # Single text field (e.g., SST-2, CoLA)
            text_field = task["text_fields"][0]
            inputs = tokenizer(
                examples[text_field],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        else:
            # Two text fields (e.g., MNLI)
            text_field1, text_field2 = task["text_fields"]
            inputs = tokenizer(
                examples[text_field1],
                examples[text_field2],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        
        # Add labels
        inputs["labels"] = examples[task["label_field"]]
        
        # Add task_id for multi-task training
        inputs["task_id"] = task_id
        
        return inputs
    
    # Process all datasets
    tokenized_datasets = {}
    for task_id in tasks:
        tokenized_datasets[task_id] = {}
        
        # Process each split (train, validation, test if available)
        for split in raw_datasets[task_id]:
            # Create a task-specific preprocessing function
            preprocess_task = lambda examples: preprocess_function(examples, task_id)
            
            # Apply preprocessing
            tokenized_datasets[task_id][split] = raw_datasets[task_id][split].map(
                preprocess_task,
                batched=True,
                remove_columns=raw_datasets[task_id][split].column_names
            )
    
    print("\nDataset preprocessing complete!")
    
    # Create combined datasets for multi-task training
    multitask_train_datasets = []
    multitask_val_datasets = []
    
    for task_id in tasks:
        multitask_train_datasets.append(tokenized_datasets[task_id]["train"])
        multitask_val_datasets.append(tokenized_datasets[task_id]["validation"])
    
    multitask_datasets = {
        "train": concatenate_datasets(multitask_train_datasets),
        "validation": concatenate_datasets(multitask_val_datasets)
    }
    
    # Print statistics on the combined multi-task dataset
    print("\nMulti-task dataset statistics:")
    print(f"Training examples: {len(multitask_datasets['train'])}")
    print(f"Validation examples: {len(multitask_datasets['validation'])}")
    
    # Count examples per task in the multi-task dataset
    task_counts = {}
    for split in ["train", "validation"]:
        task_counts[split] = {}
        task_ids = multitask_datasets[split]["task_id"]
        unique_tasks, counts = np.unique(task_ids, return_counts=True)
        
        for task, count in zip(unique_tasks, counts):
            task_counts[split][task] = count
            percentage = count / len(task_ids) * 100
            print(f"  {split} - {task}: {count} examples ({percentage:.1f}%)")
    
    # Store processed datasets
    processed_data = {
        "tasks": tasks,
        "tokenizer": tokenizer,
        "single_task": tokenized_datasets,
        "multi_task": multitask_datasets,
        "task_counts": task_counts
    }
    
    return processed_data

def single_task_finetuning(processed_data, task_id="sst2", output_dir="./results/single_task_distilbert"):
    """
    Implement single-task fine-tuning for DistilBERT
    
    Args:
        processed_data: Dict containing processed datasets
        task_id: ID of the task to fine-tune on
        output_dir: Directory to save the fine-tuned model
        
    Returns:
        Tuple of (model, trainer, metrics)
    """
    print("\n" + "=" * 60)
    print(f"SINGLE-TASK FINE-TUNING: {processed_data['tasks'][task_id]['name']}")
    print("=" * 60)
    
    # Get task information
    task_info = processed_data["tasks"][task_id]
    tokenizer = processed_data["tokenizer"]
    tokenized_datasets = processed_data["single_task"][task_id]
    
    print(f"\nFine-tuning DistilBERT for {task_info['name']} task:")
    print(f"- Task: {task_info['description']}")
    print(f"- Number of labels: {task_info['num_labels']}")
    print(f"- Evaluation metric: {task_info['metric']}")
    
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=task_info["num_labels"]
    )
    
    # Define compute metrics function based on task
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        
        return {
            "accuracy": accuracy,
            "f1": f1
        }
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_steps=100,
        report_to="none"  # Disable wandb, tensorboard etc.
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Log training metrics
    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate model
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    
    print("\nValidation metrics:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, trainer, eval_results

def multi_task_finetuning(processed_data, output_dir="./results/multi_task_distilbert"):
    """
    Implement multi-task fine-tuning for DistilBERT
    
    Args:
        processed_data: Dict containing processed datasets
        output_dir: Directory to save the fine-tuned model
        
    Returns:
        Tuple of (model, trainer, metrics)
    """
    print("\n" + "=" * 60)
    print("MULTI-TASK FINE-TUNING")
    print("=" * 60)
    
    # Get data
    tasks = processed_data["tasks"]
    tokenizer = processed_data["tokenizer"]
    multitask_datasets = processed_data["multi_task"]
    
    print("\nFine-tuning DistilBERT on multiple tasks:")
    for task_id, task_info in tasks.items():
        print(f"- {task_info['name']}: {task_info['description']}")
    
    # Initialize multi-task model
    model = DistilBERTMultiTaskModel(tasks)
    
    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        
        return {
            "accuracy": accuracy,
            "f1": f1
        }
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        logging_steps=100,
        report_to="none"  # Disable wandb, tensorboard etc.
    )
    
    # Initialize custom data collator
    data_collator = MultiTaskDataCollator(tokenizer)
    
    # Initialize multi-task trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=multitask_datasets["train"],
        eval_dataset=multitask_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\nStarting multi-task training...")
    train_result = trainer.train()
    
    # Log training metrics
    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate model
    print("\nEvaluating on combined validation set...")
    eval_results = trainer.evaluate()
    
    print("\nValidation metrics:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate separately on each task
    task_eval_results = {}
    for task_id, task_info in tasks.items():
        print(f"\nEvaluating on {task_info['name']} task...")
        task_eval_dataset = processed_data["single_task"][task_id]["validation"]
        
        # Ensure task_id is set correctly
        task_eval_dataset = task_eval_dataset.map(lambda x: {"task_id": task_id})
        
        task_results = trainer.evaluate(eval_dataset=task_eval_dataset)
        task_eval_results[task_id] = task_results
        
        print(f"{task_info['name']} metrics:")
        for key, value in task_results.items():
            print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Combine all evaluation results
    all_eval_results = {
        "combined": eval_results,
        "per_task": task_eval_results
    }
    
    return model, trainer, all_eval_results

def evaluate_models_on_transfer_tasks(processed_data, 
                                     single_task_model_path="./results/single_task_distilbert",
                                     multi_task_model_path="./results/multi_task_distilbert",
                                     transfer_dataset_name="imdb"):
    """
    Evaluate both models on transfer tasks to assess generalization
    
    Args:
        processed_data: Dict containing processed datasets
        single_task_model_path: Path to single-task model
        multi_task_model_path: Path to multi-task model
        transfer_dataset_name: Name of transfer dataset
        
    Returns:
        Dict of evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODELS ON TRANSFER TASKS")
    print("=" * 60)
    
    tasks = processed_data["tasks"]
    tokenizer = processed_data["tokenizer"]
    
    # Load transfer dataset
    print(f"\nLoading transfer dataset: {transfer_dataset_name}")
    if transfer_dataset_name == "imdb":
        transfer_dataset = load_dataset("imdb", split="test[:1000]")
        text_field = "text"
        label_field = "label"
        num_labels = 2
    else:
        raise ValueError(f"Transfer dataset {transfer_dataset_name} not supported")
    
    # Preprocess transfer dataset
    def preprocess_transfer(examples):
        inputs = tokenizer(
            examples[text_field],
            padding="max_length",
            truncation=True,
            max_length=256  # Longer for IMDB
        )
        inputs["labels"] = examples[label_field]
        return inputs
    
    # Process dataset
    processed_transfer = transfer_dataset.map(
        preprocess_transfer,
        batched=True,
        remove_columns=transfer_dataset.column_names
    )
    
    print(f"Processed {len(processed_transfer)} examples from transfer dataset")
    
    # Load single-task model
    print("\nLoading single-task model...")
    single_task_model = AutoModelForSequenceClassification.from_pretrained(
        single_task_model_path,
        num_labels=num_labels
    )
    
    # Load multi-task model (assuming it's a custom model saved with torch.save)
    print("\nLoading multi-task model...")
    # This would need to be adapted based on how the multi-task model is saved
    multi_task_model = torch.load(f"{multi_task_model_path}/pytorch_model.bin")
    multi_task_model = DistilBERTMultiTaskModel(tasks)
    multi_task_model.load_state_dict(torch.load(f"{multi_task_model_path}/pytorch_model.bin"))
    
    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        cf_matrix = confusion_matrix(labels, preds)
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": cf_matrix
        }
    
    # Evaluate single-task model
    print("\nEvaluating single-task model on transfer dataset...")
    single_task_trainer = Trainer(
        model=single_task_model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    single_task_results = single_task_trainer.evaluate(processed_transfer)
    
    print("\nSingle-task model transfer results:")
    for key, value in single_task_results.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")
    
    # Evaluate multi-task model
    print("\nEvaluating multi-task model on transfer dataset...")
    # For multi-task model, we need to decide which task head to use
    # Let's use SST-2 head since it's most similar to IMDB
    transfer_task_id = "sst2"
    
    # Create a simple wrapper for evaluation
    class MultiTaskWrapper(nn.Module):
        def __init__(self, multi_task_model, task_id):
            super().__init__()
            self.model = multi_task_model
            self.task_id = task_id
        
        def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_id=self.task_id,
                labels=labels
            )
            
            return outputs
    
    # Wrap multi-task model
    wrapped_model = MultiTaskWrapper(multi_task_model, transfer_task_id)
    
    multi_task_trainer = Trainer(
        model=wrapped_model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    multi_task_results = multi_task_trainer.evaluate(processed_transfer)
    
    print("\nMulti-task model transfer results:")
    for key, value in multi_task_results.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")
    
    # Compare results
    print("\nTransfer Learning Performance Comparison:")
    for metric in ["eval_accuracy", "eval_f1"]:
        if metric in single_task_results and metric in multi_task_results:
            single_task_value = single_task_results[metric]
            multi_task_value = multi_task_results[metric]
            diff = multi_task_value - single_task_value
            diff_percent = (diff / single_task_value) * 100 if single_task_value != 0 else float('inf')
            
            print(f"  {metric[5:]}: Single-Task: {single_task_value:.4f}, Multi-Task: {multi_task_value:.4f}")
            print(f"    Difference: {diff:.4f} ({diff_percent:+.2f}%)")
    
    # Return all results
    transfer_results = {
        "single_task": single_task_results,
        "multi_task": multi_task_results,
        "dataset": transfer_dataset_name
    }
    
    return transfer_results

def visualize_results(single_task_results, multi_task_results, transfer_results=None):
    """
    Visualize and compare performance between single-task and multi-task models
    
    Args:
        single_task_results: Results from single-task fine-tuning
        multi_task_results: Results from multi-task fine-tuning
        transfer_results: Results from transfer task evaluation
    """
    print("\n" + "=" * 60)
    print("VISUALIZATION OF RESULTS")
    print("=" * 60)
    
    # Extract metrics for original tasks
    original_task_metrics = {
        "single_task": {
            "accuracy": single_task_results.get("eval_accuracy", 0),
            "f1": single_task_results.get("eval_f1", 0)
        },
        "multi_task": {}
    }
    
    # Extract multi-task per-task results
    if isinstance(multi_task_results, dict) and "per_task" in multi_task_results:
        for task_id, task_results in multi_task_results["per_task"].items():
            original_task_metrics["multi_task"][task_id] = {
                "accuracy": task_results.get("eval_accuracy", 0),
                "f1": task_results.get("eval_f1", 0)
            }
    
    # Create bar chart comparing performance
    metrics = ["accuracy", "f1"]
    
    # Performance on original task
    plt.figure(figsize=(10, 6))
    
    # Set up bars
    x = np.arange(len(metrics))
    width = 0.35
    
    single_task_values = [original_task_metrics["single_task"][m] for m in metrics]
    multi_task_sst2_values = [original_task_metrics["multi_task"].get("sst2", {}).get(f"eval_{m}", 0) for m in metrics]
    
    plt.bar(x - width/2, single_task_values, width, label='Single-Task', color='skyblue')
    plt.bar(x + width/2, multi_task_sst2_values, width, label='Multi-Task (SST-2)', color='orange')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Original Task Performance (SST-2)')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results/original_performance.png')
    print("\nSaved visualization of original task performance to './results/original_performance.png'")
    
    # If transfer results available, create plot for transfer performance
    if transfer_results:
        plt.figure(figsize=(10, 6))
        
        transfer_single_values = [
            transfer_results["single_task"].get("eval_accuracy", 0),
            transfer_results["single_task"].get("eval_f1", 0)
        ]
        
        transfer_multi_values = [
            transfer_results["multi_task"].get("eval_accuracy", 0),
            transfer_results["multi_task"].get("eval_f1", 0)
        ]
        
        plt.bar(x - width/2, transfer_single_values, width, label='Single-Task', color='skyblue')
        plt.bar(x + width/2, transfer_multi_values, width, label='Multi-Task', color='orange')
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(f'Transfer Performance ({transfer_results["dataset"]})')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./results/transfer_performance.png')
        print(f"\nSaved visualization of transfer performance to './results/transfer_performance.png'")

def main():
    """
    Main function to run the exercise
    """
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/single_task_distilbert", exist_ok=True)
    os.makedirs("./results/multi_task_distilbert", exist_ok=True)
    
    # Introduction
    introduction_to_exercise()
    
    # Load and process datasets
    processed_data = load_and_process_datasets()
    
    # Single-task fine-tuning
    single_task_model, single_task_trainer, single_task_results = single_task_finetuning(
        processed_data,
        task_id="sst2",
        output_dir="./results/single_task_distilbert"
    )
    
    # Multi-task fine-tuning
    multi_task_model, multi_task_trainer, multi_task_results = multi_task_finetuning(
        processed_data,
        output_dir="./results/multi_task_distilbert"
    )
    
    # Evaluate on transfer task
    transfer_results = evaluate_models_on_transfer_tasks(
        processed_data,
        single_task_model_path="./results/single_task_distilbert",
        multi_task_model_path="./results/multi_task_distilbert",
        transfer_dataset_name="imdb"
    )
    
    # Visualize results
    visualize_results(
        single_task_results,
        multi_task_results,
        transfer_results
    )
    
    print("\n" + "=" * 60)
    print("DISTILBERT FINE-TUNING EXERCISE COMPLETED")
    print("=" * 60)
    
    print("\nSummary of Results:")
    print("\n1. Single-Task Performance:")
    for key, value in single_task_results.items():
        if not key.startswith("eval_"):
            continue
        print(f"  {key[5:]}: {value:.4f}")
    
    print("\n2. Multi-Task Performance:")
    if isinstance(multi_task_results, dict) and "per_task" in multi_task_results:
        for task_id, task_results in multi_task_results["per_task"].items():
            print(f"  {task_id}:")
            for key, value in task_results.items():
                if not key.startswith("eval_"):
                    continue
                print(f"    {key[5:]}: {value:.4f}")
    
    print("\n3. Transfer Performance (IMDB):")
    print("  Single-Task Model:")
    for key, value in transfer_results["single_task"].items():
        if key == "confusion_matrix" or not key.startswith("eval_"):
            continue
        print(f"    {key[5:]}: {value:.4f}")
    
    print("  Multi-Task Model:")
    for key, value in transfer_results["multi_task"].items():
        if key == "confusion_matrix" or not key.startswith("eval_"):
            continue
        print(f"    {key[5:]}: {value:.4f}")
    
    print("\nKey Observations:")
    print("1. [Insert your observations here]")
    print("2. [Insert your observations here]")
    print("3. [Insert your observations here]")

if __name__ == "__main__":
    main()
