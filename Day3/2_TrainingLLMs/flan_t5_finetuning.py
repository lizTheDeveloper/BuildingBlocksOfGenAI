"""
Flan-T5 Fine-tuning Exercise
--------------------------
This file demonstrates how to perform full fine-tuning of a Flan-T5 model
for a specific task and evaluate its performance before and after fine-tuning
to observe potential "catastrophic forgetting".
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def introduction_to_flan_t5():
    """
    Introduction to Flan-T5 models and their capabilities
    """
    print("=" * 60)
    print("FLAN-T5 FINE-TUNING EXERCISE")
    print("=" * 60)
    
    print("\nAbout Flan-T5:")
    print("Flan-T5 is a collection of models fine-tuned from T5 using")
    print("the 'FLAN' instruction tuning methodology. These models were")
    print("developed by Google and are particularly well-suited for")
    print("instruction-based fine-tuning.")
    
    print("\nKey characteristics:")
    print("- Trained on a mixture of tasks with instructions")
    print("- Strong zero-shot capabilities across many tasks")
    print("- Available in multiple sizes (small to xxl)")
    print("- Uses an encoder-decoder architecture")
    print("- Designed to follow natural language instructions")
    
    print("\nIn this exercise, we will:")
    print("1. Evaluate Flan-T5 on both our target task and general tasks")
    print("2. Fine-tune the model on a specific dataset")
    print("3. Re-evaluate to measure performance gains on the target task")
    print("4. Check for catastrophic forgetting on general capabilities")

def prepare_dataset(dataset_name="samsum"):
    """
    Prepare the dataset for fine-tuning
    
    Args:
        dataset_name (str): Name of the dataset to use
        
    Returns:
        datasets: Training and validation datasets
    """
    print("\n" + "=" * 60)
    print(f"PREPARING THE {dataset_name.upper()} DATASET")
    print("=" * 60)
    
    # For the exercise, we'll use the SAMSum dataset (conversation summarization)
    print(f"\nLoading {dataset_name} dataset...")
    
    if dataset_name == "samsum":
        dataset = load_dataset("samsum")
        
        # Define a function to format the examples
        def format_samsum_example(example):
            # Format as an instruction
            example["input"] = f"Summarize the following conversation:\n\n{example['dialogue']}"
            example["output"] = example["summary"]
            return example
        
        # Apply the formatting
        dataset = dataset.map(format_samsum_example)
        
        print(f"\nDataset loaded. Statistics:")
        print(f"- Training examples: {len(dataset['train'])}")
        print(f"- Validation examples: {len(dataset['validation'])}")
        print(f"- Test examples: {len(dataset['test'])}")
        
        # Display an example
        print("\nExample from the dataset:")
        example = dataset["train"][0]
        print(f"Input: {example['input'][:200]}...")
        print(f"Output: {example['output']}")
        
        return dataset
    
    elif dataset_name == "squad_v2":
        dataset = load_dataset("squad_v2")
        
        # Define a function to format the examples
        def format_squad_example(example):
            # Format as a question-answering instruction
            example["input"] = f"Answer the following question based on the context. If the answer isn't in the context, say 'unanswerable'.\n\nContext: {example['context']}\n\nQuestion: {example['question']}"
            example["output"] = example.get("answers", {}).get("text", [""])[0] if example.get("answers", {}).get("text", [""]) else "unanswerable"
            return example
        
        # Apply the formatting
        dataset = dataset.map(format_squad_example)
        
        print(f"\nDataset loaded. Statistics:")
        print(f"- Training examples: {len(dataset['train'])}")
        print(f"- Validation examples: {len(dataset['validation'])}")
        
        # Display an example
        print("\nExample from the dataset:")
        example = dataset["train"][0]
        print(f"Input: {example['input'][:200]}...")
        print(f"Output: {example['output']}")
        
        return dataset
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def tokenize_dataset(dataset, tokenizer, max_input_length=512, max_output_length=128):
    """
    Tokenize the dataset for fine-tuning
    
    Args:
        dataset: The dataset to tokenize
        tokenizer: The tokenizer to use
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        
    Returns:
        tokenized_dataset: The tokenized dataset
    """
    print("\n" + "=" * 60)
    print("TOKENIZING THE DATASET")
    print("=" * 60)
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input_length,
            padding="max_length",
            truncation=True
        )
        
        labels = tokenizer(
            examples["output"],
            max_length=max_output_length,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply tokenization to the dataset
    print("\nTokenizing the dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names  # Remove original columns
    )
    
    print("\nTokenization complete.")
    print(f"- Example input_ids length: {len(tokenized_dataset['train'][0]['input_ids'])}")
    print(f"- Example labels length: {len(tokenized_dataset['train'][0]['labels'])}")
    
    return tokenized_dataset

def evaluate_before_finetuning(model, tokenizer, eval_dataset, general_tasks=None):
    """
    Evaluate the model before fine-tuning to establish a baseline
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        eval_dataset: The evaluation dataset for the target task
        general_tasks: A list of general tasks to evaluate
        
    Returns:
        results: Dictionary of evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL BEFORE FINE-TUNING")
    print("=" * 60)
    
    # Initialize evaluation metrics
    rouge = evaluate.load("rouge")
    
    # Define the compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode the generated outputs and reference summaries
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 (padding token ID for labels) with the pad token ID
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }
    
    # Set up training arguments for evaluation only
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
    )
    
    # Create the trainer for evaluation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate on the target task
    print("\nEvaluating on the target task...")
    target_results = trainer.evaluate(eval_dataset=eval_dataset["validation"])
    
    print("\nBaseline performance on target task:")
    for metric, value in target_results.items():
        print(f"- {metric}: {value:.4f}")
    
    results = {"target_task": target_results}
    
    # Evaluate on general tasks (if provided)
    if general_tasks:
        print("\nEvaluating on general tasks...")
        general_results = {}
        
        for task_name, task_data in general_tasks.items():
            print(f"\nEvaluating on {task_name}...")
            
            # Sample a few examples for quick evaluation
            sample_inputs = task_data["inputs"][:5]
            sample_outputs = task_data["outputs"][:5]
            
            generated_outputs = []
            for input_text in sample_inputs:
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate output
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_outputs.append(decoded_output)
            
            # Compute task-specific metrics
            if task_name == "question_answering":
                # Simple exact match for QA
                exact_matches = [gen.strip() == ref.strip() for gen, ref in zip(generated_outputs, sample_outputs)]
                accuracy = sum(exact_matches) / len(exact_matches)
                task_result = {"accuracy": accuracy}
            
            elif task_name == "translation":
                # Use BLEU score for translation
                bleu = evaluate.load("bleu")
                references = [[ref.split()] for ref in sample_outputs]  # BLEU expects tokenized references
                task_result = bleu.compute(predictions=[gen.split() for gen in generated_outputs], references=references)
            
            else:
                # Default to ROUGE for other tasks
                task_result = rouge.compute(predictions=generated_outputs, references=sample_outputs)
            
            # Print evaluation results
            print(f"Performance on {task_name}:")
            for metric, value in task_result.items():
                print(f"- {metric}: {value:.4f}")
            
            # Store the results
            general_results[task_name] = task_result
            
            # Print a few examples
            print("\nExample predictions:")
            for i, (input_text, ref, gen) in enumerate(zip(sample_inputs[:2], sample_outputs[:2], generated_outputs[:2])):
                print(f"\nExample {i+1}:")
                print(f"Input: {input_text[:100]}...")
                print(f"Reference: {ref}")
                print(f"Generated: {gen}")
        
        results["general_tasks"] = general_results
    
    return results

def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./finetuned_flan_t5"):
    """
    Fine-tune the Flan-T5 model on the target dataset
    
    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        output_dir: Directory to save the fine-tuned model
        
    Returns:
        trainer: The trained Seq2SeqTrainer
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING FLAN-T5 MODEL")
    print("=" * 60)
    
    # Initialize evaluation metrics
    rouge = evaluate.load("rouge")
    
    # Define the compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode the generated outputs and reference summaries
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 (padding token ID for labels) with the pad token ID
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,  # Enable mixed precision training
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=100,
        push_to_hub=False,  # Set to True to push the model to the Hugging Face Hub
    )
    
    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Start fine-tuning
    print("\nStarting fine-tuning process...")
    trainer.train()
    
    # Evaluate the model
    print("\nEvaluating the fine-tuned model...")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset["validation"])
    
    print("\nFine-tuning complete. Final evaluation results:")
    for metric, value in eval_results.items():
        print(f"- {metric}: {value:.4f}")
    
    # Save the model
    print(f"\nSaving fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return trainer

def evaluate_after_finetuning(model, tokenizer, eval_dataset, general_tasks=None, baseline_results=None):
    """
    Evaluate the model after fine-tuning to measure improvements and check for
    catastrophic forgetting
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        eval_dataset: The evaluation dataset for the target task
        general_tasks: A list of general tasks to evaluate
        baseline_results: Results from before fine-tuning for comparison
        
    Returns:
        results: Dictionary of evaluation results
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL AFTER FINE-TUNING")
    print("=" * 60)
    
    # Initialize evaluation metrics
    rouge = evaluate.load("rouge")
    
    # Define the compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode the generated outputs and reference summaries
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 (padding token ID for labels) with the pad token ID
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"]
        }
    
    # Set up training arguments for evaluation only
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
    )
    
    # Create the trainer for evaluation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate on the target task
    print("\nEvaluating on the target task...")
    target_results = trainer.evaluate(eval_dataset=eval_dataset["validation"])
    
    print("\nFine-tuned performance on target task:")
    for metric, value in target_results.items():
        print(f"- {metric}: {value:.4f}")
    
    # Compare with baseline if available
    if baseline_results and "target_task" in baseline_results:
        print("\nComparison with baseline:")
        for metric in target_results:
            if metric in baseline_results["target_task"]:
                baseline = baseline_results["target_task"][metric]
                finetuned = target_results[metric]
                change = finetuned - baseline
                change_pct = (change / baseline) * 100 if baseline != 0 else float('inf')
                
                print(f"- {metric}: {baseline:.4f} → {finetuned:.4f} ({change_pct:+.2f}%)")
    
    results = {"target_task": target_results}
    
    # Evaluate on general tasks (if provided)
    if general_tasks:
        print("\nEvaluating on general tasks...")
        general_results = {}
        
        for task_name, task_data in general_tasks.items():
            print(f"\nEvaluating on {task_name}...")
            
            # Sample a few examples for quick evaluation
            sample_inputs = task_data["inputs"][:5]
            sample_outputs = task_data["outputs"][:5]
            
            generated_outputs = []
            for input_text in sample_inputs:
                inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate output
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_outputs.append(decoded_output)
            
            # Compute task-specific metrics
            if task_name == "question_answering":
                # Simple exact match for QA
                exact_matches = [gen.strip() == ref.strip() for gen, ref in zip(generated_outputs, sample_outputs)]
                accuracy = sum(exact_matches) / len(exact_matches)
                task_result = {"accuracy": accuracy}
            
            elif task_name == "translation":
                # Use BLEU score for translation
                bleu = evaluate.load("bleu")
                references = [[ref.split()] for ref in sample_outputs]  # BLEU expects tokenized references
                task_result = bleu.compute(predictions=[gen.split() for gen in generated_outputs], references=references)
            
            else:
                # Default to ROUGE for other tasks
                task_result = rouge.compute(predictions=generated_outputs, references=sample_outputs)
            
            # Print evaluation results
            print(f"Performance on {task_name}:")
            for metric, value in task_result.items():
                print(f"- {metric}: {value:.4f}")
            
            # Store the results
            general_results[task_name] = task_result
            
            # Compare with baseline if available
            if baseline_results and "general_tasks" in baseline_results and task_name in baseline_results["general_tasks"]:
                print(f"\nComparison with baseline on {task_name}:")
                baseline_task = baseline_results["general_tasks"][task_name]
                
                for metric in task_result:
                    if metric in baseline_task:
                        baseline = baseline_task[metric]
                        finetuned = task_result[metric]
                        change = finetuned - baseline
                        change_pct = (change / baseline) * 100 if baseline != 0 else float('inf')
                        
                        print(f"- {metric}: {baseline:.4f} → {finetuned:.4f} ({change_pct:+.2f}%)")
                        
                        # Check for catastrophic forgetting
                        if change < 0 and abs(change_pct) > 10:  # More than 10% decrease
                            print(f"  WARNING: Possible catastrophic forgetting detected on {task_name} ({metric})!")
            
            # Print a few examples
            print("\nExample predictions:")
            for i, (input_text, ref, gen) in enumerate(zip(sample_inputs[:2], sample_outputs[:2], generated_outputs[:2])):
                print(f"\nExample {i+1}:")
                print(f"Input: {input_text[:100]}...")
                print(f"Reference: {ref}")
                print(f"Generated: {gen}")
        
        results["general_tasks"] = general_results
    
    return results

def visualize_results(baseline_results, finetuned_results):
    """
    Visualize the comparison between baseline and fine-tuned model performance
    
    Args:
        baseline_results: Results from before fine-tuning
        finetuned_results: Results from after fine-tuning
    """
    print("\n" + "=" * 60)
    print("VISUALIZING RESULTS")
    print("=" * 60)
    
    # Extract the target task results
    target_baseline = baseline_results["target_task"]
    target_finetuned = finetuned_results["target_task"]
    
    # Plot the target task results
    print("\nTarget Task Performance Comparison:")
    
    # Prepare the data for plotting
    metrics = []
    baseline_values = []
    finetuned_values = []
    
    for metric in target_baseline:
        if metric.startswith("eval"):
            # Skip the evaluation loss and other non-metric values
            continue
            
        metrics.append(metric)
        baseline_values.append(target_baseline[metric])
        finetuned_values.append(target_finetuned[metric])
    
    print("\nPlotting Code for Target Task Comparison:")
    print("""
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Set the width of the bars
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    # Create the bars
    plt.bar(index, baseline_values, bar_width, label='Baseline', color='skyblue')
    plt.bar(index + bar_width, finetuned_values, bar_width, label='Fine-tuned', color='orange')
    
    # Add labels and title
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Target Task Performance Comparison')
    plt.xticks(index + bar_width / 2, metrics)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('target_task_comparison.png')
    plt.show()
    """)
    
    # If general tasks were evaluated, plot those too
    if "general_tasks" in baseline_results and "general_tasks" in finetuned_results:
        general_baseline = baseline_results["general_tasks"]
        general_finetuned = finetuned_results["general_tasks"]
        
        print("\nGeneral Tasks Performance Comparison:")
        
        # For each general task
        for task_name in general_baseline:
            if task_name not in general_finetuned:
                continue
            
            baseline_task = general_baseline[task_name]
            finetuned_task = general_finetuned[task_name]
            
            # Prepare data for plotting
            metrics = []
            baseline_values = []
            finetuned_values = []
            
            for metric in baseline_task:
                if metric in finetuned_task:
                    metrics.append(metric)
                    baseline_values.append(baseline_task[metric])
                    finetuned_values.append(finetuned_task[metric])
            
            print(f"\nPlotting Code for {task_name} Comparison:")
            print(f"""
            # Set up the plot
            plt.figure(figsize=(10, 6))
            
            # Set the width of the bars
            bar_width = 0.35
            index = np.arange(len({metrics}))
            
            # Create the bars
            plt.bar(index, {baseline_values}, bar_width, label='Baseline', color='skyblue')
            plt.bar(index + bar_width, {finetuned_values}, bar_width, label='Fine-tuned', color='orange')
            
            # Add labels and title
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('{task_name} Performance Comparison')
            plt.xticks(index + bar_width / 2, {metrics})
            plt.legend()
            
            # Show the plot
            plt.tight_layout()
            plt.savefig('{task_name}_comparison.png')
            plt.show()
            """)
    
    # Plot a catastrophic forgetting visualization if applicable
    if "general_tasks" in baseline_results and "general_tasks" in finetuned_results:
        print("\nCatastrophic Forgetting Visualization:")
        
        # Collect changes across all general tasks
        task_names = []
        metric_changes = []
        
        for task_name in general_baseline:
            if task_name not in general_finetuned:
                continue
            
            baseline_task = general_baseline[task_name]
            finetuned_task = general_finetuned[task_name]
            
            # Calculate average change across all metrics for this task
            changes = []
            for metric in baseline_task:
                if metric in finetuned_task:
                    baseline = baseline_task[metric]
                    finetuned = finetuned_task[metric]
                    
                    if baseline != 0:
                        change_pct = ((finetuned - baseline) / baseline) * 100
                        changes.append(change_pct)
            
            if changes:
                avg_change = sum(changes) / len(changes)
                task_names.append(task_name)
                metric_changes.append(avg_change)
        
        print("\nPlotting Code for Catastrophic Forgetting Analysis:")
        print(f"""
        # Set up the plot
        plt.figure(figsize=(12, 6))
        
        # Create the bars
        plt.bar({task_names}, {metric_changes}, color=['green' if x >= 0 else 'red' for x in {metric_changes}])
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Task')
        plt.ylabel('Percent Change (%)')
        plt.title('Performance Change After Fine-tuning (General Tasks)')
        plt.xticks(rotation=45, ha='right')
        
        # Annotate the bars
        for i, v in enumerate({metric_changes}):
            plt.text(i, v + (5 if v >= 0 else -5), f"{v:.1f}%", ha='center', va='bottom' if v >= 0 else 'top')
        
        # Show the plot
        plt.tight_layout()
        plt.savefig('catastrophic_forgetting_analysis.png')
        plt.show()
        """)

def create_general_task_examples():
    """
    Create a set of examples for general tasks to evaluate catastrophic forgetting
    
    Returns:
        general_tasks: Dictionary of general task examples
    """
    general_tasks = {
        "question_answering": {
            "inputs": [
                "Answer the following question: What is the capital of France?",
                "Answer the following question: What is photosynthesis?",
                "Answer the following question: Who wrote the novel '1984'?",
                "Answer the following question: What is the Pythagorean theorem?",
                "Answer the following question: What is the largest planet in our solar system?"
            ],
            "outputs": [
                "Paris",
                "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water.",
                "George Orwell",
                "In a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
                "Jupiter"
            ]
        },
        "translation": {
            "inputs": [
                "Translate to French: Hello, how are you?",
                "Translate to French: I love to learn new languages.",
                "Translate to French: The weather is beautiful today.",
                "Translate to French: Where is the nearest restaurant?",
                "Translate to French: Thank you for your help."
            ],
            "outputs": [
                "Bonjour, comment allez-vous ?",
                "J'aime apprendre de nouvelles langues.",
                "Le temps est magnifique aujourd'hui.",
                "Où est le restaurant le plus proche ?",
                "Merci pour votre aide."
            ]
        },
        "text_classification": {
            "inputs": [
                "Classify the sentiment: I absolutely loved this movie!",
                "Classify the sentiment: The service at this restaurant was terrible.",
                "Classify the sentiment: The product is okay, nothing special.",
                "Classify the sentiment: I can't recommend this book enough.",
                "Classify the sentiment: I was disappointed with the quality."
            ],
            "outputs": [
                "Positive",
                "Negative",
                "Neutral",
                "Positive",
                "Negative"
            ]
        },
        "creative_writing": {
            "inputs": [
                "Write a short poem about nature.",
                "Write a brief story about a lost dog.",
                "Write a haiku about the ocean.",
                "Write a short description of a sunset.",
                "Write a brief dialogue between two old friends."
            ],
            "outputs": [
                "Trees sway in the wind,\nFlowers bloom in vibrant hues,\nNature's art displayed.",
                "Max wagged his tail as he explored the unfamiliar streets. His collar had slipped off during his morning walk. A kind stranger found him, read his tag, and reunited him with his worried family by evening.",
                "Waves crash on the shore,\nSalt spray fills the summer air,\nOcean dreams begin.",
                "The sun dipped below the horizon, painting the sky in vibrant oranges and pinks, casting long shadows across the landscape as day turned to dusk.",
                "\"It's been ages!\"\n\"Too long. You haven't changed a bit.\"\n\"Neither have you, still terrible at lying.\"  Both laughed, years of separation instantly dissolved."
            ]
        }
    }
    
    return general_tasks

def main():
    """
    Main function to run the Flan-T5 fine-tuning exercise
    """
    print("\n" + "=" * 60)
    print("RUNNING FLAN-T5 FINE-TUNING EXERCISE")
    print("=" * 60)
    
    # Introduction
    introduction_to_flan_t5()
    
    # Use a smaller model for the exercise (to be manageable on limited hardware)
    model_name = "google/flan-t5-small"  # Options: small, base, large, xl, xxl
    
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prepare the dataset
    dataset = prepare_dataset("samsum")  # For conversation summarization
    
    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Create general task examples
    general_tasks = create_general_task_examples()
    
    # Evaluate before fine-tuning
    baseline_results = evaluate_before_finetuning(model, tokenizer, dataset, general_tasks)
    
    # Fine-tune the model
    trainer = fine_tune_model(model, tokenizer, tokenized_dataset, dataset)
    
    # Get the fine-tuned model
    finetuned_model = trainer.model
    
    # Evaluate after fine-tuning
    finetuned_results = evaluate_after_finetuning(
        finetuned_model, 
        tokenizer, 
        dataset, 
        general_tasks, 
        baseline_results
    )
    
    # Visualize the results
    visualize_results(baseline_results, finetuned_results)
    
    print("\n" + "=" * 60)
    print("FLAN-T5 FINE-TUNING EXERCISE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
