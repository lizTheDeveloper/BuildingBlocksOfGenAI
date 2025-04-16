"""
Evaluation Utilities for DistilBERT Fine-tuning
------------------------------------------
This file contains functions for evaluating models and analyzing results
from the DistilBERT fine-tuning exercise.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Optional, Any, Union, Tuple

def evaluate_model_on_dataset(model, tokenizer, dataset, task_id=None, batch_size=32):
    """
    Evaluate a model on a dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use for processing
        dataset: Dataset to evaluate on
        task_id: Task ID for multi-task models
        batch_size: Batch size for evaluation
        
    Returns:
        Dict of evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Tracking variables
    all_preds = []
    all_labels = []
    
    # Evaluate in batches
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # Forward pass (handle multi-task models differently)
            if task_id is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id
                )
                logits = outputs["logits"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            
            # Convert logits to predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cf_matrix,
        "classification_report": report
    }

def analyze_catastrophic_forgetting(model, tokenizer, datasets, task_ids, task_names=None):
    """
    Analyze catastrophic forgetting by comparing performance across tasks
    
    Args:
        model: Fine-tuned multi-task model
        tokenizer: Tokenizer to use for processing
        datasets: Dict of datasets for each task
        task_ids: List of task IDs to evaluate
        task_names: Optional nice names for tasks
        
    Returns:
        DataFrame of evaluation metrics for each task
    """
    # Use task_ids as names if not provided
    if task_names is None:
        task_names = task_ids
    
    # Evaluate on each task
    results = []
    for i, task_id in enumerate(task_ids):
        task_name = task_names[i]
        dataset = datasets[task_id]
        
        print(f"Evaluating on {task_name} task...")
        metrics = evaluate_model_on_dataset(model, tokenizer, dataset, task_id=task_id)
        
        # Store results
        results.append({
            "task_id": task_id,
            "task_name": task_name,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"]
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nEvaluation Results:")
    print(results_df)
    
    return results_df

def visualize_confusion_matrices(confusion_matrices, task_names, labels_dict=None):
    """
    Visualize confusion matrices for multiple tasks
    
    Args:
        confusion_matrices: Dict of confusion matrices for each task
        task_names: List of task names
        labels_dict: Optional dict mapping task names to label names
    """
    num_tasks = len(confusion_matrices)
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    
    # Handle single task case
    if num_tasks == 1:
        axes = [axes]
    
    for i, task_name in enumerate(task_names):
        ax = axes[i]
        cm = confusion_matrices[task_name]
        
        # Get label names if provided
        if labels_dict and task_name in labels_dict:
            labels = labels_dict[task_name]
        else:
            labels = [str(i) for i in range(cm.shape[0])]
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{task_name} Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.close()

def compare_model_sizes(models_dict):
    """
    Compare the number of parameters in different models
    
    Args:
        models_dict: Dict of models to compare
        
    Returns:
        DataFrame of model sizes
    """
    sizes = []
    
    for name, model in models_dict.items():
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate parameter efficiency
        param_efficiency = trainable_params / total_params if total_params > 0 else 0
        
        sizes.append({
            "model": name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_efficiency": param_efficiency
        })
    
    # Convert to DataFrame
    sizes_df = pd.DataFrame(sizes)
    
    # Print results
    print("\nModel Size Comparison:")
    print(sizes_df)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(sizes_df))
    width = 0.35
    
    plt.bar(
        x - width/2, 
        sizes_df["total_params"] / 1_000_000, 
        width, 
        label="Total Parameters (M)"
    )
    plt.bar(
        x + width/2, 
        sizes_df["trainable_params"] / 1_000_000, 
        width, 
        label="Trainable Parameters (M)"
    )
    
    plt.xlabel("Model")
    plt.ylabel("Parameters (millions)")
    plt.title("Model Size Comparison")
    plt.xticks(x, sizes_df["model"])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("model_sizes.png")
    plt.close()
    
    return sizes_df

def plot_training_metrics(metrics_dict, save_path="training_metrics.png"):
    """
    Plot training metrics for different models
    
    Args:
        metrics_dict: Dict mapping model names to their training histories
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot training loss
    for model_name, metrics in metrics_dict.items():
        if "loss" in metrics:
            axes[0].plot(metrics["loss"], label=f"{model_name} Loss")
    
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot validation accuracy
    for model_name, metrics in metrics_dict.items():
        if "eval_accuracy" in metrics:
            axes[1].plot(metrics["eval_accuracy"], label=f"{model_name} Accuracy")
    
    axes[1].set_xlabel("Evaluation Steps")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_attention_patterns(model, tokenizer, example_texts, save_path="attention_patterns.png"):
    """
    Analyze and visualize attention patterns in the model
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer to use for processing
        example_texts: List of example texts to analyze
        save_path: Path to save the visualization
    """
    # Process examples
    inputs = tokenizer(
        example_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass with output_attentions=True
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention weights
    attention = outputs.attentions
    
    # Calculate number of attention heads and layers
    num_layers = len(attention)
    num_heads = attention[0].size(1)
    
    # Create visualization
    fig, axes = plt.subplots(
        num_layers, 
        num_heads, 
        figsize=(num_heads * 2, num_layers * 2)
    )
    
    # Handle single layer or head case
    if num_layers == 1:
        axes = [axes]
    if num_heads == 1:
        axes = [[ax] for ax in axes]
    
    # Plot attention patterns
    for layer in range(num_layers):
        for head in range(num_heads):
            # Get attention weights for first example
            attn_weights = attention[layer][0, head].cpu().numpy()
            
            # Plot as heatmap
            ax = axes[layer][head]
            im = ax.imshow(attn_weights, cmap="viridis")
            
            # Add titles
            if layer == 0:
                ax.set_title(f"Head {head}")
            if head == 0:
                ax.set_ylabel(f"Layer {layer}")
            
            # Remove ticks for clarity
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_cross_dataset_generalization(models_dict, tokenizer, datasets_dict):
    """
    Evaluate and compare model performance across different datasets
    
    Args:
        models_dict: Dict mapping model names to models
        tokenizer: Tokenizer to use for processing
        datasets_dict: Dict mapping dataset names to datasets
        
    Returns:
        DataFrame with cross-dataset evaluation results
    """
    # Initialize results structure
    results = []
    
    # Evaluate each model on each dataset
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        for dataset_name, dataset in datasets_dict.items():
            print(f"  On {dataset_name}...")
            
            # For multi-task model, use the appropriate task head
            task_id = None
            if "MultiTask" in model_name:
                # Assume task_id is contained in dataset_name (e.g., "sst2_validation")
                task_id = dataset_name.split("_")[0]
            
            # Evaluate
            metrics = evaluate_model_on_dataset(
                model, tokenizer, dataset, task_id=task_id
            )
            
            # Store results
            results.append({
                "model": model_name,
                "dataset": dataset_name,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"]
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nCross-Dataset Evaluation Results:")
    # Pivot for better visualization
    pivot_df = results_df.pivot(index="model", columns="dataset", values="accuracy")
    print(pivot_df)
    
    # Create heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="YlGnBu",
        fmt=".3f",
        linewidths=0.5
    )
    plt.title("Cross-Dataset Generalization (Accuracy)")
    plt.tight_layout()
    plt.savefig("cross_dataset_generalization.png")
    plt.close()
    
    return results_df

def compare_inference_speed(models_dict, tokenizer, example_texts, batch_sizes=[1, 4, 16], n_runs=5):
    """
    Compare inference speed of different models
    
    Args:
        models_dict: Dict mapping model names to models
        tokenizer: Tokenizer to use for processing
        example_texts: List of example texts to use for inference
        batch_sizes: List of batch sizes to test
        n_runs: Number of runs for each configuration
        
    Returns:
        DataFrame with inference speed results
    """
    # Initialize results structure
    results = []
    
    # Test each model
    for model_name, model in models_dict.items():
        print(f"\nMeasuring inference speed for {model_name}...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Test each batch size
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}...")
            
            # Create input batch by repeating examples
            n_examples = min(batch_size, len(example_texts))
            batch_texts = example_texts[:n_examples] * (batch_size // n_examples + 1)
            batch_texts = batch_texts[:batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Warm-up run
            with torch.no_grad():
                _ = model(**inputs)
            
            # Timed runs
            times = []
            for _ in range(n_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                # Start timer
                start_time.record()
                
                # Forward pass
                with torch.no_grad():
                    _ = model(**inputs)
                
                # End timer
                end_time.record()
                
                # Wait for GPU to finish
                torch.cuda.synchronize()
                
                # Calculate elapsed time in milliseconds
                elapsed_time = start_time.elapsed_time(end_time)
                times.append(elapsed_time)
            
            # Calculate average time
            avg_time = np.mean(times)
            
            # Store results
            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "avg_time_ms": avg_time,
                "examples_per_second": 1000 * batch_size / avg_time
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nInference Speed Comparison:")
    print(results_df)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    for model_name in models_dict.keys():
        model_results = results_df[results_df["model"] == model_name]
        plt.plot(
            model_results["batch_size"],
            model_results["examples_per_second"],
            marker="o",
            label=model_name
        )
    
    plt.xlabel("Batch Size")
    plt.ylabel("Examples per Second")
    plt.title("Inference Speed Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("inference_speed_comparison.png")
    plt.close()
    
    return results_df

def analyze_embeddings(model, tokenizer, texts, labels, method='pca', save_path="embeddings_visualization.png"):
    """
    Analyze and visualize embeddings from the model
    
    Args:
        model: Model to analyze
        tokenizer: Tokenizer to use for processing
        texts: List of texts to analyze
        labels: List of labels or categories for the texts
        method: Dimensionality reduction method ('pca' or 'tsne')
        save_path: Path to save the visualization
    """
    # Process texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    # Get embeddings from last layer's [CLS] token
    embeddings = hidden_states[-1][:, 0, :].cpu().numpy()
    
    # Reduce dimensionality for visualization
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Plot each label with a different color
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=str(label),
            alpha=0.7
        )
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Text Embeddings ({method.upper()})")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return reduced_embeddings, labels
