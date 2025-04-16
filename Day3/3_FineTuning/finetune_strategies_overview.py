"""
Fine-tuning Strategies Overview
-----------------------------
This file provides an overview of different fine-tuning strategies for LLMs,
with a focus on single-task vs. multi-task approaches and their trade-offs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def introduction_to_finetuning_strategies():
    """
    Introduction to different fine-tuning strategies for LLMs
    """
    print("=" * 60)
    print("FINE-TUNING STRATEGIES OVERVIEW")
    print("=" * 60)
    
    print("\nFine-tuning Approaches:")
    print("1. Single-Task Fine-Tuning")
    print("2. Multi-Task Fine-Tuning")
    print("3. Parameter-Efficient Fine-Tuning (PEFT)")
    print("4. Instruction Fine-Tuning")
    print("5. Reinforcement Learning from Human Feedback (RLHF)")
    
    print("\nKey Considerations:")
    print("- Data requirements for each approach")
    print("- Computational resources needed")
    print("- Performance on target tasks")
    print("- Generalization to unseen tasks")
    print("- Catastrophic forgetting risks")
    print("- Practical implementation challenges")

def explain_single_task_finetuning():
    """
    Explain single-task fine-tuning approach
    """
    print("\n" + "=" * 60)
    print("SINGLE-TASK FINE-TUNING")
    print("=" * 60)
    
    print("\nWhat is Single-Task Fine-Tuning?")
    print("Single-task fine-tuning adapts a pre-trained model specifically")
    print("for one task by training it on task-specific data.")
    
    print("\nMethodology:")
    print("1. Start with a pre-trained LLM")
    print("2. Prepare dataset specific to the target task")
    print("3. Fine-tune all or a subset of model parameters")
    print("4. Evaluate on task-specific metrics")
    
    print("\nAdvantages:")
    print("- Optimized performance on the specific task")
    print("- Simpler data preparation")
    print("- Easier to interpret results")
    print("- Less computational resources than multi-task")
    print("- Clear evaluation metrics")
    
    print("\nDisadvantages:")
    print("- Risk of catastrophic forgetting")
    print("- Limited generalization to other tasks")
    print("- Requires separate models for each task")
    print("- Poor sample efficiency (requires substantial task data)")
    
    print("\nBest Use Cases:")
    print("- When maximizing performance on a single task is critical")
    print("- When you have abundant task-specific data")
    print("- When computational resources for inference are limited")
    print("- When the task differs significantly from pre-training distribution")
    
    print("\nExample Implementation:")
    print("""
    # Single-Task Fine-Tuning Example
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    
    # Load pre-trained model
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load task-specific data (e.g., sentiment analysis)
    dataset = load_dataset("glue", "sst2")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    trainer.train()
    """)
    
    print("\nTips for Effective Single-Task Fine-Tuning:")
    print("1. Use an appropriate learning rate (usually 1e-5 to 5e-5)")
    print("2. Implement early stopping to prevent overfitting")
    print("3. Use task-specific metrics for model selection")
    print("4. Consider freezing some layers for very small datasets")
    print("5. Evaluate on a separate test set with diverse examples")
    print("6. Benchmark against task-specific SOTA models")

def explain_multi_task_finetuning():
    """
    Explain multi-task fine-tuning approach
    """
    print("\n" + "=" * 60)
    print("MULTI-TASK FINE-TUNING")
    print("=" * 60)
    
    print("\nWhat is Multi-Task Fine-Tuning?")
    print("Multi-task fine-tuning trains a single model on multiple tasks")
    print("simultaneously, allowing it to share knowledge across tasks and")
    print("potentially improve generalization.")
    
    print("\nMethodology:")
    print("1. Start with a pre-trained LLM")
    print("2. Prepare datasets from multiple related or diverse tasks")
    print("3. Train on all tasks simultaneously or in a curriculum")
    print("4. Evaluate on all target tasks individually")
    
    print("\nAdvantages:")
    print("- Better generalization to unseen tasks")
    print("- Improved sample efficiency")
    print("- Reduced catastrophic forgetting")
    print("- Single model serving multiple tasks")
    print("- Potential for positive transfer between tasks")
    
    print("\nDisadvantages:")
    print("- More complex training setup")
    print("- Requires balancing multiple tasks")
    print("- Risk of negative interference between tasks")
    print("- Higher computational requirements")
    print("- May not reach single-task performance on each task")
    
    print("\nBest Use Cases:")
    print("- When generalization to new tasks is important")
    print("- When data for individual tasks is limited")
    print("- When tasks share underlying knowledge or structure")
    print("- When maintaining general capabilities is critical")
    
    print("\nMulti-Task Training Approaches:")
    print("1. Joint Training: Train on all tasks simultaneously")
    print("2. Task Sampling: Sample tasks based on a strategy (e.g., proportional)")
    print("3. Curriculum Learning: Start with easier tasks, progress to harder ones")
    print("4. Meta-Learning: Learn to quickly adapt to new tasks")
    
    print("\nExample Implementation:")
    print("""
    # Multi-Task Fine-Tuning Example using T5
    from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
        Seq2SeqTrainer, 
        Seq2SeqTrainingArguments
    )
    from datasets import load_dataset, concatenate_datasets
    
    # Load pre-trained model
    model_name = "t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load multiple datasets
    summarization_data = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
    translation_data = load_dataset("wmt16", "ro-en", split="train[:1000]")
    qa_data = load_dataset("squad", split="train[:1000]")
    
    # Preprocess each dataset to add task prefixes
    def preprocess_summarization(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        targets = examples["highlights"]
        return {"input": inputs, "target": targets}
    
    def preprocess_translation(examples):
        inputs = ["translate English to Romanian: " + doc for doc in examples["en"]]
        targets = examples["ro"]
        return {"input": inputs, "target": targets}
    
    def preprocess_qa(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        targets = examples["answers"]["text"]  # Using first answer only
        return {"input": inputs, "target": targets}
    
    # Apply preprocessing
    summarization_processed = summarization_data.map(preprocess_summarization, batched=True)
    translation_processed = translation_data.map(preprocess_translation, batched=True)
    qa_processed = qa_data.map(preprocess_qa, batched=True)
    
    # Combine datasets
    combined_dataset = concatenate_datasets([
        summarization_processed, translation_processed, qa_processed
    ])
    
    # Tokenize function for seq2seq
    def tokenize_function(examples):
        model_inputs = tokenizer(examples["input"], max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize the combined dataset
    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)
    
    # Split into train/validation
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=128,
    )
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    trainer.train()
    """)
    
    print("\nTask Balancing Strategies:")
    task_balancing = [
        ["Strategy", "Description", "When to Use"],
        ["Uniform", "Sample equally from all tasks", "Similar sized, equally important tasks"],
        ["Proportional", "Sample according to dataset size", "Different sized datasets"],
        ["Temperature Sampling", "Use temperature parameter to control task distribution", "When balancing diversity and frequency"],
        ["Loss-based", "Sample more from tasks with higher loss", "When some tasks are harder to learn"],
        ["Static Mixing", "Fixed mixing ratios throughout training", "Simple implementation"],
        ["Dynamic Mixing", "Adapt mixing ratios during training", "Complex task relationships"]
    ]
    
    for row in task_balancing:
        print(" | ".join(f"{item}".ljust(26) for item in row))
    
    print("\nTips for Effective Multi-Task Fine-Tuning:")
    print("1. Choose related tasks for positive transfer")
    print("2. Balance task sampling to prevent dominant tasks")
    print("3. Use task-specific prefixes or tokens for clarity")
    print("4. Monitor per-task performance to detect negative transfer")
    print("5. Consider larger models for multiple diverse tasks")
    print("6. Use gradient accumulation for consistent updates")

def compare_single_vs_multi_task():
    """
    Compare single-task and multi-task fine-tuning approaches
    """
    print("\n" + "=" * 60)
    print("SINGLE-TASK VS. MULTI-TASK COMPARISON")
    print("=" * 60)
    
    print("\nPerformance Comparison:")
    performance = [
        ["Metric", "Single-Task", "Multi-Task"],
        ["Performance on target task", "Higher (specialized)", "Good (not always SOTA)"],
        ["Zero-shot performance", "Limited", "Better"],
        ["Few-shot performance", "Limited", "Better"],
        ["Generalization to new tasks", "Poor", "Good"],
        ["Retention of general knowledge", "Poor (forgetting)", "Better preserved"],
        ["Sample efficiency", "Lower", "Higher"]
    ]
    
    for row in performance:
        print(" | ".join(f"{item}".ljust(28) for item in row))
    
    print("\nResource Requirements:")
    resources = [
        ["Resource", "Single-Task", "Multi-Task"],
        ["Training compute", "Lower", "Higher"],
        ["Memory requirements", "Lower", "Higher"],
        ["Training time", "Shorter", "Longer"],
        ["Data preparation complexity", "Simple", "Complex"],
        ["Hyperparameter tuning", "Easier", "More difficult"],
        ["Storage (models)", "Multiple models", "Single model for multiple tasks"]
    ]
    
    for row in resources:
        print(" | ".join(f"{item}".ljust(26) for item in row))
    
    print("\nVisual Example (Resource Efficiency vs. Performance):")
    print("""
    # Plot comparing approaches (example code)
    import matplotlib.pyplot as plt
    
    # Data points
    approaches = ['No Fine-tuning', 'Single-Task', 'Multi-Task', 'Instruction-tuned']
    performance = [70, 90, 85, 83]  # Performance on primary task
    generalization = [70, 65, 80, 85]  # Generalization to new tasks
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(approaches))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], performance, width, label='Primary Task Performance')
    ax.bar([i + width/2 for i in x], generalization, width, label='Generalization')
    
    ax.set_xlabel('Fine-tuning Approach')
    ax.set_ylabel('Performance Score')
    ax.set_title('Performance vs. Generalization Trade-off')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('finetuning_comparison.png')
    plt.show()
    """)
    
    print("\nDecision Framework for Choosing Fine-tuning Strategy:")
    decision_tree = [
        "1. Is performance on a specific task the primary goal?",
        "   └─ Yes: Do you have sufficient task-specific data?",
        "      └─ Yes: Single-Task Fine-Tuning",
        "      └─ No: Consider Few-Shot Learning or PEFT",
        "   └─ No: Is generalization to new tasks important?",
        "      └─ Yes: Multi-Task Fine-Tuning",
        "      └─ No: Are resources limited?",
        "         └─ Yes: Parameter-Efficient Fine-Tuning",
        "         └─ No: Full Multi-Task Fine-Tuning"
    ]
    
    for line in decision_tree:
        print(line)
    
    print("\nHybrid Approaches:")
    print("1. Pre-train with multi-task, fine-tune with single-task")
    print("2. Use PEFT methods with multi-task objectives")
    print("3. Curriculum learning: multi-task then specialized")
    print("4. Task-specific adapters with shared base model")
    print("5. Instruction tuning followed by task-specific fine-tuning")
    
    print("\nIndustry Best Practices:")
    print("- Large models (>10B parameters): Start with instruction tuning")
    print("  then apply PEFT methods for specific tasks")
    print("- Medium models (1-10B parameters): Multi-task instruction tuning")
    print("  with task-specific adapters")
    print("- Small models (<1B parameters): Task-specific fine-tuning often")
    print("  outperforms multi-task approaches")

def explain_transfer_learning():
    """
    Explain transfer learning approaches for LLMs
    """
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING WITH LLMS")
    print("=" * 60)
    
    print("\nWhat is Transfer Learning?")
    print("Transfer learning leverages knowledge from one task to improve")
    print("performance on another task, typically by initializing with")
    print("weights from a model trained on a related task.")
    
    print("\nTransfer Learning in the LLM Context:")
    print("1. Pre-training → Fine-tuning: Standard LLM development")
    print("2. Domain Adaptation: Adapting to specific text domains")
    print("3. Task Transfer: Leveraging knowledge between related tasks")
    print("4. Cross-lingual Transfer: Transferring between languages")
    
    print("\nDomain Adaptation Approaches:")
    
    print("\n1. Continued Pre-training:")
    print("- Continue pre-training on domain-specific corpus")
    print("- Retains general knowledge while adding domain knowledge")
    print("- Example domains: legal, medical, scientific, financial")
    print("- Implementation:")
    print("""
    # Continued Pre-training Example
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    
    # Load pre-trained model
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load domain-specific data (e.g., biomedical papers)
    dataset = load_dataset("text", data_files={"train": "biomedical_papers.txt"})
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Training arguments (lighter than full pre-training)
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        warmup_steps=500,
        save_strategy="steps",
        save_steps=10000,
        logging_steps=500,
        fp16=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Continue pre-training
    trainer.train()
    """)
    
    print("\n2. Sequential Fine-tuning:")
    print("- Fine-tune on related high-resource task first")
    print("- Then fine-tune on target task with limited data")
    print("- Useful when target task has limited data")
    print("- Example: MNLI → RTE (entailment tasks)")
    
    print("\n3. Cross-lingual Transfer:")
    print("- Leverage multilingual pre-trained models (mBERT, XLM-R)")
    print("- Fine-tune on high-resource language, apply to low-resource")
    print("- Zero-shot or few-shot cross-lingual transfer")
    
    print("\nEffects of Transfer Distance:")
    transfer_distance = [
        ["Distance", "Example", "Expected Performance"],
        ["Very Close", "Movie reviews → Book reviews", "Excellent transfer"],
        ["Close", "News classification → Scientific paper classification", "Good transfer"],
        ["Moderate", "English sentiment → French sentiment", "Moderate transfer"],
        ["Far", "Image classification → Text generation", "Limited transfer"],
        ["Very Far", "Speech recognition → Protein folding", "Minimal transfer"]
    ]
    
    for row in transfer_distance:
        print(" | ".join(f"{item}".ljust(30) for item in row))
    
    print("\nTransfer Learning Patterns:")
    
    print("\n1. Positive Transfer:")
    print("- Knowledge from source task improves target task performance")
    print("- Common with related tasks and domains")
    print("- Example: Pre-training on Wikipedia improves most NLP tasks")
    
    print("\n2. Negative Transfer:")
    print("- Source task knowledge harms target task performance")
    print("- Often due to task conflict or domain mismatch")
    print("- Example: Translation knowledge can harm summarization")
    
    print("\n3. Zero Transfer:")
    print("- No significant benefit from source task")
    print("- Tasks too different or irrelevant to each other")
    print("- Example: Chess moves prediction → Weather forecasting")
    
    print("\nStrategies to Maximize Transfer:")
    
    print("\n1. Task Selection:")
    print("- Choose source tasks with structural similarity to target")
    print("- Leverage natural task hierarchies (e.g., general → specific)")
    print("- Consider sharing input/output formats")
    
    print("\n2. Layer-wise Transfer:")
    print("- Selectively transfer different layers based on task needs")
    print("- Lower layers: General features (often more transferable)")
    print("- Higher layers: Task-specific features (often less transferable)")
    print("- Implementation:")
    print("""
    # Layer-wise Transfer Example
    from transformers import AutoModel, AutoModelForSequenceClassification
    
    # Load pre-trained model
    source_model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Initialize target model (e.g., for a classification task)
    target_model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3  # Target task has 3 classes
    )
    
    # Selective transfer: Copy weights from lower layers only
    # For example, transfer only the first 6 encoder layers
    for i in range(6):
        target_layer = target_model.bert.encoder.layer[i]
        source_layer = source_model.encoder.layer[i]
        target_layer.load_state_dict(source_layer.state_dict())
    
    # Freeze transferred layers to prevent catastrophic forgetting
    for i in range(6):
        for param in target_model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
            
    # Fine-tune only the remaining layers
    # Rest of training code as usual...
    """)
    
    print("\n3. Progressive Transfer:")
    print("- Start with fully frozen model except output layer")
    print("- Gradually unfreeze more layers during training")
    print("- Often combined with decreasing learning rates")
    print("- Similar to ULMFiT approach")
    
    print("\nEvaluating Transfer Success:")
    print("1. Performance Gap: Compare to random initialization")
    print("2. Transfer Efficiency: Compare data requirements")
    print("3. Convergence Speed: Compare training time to convergence")
    print("4. Generalization: Compare performance on unseen examples")
    
    print("\nBest Practices for Transfer Learning:")
    print("1. Prefer larger models (more parameters = better transfer)")
    print("2. Start with similar source and target domains")
    print("3. Use appropriate learning rates (smaller for target)")
    print("4. Consider freezing early layers and fine-tuning later layers")
    print("5. Monitor for signs of negative transfer")
    print("6. Use learning rate warmup and gradual unfreezing")

def explain_distilbert_finetuning():
    """
    Provide details specifically for DistilBERT fine-tuning exercise
    """
    print("\n" + "=" * 60)
    print("DISTILBERT FINE-TUNING EXERCISE")
    print("=" * 60)
    
    print("\nAbout DistilBERT:")
    print("DistilBERT is a smaller, faster version of BERT created through")
    print("knowledge distillation. It retains 97% of BERT's performance while")
    print("being 40% smaller and 60% faster.")
    
    print("\nKey DistilBERT Facts:")
    print("- 6 layers (vs. 12 in BERT-base)")
    print("- 768 hidden dimension (same as BERT-base)")
    print("- ~66M parameters (vs. ~110M in BERT-base)")
    print("- Pre-trained on same corpus as BERT")
    print("- Created by Hugging Face to improve efficiency")
    
    print("\nSingle-Task DistilBERT Fine-tuning:")
    print("We'll fine-tune DistilBERT on a sentiment analysis task to establish")
    print("strong single-task performance baseline.")
    
    print("\nImplementation:")
    print("""
    # Single-Task Fine-tuning of DistilBERT for Sentiment Analysis
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    # Load DistilBERT model and tokenizer
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary sentiment (positive/negative)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset (SST-2 for sentiment analysis)
    dataset = load_dataset("glue", "sst2")
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Define metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_distilbert_sentiment",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Save the model
    model_path = "./distilbert_sentiment"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    """)
    
    print("\nMulti-Task DistilBERT Fine-tuning:")
    print("Next, we'll implement multi-task fine-tuning on DistilBERT across")
    print("several text classification tasks to compare with single-task performance.")
    
    print("\nImplementation:")
    print("""
    # Multi-Task Fine-tuning of DistilBERT
    from transformers import (
        AutoModel,
        AutoTokenizer,
        Trainer,
        TrainingArguments
    )
    from datasets import load_dataset, concatenate_datasets
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    # Define a multi-task model using DistilBERT backbone
    class DistilBERTMultiTaskModel(nn.Module):
        def __init__(self, num_labels_dict):
            super().__init__()
            # Load DistilBERT as shared encoder
            self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
            self.dropout = nn.Dropout(0.1)
            
            # Task-specific output heads
            self.task_heads = nn.ModuleDict()
            for task_name, num_labels in num_labels_dict.items():
                self.task_heads[task_name] = nn.Linear(768, num_labels)
        
        def forward(self, task_name, **kwargs):
            # Get encoder outputs
            outputs = self.encoder(**kwargs)
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]  # CLS token
            pooled_output = self.dropout(pooled_output)
            
            # Feed through task-specific head
            logits = self.task_heads[task_name](pooled_output)
            
            # Calculate loss if labels provided
            loss = None
            if "labels" in kwargs:
                if task_name in ["sst2", "cola"]:  # Binary/single-label tasks
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), kwargs["labels"].view(-1))
                else:  # Multi-label tasks
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, kwargs["labels"])
            
            return {"loss": loss, "logits": logits}
    
    # Load datasets for multiple tasks
    tasks = {
        "sst2": {"num_labels": 2, "dataset": "glue", "subset": "sst2"},
        "cola": {"num_labels": 2, "dataset": "glue", "subset": "cola"},
        "mnli": {"num_labels": 3, "dataset": "glue", "subset": "mnli"}
    }
    
    datasets = {}
    for task_name, task_info in tasks.items():
        datasets[task_name] = load_dataset(task_info["dataset"], task_info["subset"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize function for all tasks
    def tokenize_function(examples, task_name):
        # Handle different field names in different datasets
        text_field = "sentence" if task_name in ["sst2", "cola"] else "premise"
        second_text_field = None if task_name in ["sst2", "cola"] else "hypothesis"
        
        if second_text_field:
            return tokenizer(
                examples[text_field],
                examples[second_text_field],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        else:
            return tokenizer(
                examples[text_field],
                padding="max_length",
                truncation=True,
                max_length=128
            )
    
    # Process datasets
    tokenized_datasets = {}
    for task_name, dataset in datasets.items():
        tokenize_task = lambda examples: tokenize_function(examples, task_name)
        tokenized_task = dataset.map(tokenize_task, batched=True)
        # Add task name to each example
        tokenized_task = tokenized_task.map(lambda x: {"task": task_name})
        tokenized_datasets[task_name] = tokenized_task
    
    # Create model with task-specific heads
    num_labels_dict = {task_name: info["num_labels"] for task_name, info in tasks.items()}
    model = DistilBERTMultiTaskModel(num_labels_dict)
    
    # Custom data collator for multi-task batching
    class MultiTaskDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def __call__(self, examples):
            # All examples in a batch must be from the same task
            task_name = examples[0]["task"]
            
            # Remove task field before tokenization
            for example in examples:
                del example["task"]
            
            # Use default data collator for the rest
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                padding="longest"
            )
            
            # Convert labels
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            
            # Add task name back to batch
            batch["task_name"] = task_name
            
            return batch
    
    # Custom Trainer for multi-task
    class MultiTaskTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            task_name = inputs.pop("task_name")
            outputs = model(task_name, **inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            task_name = inputs.pop("task_name")
            with torch.no_grad():
                outputs = model(task_name, **inputs)
            
            loss = outputs["loss"].detach() if outputs["loss"] is not None else None
            logits = outputs["logits"].detach() if outputs["logits"] is not None else None
            labels = inputs.get("labels", None)
            
            return loss, logits, labels
    
    # Combine datasets for training
    train_datasets = []
    eval_datasets = []
    
    for task_name, tokenized_dataset in tokenized_datasets.items():
        train_datasets.append(tokenized_dataset["train"])
        eval_datasets.append(tokenized_dataset["validation"])
    
    # Simple concatenation (will use task sampling in dataloader)
    combined_train_dataset = concatenate_datasets(train_datasets)
    combined_eval_dataset = concatenate_datasets(eval_datasets)
    
    # Define compute metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_distilbert_multitask",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    
    # Initialize multi-task trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_eval_dataset,
        tokenizer=tokenizer,
        data_collator=MultiTaskDataCollator(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Fine-tune the model
    trainer.train()
    
    # Evaluate on validation sets for each task
    for task_name, tokenized_dataset in tokenized_datasets.items():
        print(f"\\nEvaluating on {task_name}:")
        eval_dataset = tokenized_dataset["validation"]
        # Add task name to all examples
        eval_dataset = eval_dataset.map(lambda x: {"task": task_name})
        results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Task: {task_name}, Accuracy: {results['eval_accuracy']:.4f}, F1: {results['eval_f1']:.4f}")
    
    # Save the model
    model_path = "./distilbert_multitask"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    """)
    
    print("\nComparing Single-Task vs. Multi-Task Performance:")
    print("After implementing both approaches, we'll evaluate and compare")
    print("their performance across various metrics to understand the trade-offs.")
    
    print("\nEvaluation Implementation:")
    print("""
    # Evaluation Code for Comparing Single-Task vs. Multi-Task Models
    import torch
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer,
        pipeline
    )
    from datasets import load_dataset
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    # Load single-task model
    single_task_model = AutoModelForSequenceClassification.from_pretrained("./distilbert_sentiment")
    single_task_tokenizer = AutoTokenizer.from_pretrained("./distilbert_sentiment")
    
    # Create sentimen analysis pipeline
    sentiment_pipeline_single = pipeline("sentiment-analysis", model=single_task_model, tokenizer=single_task_tokenizer)
    
    # Load multi-task model
    # Note: Custom model would need custom loading logic
    multitask_model = torch.load("./distilbert_multitask/pytorch_model.bin")
    multitask_tokenizer = AutoTokenizer.from_pretrained("./distilbert_multitask")
    
    # Load test datasets
    test_datasets = {
        "sst2": load_dataset("glue", "sst2", split="validation"),
        "imdb": load_dataset("imdb", split="test[:100]")  # Out-of-distribution test
    }
    
    # Evaluate on test datasets
    results = {
        "single_task": {},
        "multi_task": {}
    }
    
    for dataset_name, dataset in test_datasets.items():
        print(f"\\nEvaluating on {dataset_name} dataset:")
        
        # Extract texts and labels
        texts = dataset["sentence"] if "sentence" in dataset.features else dataset["text"]
        labels = dataset["label"]
        
        # Single-task model predictions
        single_task_preds = sentiment_pipeline_single(list(texts))
        single_task_pred_labels = [1 if pred["label"] == "POSITIVE" else 0 for pred in single_task_preds]
        
        # Calculate metrics
        single_task_acc = accuracy_score(labels, single_task_pred_labels)
        single_task_f1 = f1_score(labels, single_task_pred_labels, average='weighted')
        
        results["single_task"][dataset_name] = {
            "accuracy": single_task_acc,
            "f1": single_task_f1
        }
        
        print(f"Single-Task Model: Accuracy = {single_task_acc:.4f}, F1 = {single_task_f1:.4f}")
        
        # Multi-task model predictions
        # Note: This is a simplified example; actual implementation would depend on your multi-task model
        # multi_task_preds = multitask_model.predict("sst2", list(texts))  # Assuming your model has a predict method
        # multi_task_pred_labels = multi_task_preds.argmax(-1)
        # 
        # multi_task_acc = accuracy_score(labels, multi_task_pred_labels)
        # multi_task_f1 = f1_score(labels, multi_task_pred_labels, average='weighted')
        # 
        # results["multi_task"][dataset_name] = {
        #     "accuracy": multi_task_acc,
        #     "f1": multi_task_f1
        # }
        # 
        # print(f"Multi-Task Model: Accuracy = {multi_task_acc:.4f}, F1 = {multi_task_f1:.4f}")
    
    # Visualize results
    datasets = list(results["single_task"].keys())
    single_task_acc = [results["single_task"][d]["accuracy"] for d in datasets]
    # multi_task_acc = [results["multi_task"][d]["accuracy"] for d in datasets]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, single_task_acc, width, label='Single-Task')
    # plt.bar(x + width/2, multi_task_acc, width, label='Multi-Task')
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Single-Task vs. Multi-Task Performance')
    plt.xticks(x, datasets)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('single_vs_multi_performance.png')
    plt.show()
    """)
    
    print("\nExercise Tasks:")
    print("1. Implement both single-task and multi-task fine-tuning for DistilBERT")
    print("2. Evaluate models on in-domain and out-of-domain datasets")
    print("3. Compare performance, resource usage, and generalization ability")
    print("4. Implement a hybrid approach using adapter modules")
    print("5. Analyze catastrophic forgetting across approaches")
    print("6. Suggest best practices based on your findings")

def main():
    """
    Main function to run the fine-tuning strategies overview
    """
    introduction_to_finetuning_strategies()
    explain_single_task_finetuning()
    explain_multi_task_finetuning()
    compare_single_vs_multi_task()
    explain_transfer_learning()
    explain_distilbert_finetuning()
    
    print("\n" + "=" * 60)
    print("FINE-TUNING STRATEGIES OVERVIEW COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
