"""
Trainers for DistilBERT Fine-tuning
-------------------------------
This file contains custom trainer classes and data collators for the
DistilBERT fine-tuning exercise, particularly for multi-task learning.
"""

import torch
import numpy as np
from transformers import Trainer, DataCollator
from typing import Dict, List, Optional, Any, Union, Tuple

class MultiTaskDataCollator:
    """
    Custom data collator for multi-task batching
    
    This collator ensures that all examples in a batch belong to the same task
    and handles the task_id field appropriately.
    """
    def __init__(self, tokenizer):
        """
        Initialize the data collator
        
        Args:
            tokenizer: Tokenizer to use for padding
        """
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        """
        Process a batch of examples
        
        Args:
            features: List of examples to batch
            
        Returns:
            Batch dictionary with task_id
        """
        # Extract task ID (should be same for all examples in batch)
        task_id = features[0]["task_id"]
        
        # Get batch inputs
        batch = {
            key: [feature[key] for feature in features if key != "task_id"] 
            for key in features[0].keys() if key != "task_id"
        }
        
        # Pad inputs
        batch = self.tokenizer.pad(
            batch,
            padding="longest",
            max_length=128,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Add task ID to batch
        batch["task_id"] = task_id
        
        return batch

class TaskSampler:
    """
    Sampler for multi-task learning that balances tasks
    
    This sampler can use different strategies for sampling tasks:
    - Uniform: Sample tasks with equal probability
    - Proportional: Sample tasks in proportion to their dataset sizes
    - Temperature: Sample tasks with a temperature parameter
    """
    def __init__(self, task_datasets, strategy="proportional", temperature=1.0):
        """
        Initialize the task sampler
        
        Args:
            task_datasets: Dict of datasets for each task
            strategy: Sampling strategy ("uniform", "proportional", "temperature")
            temperature: Temperature parameter for temperature sampling
        """
        self.task_datasets = task_datasets
        self.strategy = strategy
        self.temperature = temperature
        
        # Get dataset sizes
        self.dataset_sizes = {task_id: len(dataset) for task_id, dataset in task_datasets.items()}
        self.tasks = list(task_datasets.keys())
        
        # Calculate sampling probabilities
        self._calculate_sampling_probabilities()
    
    def _calculate_sampling_probabilities(self):
        """Calculate task sampling probabilities based on strategy"""
        if self.strategy == "uniform":
            # Equal probability for all tasks
            self.sampling_probs = np.ones(len(self.tasks)) / len(self.tasks)
        
        elif self.strategy == "proportional":
            # Proportional to dataset sizes
            total_examples = sum(self.dataset_sizes.values())
            self.sampling_probs = np.array([
                self.dataset_sizes[task_id] / total_examples for task_id in self.tasks
            ])
        
        elif self.strategy == "temperature":
            # Apply temperature to proportional sampling
            total_examples = sum(self.dataset_sizes.values())
            raw_probs = np.array([
                self.dataset_sizes[task_id] / total_examples for task_id in self.tasks
            ])
            
            # Apply temperature scaling
            scaled_probs = raw_probs ** (1 / self.temperature)
            self.sampling_probs = scaled_probs / scaled_probs.sum()
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def sample_task(self):
        """
        Sample a task based on the sampling strategy
        
        Returns:
            task_id: ID of the sampled task
        """
        return np.random.choice(self.tasks, p=self.sampling_probs)
    
    def sample_batch_indices(self, task_id, batch_size):
        """
        Sample indices for a batch from a specific task
        
        Args:
            task_id: ID of the task to sample from
            batch_size: Size of the batch to sample
            
        Returns:
            indices: List of indices for the batch
        """
        dataset_size = self.dataset_sizes[task_id]
        
        # Sample with replacement if dataset is smaller than batch size
        replacement = dataset_size < batch_size
        
        indices = np.random.choice(
            dataset_size, 
            size=min(batch_size, dataset_size), 
            replace=replacement
        )
        
        return indices

class MultiTaskTrainer(Trainer):
    """
    Custom trainer for multi-task learning
    
    This trainer handles batches with task_id and can implement different
    task sampling strategies for balanced training.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the trainer with standard Trainer arguments"""
        super().__init__(*args, **kwargs)
        
        # Task sampling configuration can be added here
        self.task_sampling_strategy = "proportional"  # Default strategy
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for a multi-task batch
        
        Args:
            model: Model to compute loss for
            inputs: Batch inputs, including task_id
            return_outputs: Whether to return outputs along with the loss
            
        Returns:
            loss or (loss, outputs)
        """
        # Extract task_id
        task_id = inputs.pop("task_id")
        
        # Forward pass
        outputs = model(task_id=task_id, **inputs)
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        """
        Create a training dataloader
        
        This implementation uses the default dataloader for simplicity.
        For more advanced task balancing, you can implement custom logic here.
        
        Returns:
            DataLoader for training
        """
        # Get default dataloader
        dataloader = super().get_train_dataloader()
        
        # Return the dataloader as is for now
        # A more sophisticated implementation would use TaskSampler
        return dataloader
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform prediction for a multi-task batch
        
        Args:
            model: Model to use for prediction
            inputs: Batch inputs, including task_id
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in the outputs
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        # Extract task_id
        task_id = inputs.pop("task_id")
        
        # Move inputs to appropriate device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        
        # Perform prediction
        with torch.no_grad():
            outputs = model(task_id=task_id, **inputs)
        
        # Get loss, logits, and labels
        loss = outputs["loss"].detach() if outputs["loss"] is not None else None
        logits = outputs["logits"].detach() if outputs["logits"] is not None else None
        labels = inputs.get("labels", None)
        
        return (loss, logits, labels)

class CurriculumLearningTrainer(MultiTaskTrainer):
    """
    Trainer that implements curriculum learning for multi-task fine-tuning
    
    Curriculum learning starts with easier tasks and gradually introduces
    harder ones during training.
    """
    def __init__(self, *args, task_difficulty=None, curriculum_epochs=None, **kwargs):
        """
        Initialize the curriculum learning trainer
        
        Args:
            task_difficulty: Dict mapping task IDs to difficulty scores
            curriculum_epochs: List of epoch cutoffs for introducing new tasks
            *args, **kwargs: Standard Trainer arguments
        """
        super().__init__(*args, **kwargs)
        
        # Set up curriculum learning parameters
        self.task_difficulty = task_difficulty or {}
        self.curriculum_epochs = curriculum_epochs or []
        
        # Sort tasks by difficulty
        if task_difficulty:
            self.sorted_tasks = sorted(
                task_difficulty.keys(), 
                key=lambda x: task_difficulty[x]
            )
        else:
            self.sorted_tasks = []
        
        # Track current epoch
        self.current_epoch = 0
    
    def get_active_tasks(self):
        """
        Get tasks that are active in the current epoch based on curriculum
        
        Returns:
            List of active task IDs
        """
        if not self.sorted_tasks or not self.curriculum_epochs:
            # If no curriculum defined, all tasks are active
            return list(self.model.task_heads.keys())
        
        # Determine how many tasks to include based on current epoch
        num_tasks = 1  # Start with at least one task
        for i, epoch_cutoff in enumerate(self.curriculum_epochs):
            if self.current_epoch >= epoch_cutoff:
                num_tasks = i + 2  # +2 because we start with 1 task
        
        # Return the N easiest tasks
        return self.sorted_tasks[:min(num_tasks, len(self.sorted_tasks))]
    
    def training_step(self, model, inputs):
        """
        Perform a training step with curriculum learning
        
        Args:
            model: Model to train
            inputs: Batch inputs, including task_id
            
        Returns:
            Loss value
        """
        # Get active tasks for current epoch
        active_tasks = self.get_active_tasks()
        
        # Skip batches for tasks that aren't active yet
        task_id = inputs.get("task_id")
        if task_id not in active_tasks:
            # Return zero loss for inactive tasks
            return torch.tensor(0.0, device=self.args.device)
        
        # Process normally for active tasks
        return super().training_step(model, inputs)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Handle end of epoch - update curriculum
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
        """
        # Call parent method first
        super().on_epoch_end(args, state, control, **kwargs)
        
        # Update current epoch
        self.current_epoch += 1
        
        # Log active tasks for next epoch
        active_tasks = self.get_active_tasks()
        self.log({"active_tasks": active_tasks})
