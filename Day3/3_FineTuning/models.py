"""
Models for DistilBERT Fine-tuning
------------------------------
This file contains model definitions for the DistilBERT fine-tuning exercise,
including the multi-task model architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class DistilBERTMultiTaskModel(nn.Module):
    """
    Multi-task model based on DistilBERT with task-specific heads
    """
    def __init__(self, tasks_dict):
        super().__init__()
        # Load DistilBERT as shared encoder
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_id, task_info in tasks_dict.items():
            self.task_heads[task_id] = nn.Linear(768, task_info["num_labels"])
    
    def forward(self, input_ids, attention_mask, task_id=None, labels=None):
        """
        Forward pass of the multi-task model
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            task_id: ID of the task being performed
            labels: Optional labels for loss calculation
            
        Returns:
            Dict containing loss and logits
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # CLS token embedding
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions from task-specific head
        logits = self.task_heads[task_id](pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

class DistilBERTWithAdapter(nn.Module):
    """
    DistilBERT model with adapter modules for parameter-efficient fine-tuning
    
    Adapters are small bottleneck modules added after each transformer layer
    that allow for efficient adaptation to new tasks while freezing most of
    the pre-trained model parameters.
    """
    def __init__(self, num_labels, adapter_size=64):
        super().__init__()
        # Load pre-trained model
        self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Freeze all parameters
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Add adapter modules to each layer
        self.adapters = nn.ModuleList()
        for i in range(6):  # DistilBERT has 6 layers
            self.adapters.append(Adapter(768, adapter_size))
        
        # Task-specific head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with adapters
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional labels for loss calculation
            
        Returns:
            Dict containing loss and logits
        """
        # Get outputs from each layer of DistilBERT
        hidden_states = self.distilbert.embeddings(input_ids)
        all_hidden_states = ()
        
        # Process through each transformer layer with adapters
        for i, layer in enumerate(self.distilbert.transformer.layer):
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            # Apply adapter to layer output
            hidden_states = self.adapters[i](hidden_states)
            
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Get final representation
        sequence_output = hidden_states
        pooled_output = sequence_output[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from task-specific head
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

class Adapter(nn.Module):
    """
    Adapter module for parameter-efficient fine-tuning
    
    The adapter consists of a down-projection, followed by a non-linearity,
    followed by an up-projection. A residual connection is used to maintain
    the original information flow.
    """
    def __init__(self, input_size, adapter_size):
        super().__init__()
        self.down_proj = nn.Linear(input_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.act_fn = nn.GELU()
        
        # Initialize with small weights
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.bias.data.zero_()
    
    def forward(self, hidden_states):
        """
        Forward pass through adapter
        
        Args:
            hidden_states: Hidden states from transformer layer
            
        Returns:
            Adapted hidden states
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Down projection
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        
        # Up projection
        hidden_states = self.up_proj(hidden_states)
        
        # Add residual connection
        output = hidden_states + residual
        
        return output

class DistilBERTPromptTuning(nn.Module):
    """
    DistilBERT model with prompt tuning
    
    Prompt tuning adds trainable "soft prompts" (continuous vector representations)
    to the input, allowing for task adaptation with minimal parameter updates.
    """
    def __init__(self, num_labels, num_virtual_tokens=20):
        super().__init__()
        # Load pre-trained model
        self.distilbert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Freeze all parameters
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # Get embedding dimension from model
        self.embedding_dim = self.distilbert.embeddings.word_embeddings.weight.shape[1]
        
        # Initialize trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(num_virtual_tokens, self.embedding_dim)
        )
        
        # Initialize with random embeddings (better than zeros)
        self._init_prompt_embeddings()
        
        # Task-specific head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.embedding_dim, num_labels)
    
    def _init_prompt_embeddings(self):
        """Initialize prompt embeddings with values from random real token embeddings"""
        word_embeddings = self.distilbert.embeddings.word_embeddings.weight.data
        vocab_size = word_embeddings.shape[0]
        
        # Sample random indices
        random_indices = torch.randint(0, vocab_size, (self.prompt_embeddings.shape[0],))
        
        # Copy embeddings from selected tokens
        sampled_embeddings = word_embeddings[random_indices].clone()
        
        # Add small noise for better training dynamics
        noise = torch.randn_like(sampled_embeddings) * 0.01
        self.prompt_embeddings.data = sampled_embeddings + noise
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with prompt tuning
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional labels for loss calculation
            
        Returns:
            Dict containing loss and logits
        """
        batch_size = input_ids.shape[0]
        
        # Get token embeddings from input IDs
        inputs_embeds = self.distilbert.embeddings.word_embeddings(input_ids)
        
        # Create prompt embeddings for batch
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend prompt embeddings to inputs
        combined_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # Adjust attention mask for prompts (all 1s for prompt tokens)
        prompt_attention_mask = torch.ones(
            batch_size, self.prompt_embeddings.shape[0], 
            device=attention_mask.device
        )
        combined_attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask], dim=1
        )
        
        # Forward pass through model with embeddings directly
        outputs = self.distilbert(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask
        )
        
        # Get sequence output and classify
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        
        # Get logits from task-specific head
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {"loss": loss, "logits": logits}
