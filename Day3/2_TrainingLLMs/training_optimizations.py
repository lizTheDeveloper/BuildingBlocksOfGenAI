"""
LLM Training Optimizations
------------------------
This file demonstrates various optimization techniques for training LLMs,
including mixed precision training, gradient checkpointing, and optimization
methods to reduce memory usage and improve training efficiency.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments
)
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def introduction_to_optimizations():
    """
    Introduction to various training optimizations for LLMs
    """
    print("=" * 60)
    print("LLM TRAINING OPTIMIZATIONS")
    print("=" * 60)
    
    print("\nKey Optimization Techniques:")
    print("1. Mixed Precision Training")
    print("2. Gradient Checkpointing")
    print("3. Optimizer State Reduction")
    print("4. Memory-Efficient Attention")
    print("5. Data Parallelism")
    print("6. Parameter-Efficient Fine-Tuning (PEFT)")
    print("7. Quantization-Aware Training")
    
    print("\nBenefits of Optimizations:")
    print("- Train larger models on limited hardware")
    print("- Reduce training time and cost")
    print("- Enable training of models that would otherwise be infeasible")
    print("- Improve training stability and convergence")
    print("- Reduce memory fragmentation and OOM errors")

def explain_mixed_precision():
    """
    Explain mixed precision training and its benefits
    """
    print("\n" + "=" * 60)
    print("MIXED PRECISION TRAINING")
    print("=" * 60)
    
    print("\nWhat is Mixed Precision Training?")
    print("Mixed precision training uses lower-precision formats (FP16 or BF16)")
    print("for most operations while keeping master weights in FP32.")
    
    print("\nKey Components:")
    print("1. Forward & backward passes in FP16/BF16")
    print("2. Master weights stored in FP32")
    print("3. Loss scaling to prevent underflow")
    print("4. Dynamic loss scaling adjustment")
    
    print("\nBenefits:")
    print("- Memory reduction: ~2x less memory usage")
    print("- Performance boost: 2-3x faster on GPUs with Tensor Cores")
    print("- Enables larger batch sizes")
    print("- Allows larger models on the same hardware")
    
    print("\nFP16 vs BF16:")
    data = [
        ["Precision", "Exponent Bits", "Mantissa Bits", "Range", "Precision"],
        ["FP32", "8", "23", "Large", "High"],
        ["FP16", "5", "10", "Limited", "Lower"],
        ["BF16", "8", "7", "Same as FP32", "Lower than FP16"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(12) for item in row))
    
    print("\nTrade-offs:")
    print("- FP16: Better precision but limited range (underflow/overflow risk)")
    print("- BF16: Same range as FP32 but less precision (TPU-friendly)")
    print("- Modern practice: BF16 > FP16 for most LLM training")
    
    print("\nImplementation with PyTorch:")
    print("""
    # Using torch.cuda.amp
    from torch.cuda.amp import autocast, GradScaler
    
    # Create gradient scaler for FP16
    scaler = GradScaler()
    
    for batch in dataloader:
        # Forward pass in mixed precision
        with autocast(dtype=torch.float16):  # or torch.bfloat16
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        # Unscale before optimizer step
        scaler.unscale_(optimizer)
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update with scaling
        scaler.step(optimizer)
        scaler.update()
    """)
    
    print("\nImplementation with Hugging Face Transformers:")
    print("""
    # Using TrainingArguments
    training_args = TrainingArguments(
        output_dir="./output",
        fp16=True,  # Enable FP16 training
        # Or alternatively:
        # bf16=True,  # Enable BF16 training (if hardware supports it)
        ...
    )
    
    # Or with native PyTorch 2.0:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Model forward and backward
    """)
    
    print("\nWhen Not to Use Mixed Precision:")
    print("- When working with very small numbers (risk of underflow)")
    print("- On hardware without native acceleration for FP16/BF16")
    print("- For numerically unstable models or operations")

def explain_gradient_checkpointing():
    """
    Explain gradient checkpointing and its benefits
    """
    print("\n" + "=" * 60)
    print("GRADIENT CHECKPOINTING")
    print("=" * 60)
    
    print("\nWhat is Gradient Checkpointing?")
    print("A technique that trades computation for memory by discarding")
    print("intermediate activations during the forward pass and recomputing")
    print("them during the backward pass.")
    
    print("\nHow It Works:")
    print("1. Standard backprop: Store all activations from forward pass")
    print("2. With checkpointing: Store only selected activations")
    print("3. During backward pass: Recompute the missing activations")
    
    print("\nMemory-Computation Trade-off:")
    print("- Memory savings: Up to 5-10x reduction in activation memory")
    print("- Computational cost: ~20-30% increase in training time")
    
    print("\nBest Use Cases:")
    print("- When model size is limited by GPU memory")
    print("- For very deep models with many layers")
    print("- When you need larger batch sizes")
    print("- Combined with mixed precision for maximum memory savings")
    
    print("\nImplementation with PyTorch:")
    print("""
    # Define a simple model with checkpointing
    import torch.utils.checkpoint as checkpoint
    
    class CheckpointedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(24)])
            self.use_checkpointing = True
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                if self.use_checkpointing:
                    x = checkpoint.checkpoint(layer, x)
                else:
                    x = layer(x)
            return x
    """)
    
    print("\nImplementation with Hugging Face Transformers:")
    print("""
    # Enable gradient checkpointing with Transformers models
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.gradient_checkpointing_enable()
    
    # Then use the model normally
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    """)
    
    print("\nCheckpointing Granularity:")
    print("- Fine-grained: Checkpoint individual operations (maximum memory saving)")
    print("- Medium-grained: Checkpoint blocks/layers (balanced approach)")
    print("- Coarse-grained: Checkpoint large sections (minimal overhead)")
    
    print("\nAdvanced Techniques:")
    print("- Selective checkpointing: Apply only to memory-intensive layers")
    print("- Adaptive checkpointing: Change strategy based on batch size")
    print("- Activation recomputation schedules: Optimize recomputation order")
    
    print("\nBenchmark (approximate, model-dependent):")
    data = [
        ["Checkpointing", "Memory Usage", "Training Speed", "Max Batch Size"],
        ["None", "100%", "100%", "1x"],
        ["Coarse", "~50%", "~90%", "~2x"],
        ["Medium", "~30%", "~80%", "~3x"],
        ["Fine", "~15%", "~70%", "~6x"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(15) for item in row))

def explain_optimizer_state_reduction():
    """
    Explain techniques to reduce optimizer state memory usage
    """
    print("\n" + "=" * 60)
    print("OPTIMIZER STATE REDUCTION")
    print("=" * 60)
    
    print("\nThe Optimizer Memory Problem:")
    print("- Adam/AdamW stores 2 additional tensors (m, v) per parameter")
    print("- Optimizer states can consume 2x or more memory than model parameters")
    print("- For 7B parameter model: ~28GB for Adam states alone (FP32)")
    
    print("\nSolutions for Optimizer State Reduction:")
    
    print("\n1. 8-bit Optimizers:")
    print("- Quantize optimizer states to 8-bit")
    print("- Memory savings: ~4x for optimizer states")
    print("- Implementation:")
    print("""
    # Using bitsandbytes library
    import bitsandbytes as bnb
    
    # 8-bit AdamW optimizer
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), 
        lr=2e-5,
        betas=(0.9, 0.999)
    )
    """)
    
    print("\n2. Optimizer Parameter Sharing/Factorization:")
    print("- LOMO (LOw-Memory Optimization)")
    print("- Memory savings: ~3-4x for optimizer states")
    print("- Adaptation of optimizers using matrix factorization")
    
    print("\n3. Memory-Efficient Optimizers:")
    print("- Adafactor: Factorized version of Adam")
    print("- Lion: Less memory-intensive optimizer")
    print("- Implementation:")
    print("""
    # Using Adafactor from Transformers
    from transformers.optimization import Adafactor
    
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        relative_step=False,
        scale_parameter=False
    )
    """)
    
    print("\n4. Distributed Optimizer States:")
    print("- ZeRO optimizer (DeepSpeed)")
    print("- Shards optimizer states across multiple GPUs")
    print("- Implementation:")
    print("""
    # Using DeepSpeed ZeRO
    import deepspeed
    
    # Define DeepSpeed config
    ds_config = {
        "zero_optimization": {
            "stage": 2,  # Shard optimizer states and gradients
            "offload_optimizer": {
                "device": "cpu"  # Optionally offload to CPU
            }
        },
        "fp16": {"enabled": True}
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    """)
    
    print("\nComparison of Methods:")
    data = [
        ["Method", "Memory Reduction", "Performance Impact", "Setup Complexity"],
        ["Standard Adam (FP32)", "1x (baseline)", "None", "Simple"],
        ["Adam (FP16 states)", "2x", "Minimal", "Simple"],
        ["8-bit Optimizers", "4x", "Negligible", "Medium"],
        ["Adafactor", "4-8x", "Task-dependent", "Medium"],
        ["ZeRO Stage 1", "1.5x", "Minimal", "Medium"],
        ["ZeRO Stage 2", "4-8x", "Minor communication", "Medium"],
        ["ZeRO Stage 3", "10-20x", "More communication", "Complex"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(20) for item in row))
    
    print("\nWhen to Use Each Method:")
    print("- 8-bit Optimizers: Single-GPU training of medium-large models")
    print("- Adafactor: When precise convergence is less critical")
    print("- ZeRO: Multi-GPU training with memory constraints")
    print("- CPU Offloading: When GPU memory is severely limited")

def explain_efficient_attention():
    """
    Explain memory-efficient attention implementations
    """
    print("\n" + "=" * 60)
    print("MEMORY-EFFICIENT ATTENTION")
    print("=" * 60)
    
    print("\nStandard Attention Memory Problem:")
    print("- Attention requires storing the full attention matrix: N²×d")
    print("- For sequence length N=4096, hidden dim d=16, FP16: ~512MB per layer")
    print("- Scales quadratically with sequence length")
    
    print("\nEfficient Attention Implementations:")
    
    print("\n1. FlashAttention:")
    print("- Algorithm reorganizes attention computation for memory efficiency")
    print("- Minimizes memory transfers between GPU HBM and SRAM")
    print("- Memory savings: 10-20x less activation memory")
    print("- Speed improvement: 2-4x faster training")
    print("- Implementation:")
    print("""
    # Using Flash Attention 2
    from flash_attn import flash_attn_func
    
    # q, k, v are query, key, value tensors
    # shape: (batch_size, seq_len, num_heads, head_dim)
    
    # Regular attention
    # attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    # attn = torch.softmax(attn, dim=-1)
    # output = torch.matmul(attn, v)
    
    # Flash Attention
    output = flash_attn_func(q, k, v, softmax_scale=1/math.sqrt(head_dim))
    """)
    
    print("\n2. Memory-Efficient Attention in Transformers:")
    print("- Implementation in Hugging Face Transformers:")
    print("""
    # Enable memory-efficient attention
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    """)
    
    print("\n3. Attention Approximation Methods:")
    print("- Sparse Attention: Attend to a subset of tokens")
    print("- Linear Attention: Approximate attention with linear complexity")
    print("- Local Attention: Only attend to nearby tokens")
    
    print("\nComparison of Methods:")
    data = [
        ["Method", "Memory", "Speed", "Accuracy", "Complexity"],
        ["Standard Attention", "O(N²)", "1x", "Baseline", "Low"],
        ["Flash Attention", "O(N)", "2-4x", "Same", "Medium"],
        ["Sparse Attention", "O(N√N)", "1.5-2x", "Slight drop", "Medium"],
        ["Linear Attention", "O(N)", "2-3x", "Task dependent", "High"],
        ["Local Attention", "O(N·window)", "1.5-2x", "Long-range loss", "Medium"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(15) for item in row))
    
    print("\nBenchmark for FlashAttention vs Standard Attention:")
    print("For a 7B parameter model, batch size 16, sequence length 2048:")
    
    flash_vs_standard = [
        ["Metric", "Standard Attention", "FlashAttention", "Improvement"],
        ["Training Throughput", "1700 tokens/sec", "4100 tokens/sec", "2.4x"],
        ["Memory Usage", "28GB", "16GB", "1.75x"],
        ["Max Sequence Length", "2048", "4096", "2x"],
        ["Max Batch Size", "16", "32", "2x"]
    ]
    
    for row in flash_vs_standard:
        print(" | ".join(f"{item}".ljust(20) for item in row))
    
    print("\nWhen to Use:")
    print("- Flash Attention: Almost always when hardware supports it (A100, H100)")
    print("- Sparse Attention: Very long sequences where exact attention isn't critical")
    print("- Local Attention: Tasks with primarily local dependencies")

def explain_data_parallelism():
    """
    Explain data parallelism techniques for distributed training
    """
    print("\n" + "=" * 60)
    print("DATA PARALLELISM TECHNIQUES")
    print("=" * 60)
    
    print("\nDistributed Training Approaches:")
    print("1. Data Parallelism (DP): Same model on each device, different data")
    print("2. Model Parallelism (MP): Different parts of model on different devices")
    print("3. Pipeline Parallelism (PP): Different layers on different devices")
    print("4. Zero Redundancy Optimizer (ZeRO): Optimizer/gradient sharding")
    
    print("\nFocus on Data Parallelism:")
    
    print("\n1. PyTorch Distributed Data Parallel (DDP):")
    print("- Synchronous data parallelism")
    print("- Full model replica on each GPU")
    print("- All-reduce for gradient synchronization")
    print("- Implementation:")
    print("""
    # Initialize process group
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    
    # Create model and move to GPU
    model = MyModel().cuda()
    
    # Wrap model with DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Use model normally
    outputs = model(inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    """)
    
    print("\n2. Fully Sharded Data Parallelism (FSDP):")
    print("- Combines data parallelism with ZeRO-3")
    print("- Shards model parameters, gradients, and optimizer states")
    print("- Implementation:")
    print("""
    # Using PyTorch FSDP
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy
    )
    
    # Define mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device()
    )
    """)
    
    print("\n3. DeepSpeed ZeRO:")
    print("- Stage 1: Shard optimizer states")
    print("- Stage 2: Shard optimizer states + gradients")
    print("- Stage 3: Shard optimizer states + gradients + parameters")
    print("- Implementation:")
    print("""
    # Using DeepSpeed
    import deepspeed
    
    # Define DeepSpeed config
    ds_config = {
        "train_batch_size": 32,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
            "overlap_comm": True
        }
    }
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    """)
    
    print("\n4. Accelerate Library (Hugging Face):")
    print("- Simplifies distributed training setup")
    print("- Works with PyTorch, DeepSpeed, and FSDP")
    print("- Implementation:")
    print("""
    # Using Accelerate
    from accelerate import Accelerator
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Prepare model, optimizer, dataloader
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    # Training loop
    for batch in train_dataloader:
        outputs = model(batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
    """)
    
    print("\nComparison of Data Parallelism Methods:")
    data = [
        ["Method", "Memory Efficiency", "Communication", "Setup Complexity", "Best Use Case"],
        ["DDP", "Low", "All-reduce", "Simple", "When model fits in GPU"],
        ["FSDP", "High", "All-to-all", "Medium", "Large models, many GPUs"],
        ["ZeRO-1", "Medium", "All-reduce", "Medium", "Optimizer memory limited"],
        ["ZeRO-2", "Medium-High", "All-reduce", "Medium", "Gradient memory limited"],
        ["ZeRO-3", "Very High", "All-to-all", "Complex", "Parameter memory limited"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(18) for item in row))
    
    print("\nScaling Behavior (10B Parameter Model):")
    scaling = [
        ["GPUs", "DDP Memory/GPU", "FSDP Memory/GPU", "ZeRO-3 Memory/GPU"],
        ["1", "80GB", "80GB", "80GB"],
        ["2", "80GB", "40GB", "40GB"],
        ["4", "80GB", "20GB", "20GB"],
        ["8", "80GB", "10GB", "10GB"],
        ["16", "80GB", "5GB", "5GB"],
        ["32", "80GB", "2.5GB", "2.5GB"]
    ]
    
    for row in scaling:
        print(" | ".join(f"{item}".ljust(18) for item in row))

def explain_parameter_efficient_finetuning():
    """
    Explain Parameter-Efficient Fine-Tuning (PEFT) methods
    """
    print("\n" + "=" * 60)
    print("PARAMETER-EFFICIENT FINE-TUNING (PEFT)")
    print("=" * 60)
    
    print("\nWhat is PEFT?")
    print("PEFT refers to a family of techniques that fine-tune LLMs")
    print("by updating only a small subset of parameters or by adding")
    print("a small number of new parameters.")
    
    print("\nBenefits of PEFT:")
    print("- Memory efficiency: Update <1% of parameters")
    print("- Faster training: Less computation and memory access")
    print("- Reduced storage: Small adapter weights vs. full model")
    print("- Avoids catastrophic forgetting: Most weights frozen")
    print("- Composability: Can combine multiple adaptations")
    
    print("\nCommon PEFT Methods:")
    
    print("\n1. LoRA (Low-Rank Adaptation):")
    print("- Adds low-rank matrices to existing weights")
    print("- Parameterized as: W + ΔW = W + A·B")
    print("- Typical rank r = 8-64 (much less than original dimension)")
    print("- Implementation:")
    print("""
    # Using PEFT library
    from peft import get_peft_model, LoraConfig
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,                      # Low-rank dimension
        lora_alpha=32,             # Scale factor
        target_modules=["q_proj", "v_proj"],  # Which modules to add LoRA to
        lora_dropout=0.1,          # Dropout for LoRA layers
        bias="none"                # Don't modify biases
    )
    
    # Create PEFT model
    model = AutoModelForCausalLM.from_pretrained("llama-7b")
    peft_model = get_peft_model(model, lora_config)
    
    # Only LoRA parameters are updated
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable params: {trainable_params} ({trainable_params/all_params:.2%})")
    """)
    
    print("\n2. Adapters:")
    print("- Insert small feed-forward networks between layers")
    print("- Bottleneck architecture with down-projection and up-projection")
    print("- Typically 0.5-5% of original parameters")
    print("- Implementation:")
    print("""
    # Using PEFT library
    from peft import get_peft_model, AdapterConfig
    
    # Define adapter configuration
    adapter_config = AdapterConfig(
        hidden_size=768,           # Base model hidden size
        adapter_size=64,           # Bottleneck dimension
        adapter_initializer_range=0.01,
        adapter_dropout=0.1
    )
    
    # Create PEFT model
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    peft_model = get_peft_model(model, adapter_config)
    """)
    
    print("\n3. Prompt Tuning/Prefix Tuning:")
    print("- Add trainable tokens to the input")
    print("- Prompt tuning: In embedding space")
    print("- Prefix tuning: In each layer's activation space")
    print("- Implementation:")
    print("""
    # Using PEFT library
    from peft import get_peft_model, PrefixTuningConfig
    
    # Define prefix tuning configuration
    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=20,     # Number of virtual tokens
        encoder_hidden_size=768,   # Hidden size of model
        prefix_projection=True     # Project prefix to model dimensions
    )
    
    # Create PEFT model
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    peft_model = get_peft_model(model, prefix_config)
    """)
    
    print("\n4. QLoRA:")
    print("- LoRA + 4-bit Quantization + Double Quantization")
    print("- Quantize base model to 4-bit precision")
    print("- Apply LoRA adapters to quantized model")
    print("- Implementation:")
    print("""
    # Using PEFT and bitsandbytes
    from peft import get_peft_model, LoraConfig
    from transformers import BitsAndBytesConfig
    
    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"  # NormalFloat 4-bit quantization
    )
    
    # Load model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-70b",
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)
    """)
    
    print("\nComparison of PEFT Methods:")
    data = [
        ["Method", "Trainable Params", "Memory Usage", "Performance", "Complexity"],
        ["Full Fine-tuning", "100%", "High", "Excellent", "Low"],
        ["LoRA", "0.1-1%", "Very Low", "Good", "Low"],
        ["Adapters", "0.5-5%", "Low", "Good", "Medium"],
        ["Prompt Tuning", "<0.1%", "Very Low", "Moderate", "Medium"],
        ["Prefix Tuning", "0.1-1%", "Low", "Good", "Medium"],
        ["QLoRA", "0.1-1%", "Extremely Low", "Good", "Medium"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(16) for item in row))
    
    print("\nWhen to Use Each Method:")
    print("- LoRA: General-purpose fine-tuning on limited hardware")
    print("- QLoRA: Very large models (>20B) on consumer hardware")
    print("- Adapters: When task-specific modularity is important")
    print("- Prompt/Prefix Tuning: Multiple lightweight task adaptations")

def explain_quantization_training():
    """
    Explain quantization techniques for training LLMs
    """
    print("\n" + "=" * 60)
    print("QUANTIZATION FOR TRAINING")
    print("=" * 60)
    
    print("\nWhat is Quantization?")
    print("Quantization reduces the precision of model weights, activations,")
    print("and/or gradients to lower bit-widths (8-bit, 4-bit, or even lower).")
    
    print("\nTypes of Quantization:")
    print("1. Post-Training Quantization (PTQ): Applied after training")
    print("2. Quantization-Aware Training (QAT): During training")
    print("3. Activation-aware Weight Quantization (AWQ): Preserves activation patterns")
    
    print("\nBenefits for Training:")
    print("- Memory reduction: 2-8x less memory for weights")
    print("- Faster computation: Potential speed-up on specialized hardware")
    print("- Larger models: Train models that wouldn't fit in full precision")
    print("- Energy efficiency: Lower power consumption")
    
    print("\nQuantization Methods:")
    
    print("\n1. 8-bit Quantization:")
    print("- INT8 representation of weights and/or activations")
    print("- Minimal accuracy loss for most models")
    print("- Implementation:")
    print("""
    # Using bitsandbytes for 8-bit training
    import bitsandbytes as bnb
    
    # Load model in 8-bit
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        load_in_8bit=True,
        device_map="auto"
    )
    """)
    
    print("\n2. 4-bit Quantization:")
    print("- INT4 or NF4 (NormalFloat) representation")
    print("- Memory savings: Up to 8x compared to FP32")
    print("- Implementation:")
    print("""
    # Using bitsandbytes for 4-bit training
    from transformers import BitsAndBytesConfig
    
    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # "fp4" or "nf4"
        bnb_4bit_use_double_quant=True  # Double quantization
    )
    
    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        quantization_config=quantization_config,
        device_map="auto"
    )
    """)
    
    print("\n3. Quantization-Aware Training (QAT):")
    print("- Simulates quantization effects during training")
    print("- Weights stored in full precision but quantized in forward pass")
    print("- Implementation:")
    print("""
    # Simple example of QAT with PyTorch
    class QuantizedLinear(nn.Module):
        def __init__(self, in_features, out_features, nbits=8):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.nbits = nbits
            self.scale = nn.Parameter(torch.ones(1))
            
        def forward(self, x):
            # Quantize weights during forward pass
            weight_q = self.quantize(self.weight, self.nbits)
            return F.linear(x, weight_q, self.bias)
            
        def quantize(self, tensor, nbits):
            # Simulate quantization
            scale = self.scale
            max_val = 2**(nbits-1) - 1
            tensor_q = torch.round(tensor / scale * max_val) * scale / max_val
            return tensor_q
    """)
    
    print("\n4. Mixed-Precision Quantization:")
    print("- Different precisions for different layers or parameter types")
    print("- Implementation:")
    print("""
    # Example of mixed quantization with Transformers
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    # Configure different quantization for different modules
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # Keep attention in 8-bit for better accuracy
        bnb_4bit_use_double_quant_for_attention=False
    )
    """)
    
    print("\nComparison of Methods:")
    data = [
        ["Method", "Memory Savings", "Accuracy Impact", "Training Speed"],
        ["FP32 (baseline)", "1x", "None", "1x"],
        ["BF16/FP16", "2x", "Minimal", "1.5-2x"],
        ["INT8", "4x", "Small", "0.8-1.2x"],
        ["INT4/NF4", "8x", "Moderate", "0.7-1x"],
        ["QAT (INT8)", "~4x", "Minimal", "0.7-0.9x"],
        ["Mixed Quantization", "4-6x", "Task dependent", "0.8-1.1x"]
    ]
    
    for row in data:
        print(" | ".join(f"{item}".ljust(18) for item in row))
    
    print("\nBest Practices:")
    print("- Start with higher precision, gradually reduce")
    print("- Layer-wise quantization (keep critical layers in higher precision)")
    print("- Combine with PEFT methods (e.g., QLoRA)")
    print("- Use specialized libraries (bitsandbytes, AutoGPTQ)")
    print("- Consider hardware-aware quantization")

def benchmark_optimization_techniques():
    """
    Present benchmark results for various optimization techniques
    """
    print("\n" + "=" * 60)
    print("BENCHMARKING OPTIMIZATION TECHNIQUES")
    print("=" * 60)
    
    print("\nBenchmark Setup:")
    print("- Model: 7B parameter decoder-only LLM (LLaMA architecture)")
    print("- Hardware: NVIDIA A100 80GB GPU")
    print("- Batch size: Variable (maximum possible for each technique)")
    print("- Sequence length: 2048 tokens")
    
    print("\nMemory Usage Benchmarks:")
    memory_benchmark = [
        ["Technique", "Memory (GB)", "Relative", "Max Batch Size"],
        ["Baseline (FP32)", "80", "100%", "4"],
        ["Mixed Precision (BF16)", "41", "51%", "8"],
        ["+ Gradient Checkpointing", "26", "33%", "12"],
        ["+ 8-bit Optimizer", "18", "23%", "16"],
        ["+ FlashAttention", "14", "18%", "20"],
        ["LoRA + BF16", "11", "14%", "24"],
        ["QLoRA (4-bit)", "6", "8%", "32"]
    ]
    
    for row in memory_benchmark:
        print(" | ".join(f"{item}".ljust(24) for item in row))
    
    print("\nTraining Throughput Benchmarks:")
    throughput_benchmark = [
        ["Technique", "Tokens/sec", "Relative"],
        ["Baseline (FP32, batch=4)", "1100", "100%"],
        ["Mixed Precision (BF16, batch=8)", "2400", "218%"],
        ["+ Gradient Checkpointing", "1900", "173%"],
        ["+ 8-bit Optimizer", "1850", "168%"],
        ["+ FlashAttention", "2800", "255%"],
        ["LoRA + BF16 (batch=24)", "3400", "309%"],
        ["QLoRA (4-bit, batch=32)", "3100", "282%"]
    ]
    
    for row in throughput_benchmark:
        print(" | ".join(f"{item}".ljust(32) for item in row))
    
    print("\nTraining Time for 10B Tokens:")
    training_time = [
        ["Technique", "Time (days)", "Cost (A100 hours)"],
        ["Baseline (FP32)", "105", "2520"],
        ["Mixed Precision (BF16)", "48", "1152"],
        ["All Optimizations", "31", "744"],
        ["LoRA + BF16", "34", "816"],
        ["QLoRA (4-bit)", "37", "888"],
        ["DeepSpeed ZeRO-3 (8x A100)", "7", "1344"]
    ]
    
    for row in training_time:
        print(" | ".join(f"{item}".ljust(24) for item in row))
    
    print("\nAccuracy Impact:")
    accuracy_impact = [
        ["Technique", "Perplexity Change", "Downstream Task Accuracy"],
        ["Baseline (FP32)", "0%", "100%"],
        ["Mixed Precision (BF16)", "~0%", "~100%"],
        ["Gradient Checkpointing", "~0%", "~100%"],
        ["8-bit Optimizer", "+1-2%", "98-100%"],
        ["FlashAttention", "~0%", "~100%"],
        ["LoRA", "+2-5%", "95-99%"],
        ["4-bit Quantization", "+3-8%", "92-98%"],
        ["QLoRA", "+3-8%", "92-98%"]
    ]
    
    for row in accuracy_impact:
        print(" | ".join(f"{item}".ljust(24) for item in row))
    
    print("\nRecommended Combinations:")
    print("1. For single GPU, limited memory:")
    print("   QLoRA + BF16 + FlashAttention + Gradient Checkpointing")
    print("2. For multi-GPU setup:")
    print("   DeepSpeed ZeRO-3 + BF16 + FlashAttention")
    print("3. For maximum throughput:")
    print("   FSDP + BF16 + FlashAttention")
    print("4. For best accuracy/memory trade-off:")
    print("   LoRA + BF16 + FlashAttention + 8-bit optimizer")

def main():
    """
    Main function to demonstrate training optimizations
    """
    introduction_to_optimizations()
    
    # Explain each optimization technique
    explain_mixed_precision()
    explain_gradient_checkpointing()
    explain_optimizer_state_reduction()
    explain_efficient_attention()
    explain_data_parallelism()
    explain_parameter_efficient_finetuning()
    explain_quantization_training()
    
    # Present benchmarks
    benchmark_optimization_techniques()

if __name__ == "__main__":
    main()
