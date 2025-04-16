# Training LLMs: The (Not So) Easy Part

## Decision Framework: When to Train vs. Not Train

| Consideration | Train Your Own LLM | Use Pre-trained LLM |
|---------------|-------------------|---------------------|
| **Data Needs** | Domain-specific corpus (100GB+) | General knowledge sufficient |
| **Compute Resources** | High-end GPUs/TPUs clusters | API access or consumer GPU |
| **Expertise Required** | Deep ML engineering knowledge | Prompt engineering skills |
| **Time Frame** | Weeks to months | Hours to days |
| **Budget** | $10,000s to millions | $10s to $1,000s |
| **Control** | Complete architectural control | Limited to model capabilities |
| **Use Case** | Highly specialized applications | General-purpose applications |

## The Computational Challenge

![LLM Training Resources](./images/compute_resources.png)

| Model Size | GPU Memory | Training Time | Training Cost |
|------------|------------|---------------|---------------|
| 125M       | 1 GB       | Hours         | ~$100         |
| 1.5B       | 12 GB      | Days          | ~$1,500       |
| 7B         | 50 GB      | Weeks         | ~$10,000      |
| 70B        | 500 GB     | Months        | ~$75,000      |
| 175B+      | 1.4 TB+    | Months        | ~$200,000+    |

## Scaling Laws in LLM Training

- **Parameter Scaling**: Performance scales as a power law with model size
- **Data Scaling**: Performance scales logarithmically with dataset size
- **Compute Scaling**: Fixed compute budget → trade off size vs. steps
- **Chinchilla Scaling**: For optimal performance, dataset size should scale linearly with model size

![Scaling Laws](./images/scaling_laws.png)

## Catastrophic Forgetting

- **Definition**: When fine-tuning causes a model to lose previously learned capabilities
- **Example**: A medical LLM that can no longer write creative fiction
- **Mechanisms**:
  - Representational drift from pre-trained knowledge
  - Overspecialization to the fine-tuning dataset
  - Limited capacity to maintain multiple capabilities

## Measuring Catastrophic Forgetting

```python
def evaluate_forgetting(model, general_tasks, before_scores, after_scores):
    """Compare model performance before and after fine-tuning"""
    for task in general_tasks:
        retention = after_scores[task] / before_scores[task] * 100
        print(f"{task}: {retention:.1f}% retained")
        
        if retention < 80:
            print(f"WARNING: Significant forgetting detected on {task}!")
```

## Memory Optimization Techniques

1. **Mixed Precision Training**
   - Use lower precision (FP16/BF16) for most operations
   - Maintain FP32 master weights
   - 2x memory reduction, possible speedup

2. **Gradient Checkpointing**
   - Store subset of activations during forward pass
   - Recompute during backward pass
   - Trade compute for memory (30% slower, 5x less memory)

3. **Optimizer State Reduction**
   - 8-bit optimizers (bitsandbytes)
   - Parameter sharing 
   - Memory-efficient optimizers (Adafactor)

## Full Fine-tuning Approach

- **Process Overview**:
  1. Start with a pre-trained model
  2. Update all parameters during fine-tuning
  3. Optimize for the target task
  
- **Training Configuration**:
  - Learning rate: 1e-5 to 5e-5 (smaller than pre-training)
  - Batch size: As large as memory allows
  - Weight decay: 0.01 to 0.1
  - Learning schedule: Warmup + decay
  - Early stopping to prevent overfitting

## DeepSpeed ZeRO for Distributed Training

![DeepSpeed ZeRO](./images/deepspeed_zero.png)

- **ZeRO Stage 1**: Optimizer state partitioning
- **ZeRO Stage 2**: + Gradient partitioning
- **ZeRO Stage 3**: + Parameter partitioning
- **ZeRO-Offload**: CPU memory offloading
- **ZeRO-Infinity**: NVMe offloading

## Example: Fine-tuning Flan-T5

```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer

# Load pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./flan-t5-finetuned",
    learning_rate=3e-5,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,  # Mixed precision
    # ... other arguments
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # ... other arguments
)

# Start training
trainer.train()
```

## Today's Exercise: Flan-T5 Fine-tuning

1. Evaluate Flan-T5's performance on:
   - Target task: Conversation summarization
   - General tasks: Translation, question answering, etc.

2. Fine-tune on the target task

3. Re-evaluate on all tasks to measure:
   - Improvement on target task
   - Forgetting on general tasks

4. Analyze the trade-offs

---

# Questions?
