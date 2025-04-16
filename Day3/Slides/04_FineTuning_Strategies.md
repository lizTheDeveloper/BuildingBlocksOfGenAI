# Single-Task vs. Multi-Task Fine-Tuning

## Fine-tuning Approaches Overview

| Approach | Description | Best Use Case |
|----------|-------------|---------------|
| **Single-Task** | Fine-tune on one specific task | Maximum performance on target task |
| **Multi-Task** | Fine-tune on multiple tasks simultaneously | Better generalization, reduced forgetting |
| **Sequential** | Fine-tune on one task, then another | Transfer learning from related tasks |
| **Parameter-Efficient** | Update small subset of parameters | Resource constraints, preserve capabilities |

## Single-Task Fine-Tuning

- **Definition**: Adapt a pre-trained model for a specific task
- **Process**:
  1. Select pre-trained model
  2. Prepare task-specific dataset
  3. Fine-tune all model parameters
  4. Evaluate on task-specific metrics

![Single-Task Fine-Tuning](./images/single_task.png)

## Single-Task: Pros and Cons

### Advantages
- Maximizes performance on the target task
- Simpler implementation and training process
- Easier to interpret results
- Lower computational requirements
- Clear evaluation metrics

### Disadvantages
- Risk of catastrophic forgetting
- Poor generalization to other tasks
- Requires separate models for each task
- Less sample-efficient (needs more data)
- May overfit to task-specific patterns

## Multi-Task Fine-Tuning

- **Definition**: Train a single model on multiple tasks simultaneously
- **Process**:
  1. Select pre-trained model
  2. Prepare datasets for multiple tasks
  3. Format with task identifiers or prefixes
  4. Train on all tasks with appropriate sampling
  5. Evaluate on all tasks individually

![Multi-Task Fine-Tuning](./images/multi_task.png)

## Multi-Task: Pros and Cons

### Advantages
- Better generalization to new tasks
- Reduced catastrophic forgetting
- Improved sample efficiency
- Single model for multiple capabilities
- Positive transfer between related tasks

### Disadvantages
- More complex training setup
- Harder to balance multiple objectives
- Risk of negative interference between tasks
- Higher computational requirements
- May not reach single-task performance

## Task Sampling Strategies

![Task Sampling](./images/task_sampling.png)

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Uniform** | Equal probability for all tasks | Tasks of equal importance |
| **Proportional** | Sample by dataset size | Different sized datasets |
| **Temperature** | Apply temperature to adjust distribution | Balance diversity and frequency |
| **Loss-based** | Sample more from tasks with higher loss | Dynamic focus on harder tasks |
| **Curriculum** | Start with easier tasks, progress to harder | Optimization for complex tasks |

## Example: Multi-Task Learning with T5

```python
# Format examples with task prefixes
summarization = "summarize: " + document
translation = "translate English to French: " + text
qa = f"question: {question} context: {context}"

# T5 handles different tasks through prefixes
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

# Process batch with mixed tasks
outputs = model(
    input_ids=batch_inputs,
    attention_mask=batch_masks,
    labels=batch_labels
)

loss = outputs.loss  # Works for all tasks!
```

## Parameter-Efficient Fine-Tuning (PEFT)

- **Definition**: Update only a small subset of parameters during fine-tuning
- **Motivation**: 
  - Reduce memory requirements
  - Faster training
  - Mitigate catastrophic forgetting
  - Enable multiple task adaptations

![PEFT Methods](./images/peft_methods.png)

## Low-Rank Adaptation (LoRA)

- **Concept**: Update low-rank decomposition of weight matrices
- **Formula**: $W + ΔW = W + A×B$ where $A$ and $B$ are low-rank matrices
- **Parameters**: ~0.1-1% of full model parameters
- **Advantages**:
  - Dramatically reduced memory footprint
  - Almost no inference overhead
  - Can be merged back into original weights

![LoRA](./images/lora.png)

## Adapter Modules

- **Concept**: Insert small feed-forward networks between layers
- **Structure**: Down-projection → Activation → Up-projection
- **Parameters**: ~0.5-5% of full model parameters
- **Advantages**:
  - Modular (can swap adapters for different tasks)
  - Original weights preserved entirely
  - Established approach with strong performance

![Adapters](./images/adapters.png)

## Prompt/Prefix Tuning

- **Concept**: Add trainable "soft prompts" to the input
- **Implementation**:
  - Prompt Tuning: Add tokens to input embedding space
  - Prefix Tuning: Add tokens to each layer's activation space
- **Parameters**: <0.1% of full model parameters
- **Advantages**:
  - Extremely parameter-efficient
  - Works better with larger models
  - Simple implementation

![Prompt Tuning](./images/prompt_tuning.png)

## Comparison of PEFT Methods

| Method | Parameters | Memory | Performance | Complexity |
|--------|------------|--------|-------------|------------|
| Full Fine-tuning | 100% | High | Excellent | Low |
| LoRA | 0.1-1% | Very Low | Good | Low |
| Adapters | 0.5-5% | Low | Good | Medium |
| Prompt Tuning | <0.1% | Very Low | Moderate | Medium |
| Prefix Tuning | 0.1-1% | Low | Good | Medium |
| QLoRA | 0.1-1% | Extremely Low | Good | Medium |

## Transfer Learning with LLMs

- **Definition**: Leverage knowledge from source task to improve target task
- **Types**:
  1. **Domain Transfer**: General → Domain-specific (e.g., medical)
  2. **Task Transfer**: Related task → Target task
  3. **Cross-lingual Transfer**: High-resource → Low-resource language

![Transfer Learning](./images/transfer_learning.png)

## Continued Pre-training

- **Process**: Continue pre-training on domain-specific corpus
- **Benefits**:
  - Adapts to domain vocabulary and style
  - Retains general capabilities
  - Better foundation for task fine-tuning
- **Examples**:
  - BioBERT (biomedical domain)
  - LegalBERT (legal domain)
  - FinBERT (financial domain)

## Sequential Fine-tuning

- **Process**: Fine-tune on related task, then target task
- **Benefits**:
  - Useful when target task has limited data
  - Transfers knowledge from high-resource to low-resource tasks
- **Example**:
  - MNLI → RTE (entailment tasks)
  - English → Other languages

```python
# First, fine-tune on high-resource task
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
trainer_high_resource.train()

# Then, fine-tune on low-resource task
# (Same model, new trainer)
trainer_low_resource.train()
```

## Today's Exercise: DistilBERT Fine-tuning

1. **Single-Task Fine-tuning**:
   - Fine-tune DistilBERT on sentiment analysis (SST-2)
   - Evaluate performance on in-domain and out-of-domain data

2. **Multi-Task Fine-tuning**:
   - Fine-tune DistilBERT on multiple text classification tasks
   - Implement task-specific heads and task sampling
   - Compare performance with single-task model

3. **Analysis**:
   - Measure generalization capabilities
   - Evaluate catastrophic forgetting
   - Compare resource efficiency

## Performance vs. Generalization Trade-off

![Performance vs. Generalization](./images/performance_generalization.png)

- **Single-Task**: Higher performance on specific task
- **Multi-Task**: Better generalization to related tasks
- **PEFT**: Good balance of performance, efficiency, and preservation
- **Zero-Shot**: Excellent general capabilities, weaker on specific tasks

---

# Questions?
