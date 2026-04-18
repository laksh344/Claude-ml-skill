# LLM Fine-tuning Competitions (Nemotron, LoRA, RLHF, SFT)

## Overview & 2026 Research State

The NVIDIA Nemotron Reasoning Challenge provides a shared baseline (Nemotron-3-Nano-30B)
and a strictly controlled Kaggle evaluation environment to prevent "illusion of competence":

- **Temperature = 0.0**: absolute determinism, no sampling variance
- **Exact string match OR ε = 1×10⁻⁵ tolerance**: near-zero tolerance for errors  
- **Hard 7,680 token limit** on chain-of-thought: forces concise, efficient reasoning
- **Hidden test set**: models must generalize algorithms, not memorize training data

Key 2026 finding: **QLoRA on larger model > standard LoRA on smaller model**.
Quantization accuracy loss << raw capability gain from more parameters.

## Fine-tuning with Unsloth (1.5× Speed, 60% Less VRAM)

```python
# Unsloth dramatically speeds up training — essential for competition time constraints
from unsloth import FastLanguageModel
import torch

# Load model with Unsloth optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "nvidia/Nemotron-3-Nano-3B",   # or larger target model
    max_seq_length= 2048,
    dtype         = None,                           # auto-detect
    load_in_4bit  = True,                           # QLoRA
)

# Apply LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r              = 32,          # rank: 16–64 for complex reasoning tasks
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_alpha     = 64,          # 2× rank is standard
    lora_dropout   = 0.1,         # regularization to prevent overfitting
    bias           = "none",
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state   = 42,
)
model.print_trainable_parameters()
# Should print ~0.5–2% of total parameters
```

## Critical: Dataset Composition for Reasoning Preservation

**2026 research finding**: training datasets must maintain ≥75% reasoning-style examples.
Falling below this threshold causes catastrophic forgetting of the chain-of-thought protocol.

```python
from datasets import Dataset

def format_reasoning_example(example):
    """Format with full chain-of-thought trace — REQUIRED for reasoning preservation."""
    return {
        "text": f"""<|user|>
{example['question']}
<|assistant|>
<think>
{example['reasoning']}
</think>

{example['answer']}"""
    }

def format_direct_example(example):
    """Direct answer — use for at most 25% of training data."""
    return {
        "text": f"<|user|>\n{example['question']}\n<|assistant|>\n{example['answer']}"
    }

# Build dataset: 75% CoT + 25% direct answers
cot_data    = reasoning_dataset.map(format_reasoning_example)
direct_data = direct_dataset.select(range(len(cot_data) // 3)).map(format_direct_example)
combined    = concatenate_datasets([cot_data, direct_data]).shuffle(seed=42)
```

## Hyperparameter Guide (Nemotron-Optimized)

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir               = "./nemotron-lora",
    num_train_epochs         = 3,
    per_device_train_batch_size = 1,        # keep small to avoid GPU OOM
    gradient_accumulation_steps = 16,       # effective batch = 16 (stable training)
    learning_rate            = 2e-4,        # start here; reduce if loss spikes
    lr_scheduler_type        = "cosine",
    warmup_ratio             = 0.05,
    bf16                     = True,        # bfloat16 for modern GPUs
    logging_steps            = 10,
    save_strategy            = "epoch",
    evaluation_strategy      = "epoch",
    load_best_model_at_end   = True,
    # CRITICAL: mask user inputs, train only on assistant completions
    # This is handled by SFTTrainer's dataset_text_field approach
)

trainer = SFTTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = combined["train"],
    eval_dataset     = combined["test"],
    dataset_text_field = "text",
    max_seq_length   = 2048,
    packing          = True,             # pack short examples together for efficiency
)
trainer.train()
```

## Handling Overfitting and Underfitting

```python
# OVERFITTING SYMPTOMS: perfect train loss, poor eval; model loses general conversation
# FIXES:
#   1. Increase lora_dropout to 0.1–0.15
#   2. Increase gradient_accumulation_steps (effective larger batch)
#   3. Weight averaging: blend fine-tuned weights with base model
def weight_average(base_model, finetuned_model, alpha=0.5):
    """Blend fine-tuned with base to recover general capabilities."""
    for (name, base_param), (_, ft_param) in zip(
        base_model.named_parameters(), finetuned_model.named_parameters()
    ):
        ft_param.data = alpha * ft_param.data + (1 - alpha) * base_param.data
    return finetuned_model

# UNDERFITTING SYMPTOMS: high train loss; model ignores instruction format
# FIXES:
#   1. Decrease batch size to 1
#   2. Use more domain-specific / harder examples
#   3. Increase learning_rate (try 3e-4 or 4e-4)
#   4. Increase num_train_epochs to 5
```

## Synthetic Data Generation (Top-Competitor Strategy)

```python
import anthropic

client = anthropic.Anthropic()

def generate_reasoning_traces(topic: str, n: int = 100) -> list:
    """
    Generate chain-of-thought training examples using a strong model.
    Top Nemotron competitors generate MILLIONS of traces, then filter.
    """
    examples = []
    for _ in range(n):
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1500,
            messages   = [{
                "role": "user",
                "content": f"""Generate a challenging {topic} reasoning problem.
Format EXACTLY as:
Question: [problem]
Reasoning: [detailed step-by-step thinking — show ALL work]
Answer: [final answer]"""
            }]
        )
        examples.append(parse_qa(response.content[0].text))
    return examples

def filter_high_quality_traces(examples, model, tokenizer):
    """
    Nemotron top strategy: filter traces by minimum log probability of answer tokens.
    Penalize and remove tokens that exhibit high loss (hallucinated logic).
    """
    filtered = []
    for ex in examples:
        # Compute log prob of the answer token sequence
        log_prob = compute_answer_log_prob(model, tokenizer, ex)
        if log_prob > -2.0:   # threshold: only keep high-confidence traces
            filtered.append(ex)
    return filtered
```

## RLHF / PPO (Advanced — After SFT Baseline)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

ppo_config = PPOConfig(
    model_name     = "path/to/sft-checkpoint",
    learning_rate  = 1.4e-5,
    mini_batch_size= 4,
    batch_size     = 16,
    ppo_epochs     = 4,
    kl_penalty     = "kl",
    init_kl_coef   = 0.2,     # controls deviation from reference model
)

# Rule-based reward for math/reasoning (avoid training reward model — too slow)
def compute_reward(question, generated, correct_answer):
    if generated.strip() == correct_answer.strip():
        return 1.0
    # Partial credit for showing work
    reward = 0.3 if correct_answer in generated else 0.0
    # Length penalty: reward concise chains (Nemotron 7680 token limit!)
    length_penalty = min(1.0, 500 / max(1, len(generated.split())))
    return reward * length_penalty
```

## Evaluation (Deterministic — Nemotron Rules)

```python
def evaluate_deterministic(model, tokenizer, benchmark):
    """Temperature=0.0, exact match — mirrors Nemotron evaluation."""
    correct = 0
    for ex in benchmark:
        inputs = tokenizer(ex['question'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                temperature    = 0.0,    # MUST be 0 for Nemotron eval
                do_sample      = False,  # greedy decoding
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer    = extract_answer(generated)
        
        # Check exact match OR numerical tolerance (ε = 1e-5)
        try:
            if abs(float(answer) - float(ex['answer'])) < 1e-5:
                correct += 1
        except (ValueError, TypeError):
            if answer and answer.strip() == ex['answer'].strip():
                correct += 1
    
    return correct / len(benchmark)
```

## Save and Submit LoRA Adapter

```python
# Save ONLY the adapter (not full model weights)
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")

# Verify adapter loads correctly
from peft import PeftModel
from transformers import AutoModelForCausalLM

base   = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-3-Nano-3B", torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "./lora-adapter")
```

## Tips
- Token limit (7,680) forces simple, direct reasoning — no rambling chains
- Tokenization awareness: numbers and symbols merge into unpredictable blocks — preprocess carefully
- For Nemotron: LoRA r=32–64 needed for complex logical datasets
- Weight averaging (blend fine-tuned + base) prevents losing general ability
- Top competitors filter millions of synthetic traces by log probability before training
