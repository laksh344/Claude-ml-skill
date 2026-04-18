# Math Reasoning Competitions (AIMO3 — AI Mathematical Olympiad)

## Overview & 2026 Research State

AIMO3: 110 original Olympiad-level problems (Algebra, Combinatorics, Geometry, Number Theory).
- Prize: $2.2M+ | Ends: April 2026 | On Kaggle
- Answers: **5-digit integers** [0, 99999] — near-zero guessing probability
- Compute: up to 128 H100s for select participants
- Requirement: all winning solutions must be open-sourced

### Past Winners
| Competition | Winner | Approach |
|-------------|--------|----------|
| AIMO1 | Project Numina | Chain-of-thought + symbolic tools |
| AIMO2 | NVIDIA NemoSkills | LoRA fine-tuned + RL + majority vote |
| AIMO3 | Ongoing (Apr 2026) | LB leaders: 44/50 (grand prize = 47/50) |

**Key 2026 finding**: Test-time compute (majority vote over many samples) consistently
beats single-sample strategies — depth of search > model size.

## Strategy: Test-Time Compute (Most Important Factor)

```python
from collections import Counter
import torch

def solve_with_majority_vote(problem: str, model, tokenizer,
                              n_samples: int = 64, temperature: float = 0.8):
    """
    Generate many solutions, pick the most frequent 5-digit answer.
    AIMO2 winner used this approach — more samples = higher accuracy.
    """
    answers = []
    
    for _ in range(n_samples):
        response = generate(model, tokenizer, problem, temperature=temperature)
        answer   = extract_final_answer(response)
        if answer is not None and 0 <= answer <= 99999:
            answers.append(answer)
    
    if not answers:
        return None
    
    return Counter(answers).most_common(1)[0][0]

def extract_final_answer(text: str):
    """Extract 5-digit answer from chain-of-thought."""
    import re
    patterns = [
        r'\\boxed\{(\d+)\}',
        r'[Aa]nswer[:\s]+(\d+)',
        r'= (\d{1,5})(?:\s|$)',
        r'\\textbf\{(\d+)\}',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 99999:
                return val
    return None
```

## Model Selection (2026 Best for Math)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Top performers for math reasoning (2025–2026)
# 1. Qwen2.5-Math-72B-Instruct — state of the art for math
# 2. DeepSeek-R1 — strong CoT, good at symbolic manipulation
# 3. Llama-3.1-70B fine-tuned on math data (NuminaMath, OpenMathInstruct)

model_name = "Qwen/Qwen2.5-Math-72B-Instruct"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype  = torch.bfloat16,
    device_map   = "auto",
)

MATH_PROMPT = """\
You are an expert at solving competition mathematics problems.
Think step by step. Show all work. The answer is a non-negative integer less than 100000.

Problem: {problem}

Solution:"""

def generate(model, tokenizer, problem, temperature=0.8, max_new_tokens=2048):
    prompt  = MATH_PROMPT.format(problem=problem)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            do_sample      = True,
            top_p          = 0.95,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                             skip_special_tokens=True)
```

## Fine-Tuning on Math Data

```python
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets

# High-quality math datasets
datasets_to_use = [
    "AI-MO/NuminaMath-CoT",        # competition math with CoT
    "nvidia/OpenMathReasoning",     # NVIDIA's open math reasoning dataset
    "nvidia/OpenCodeReasoning",     # code + math hybrid reasoning
]

def format_math(example):
    return {
        "text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}\n\n\\boxed{{{example['answer']}}}"
    }

# LoRA config (same as llm-finetune.md, but higher rank for complex math)
from peft import LoraConfig, TaskType
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 64,           # higher rank needed for mathematical nuance
    lora_alpha     = 128,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
)
```

## Symbolic / Tool-Augmented Approaches

```python
import sympy as sp
from sympy import symbols, solve, simplify, expand, factor

def solve_with_sympy_agent(problem_text, model, tokenizer):
    """
    Let LLM generate SymPy code → execute → verify numerically.
    Hybrid: symbolic execution + LLM for translation.
    """
    code_prompt = f"""Solve this math problem by writing Python/SymPy code.
The final answer must be a non-negative integer less than 100000.
Problem: {problem_text}
Write complete Python code that prints ONLY the integer answer."""
    
    code = generate(model, tokenizer, code_prompt, temperature=0.2)
    code = extract_code_block(code)
    
    # Safe execution sandbox
    try:
        import io, contextlib
        stdout_capture = io.StringIO()
        safe_globals   = {
            'sp': sp, 'symbols': symbols, 'solve': solve,
            'simplify': simplify, 'print': print,
            'range': range, 'int': int, 'float': float,
            '__builtins__': {}
        }
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        output = stdout_capture.getvalue().strip()
        val    = int(output.split('\n')[-1])
        if 0 <= val <= 99999:
            return val
    except Exception:
        pass
    return None

def hybrid_solve(problem, model, tokenizer, n_samples=32):
    """Try symbolic first, fall back to majority vote."""
    symbolic = solve_with_sympy_agent(problem, model, tokenizer)
    if symbolic is not None:
        return symbolic
    return solve_with_majority_vote(problem, model, tokenizer, n_samples=n_samples)
```

## Process Reward Model (Advanced — if time allows)

```python
# PRM: score individual reasoning steps, not just final answers
# Helps filter out hallucinated logic chains

class ProcessRewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.head     = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[:, -1, :]).squeeze(-1)   # score at last token

def filter_with_prm(candidates, prm, tokenizer):
    """Re-rank solutions by PRM score — select the best reasoning chain."""
    scores = []
    for text in candidates:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
        with torch.no_grad():
            score = prm(**inputs).item()
        scores.append(score)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best_idx]
```

## Evaluation

```python
def evaluate_math(model, tokenizer, problems, n_samples=16):
    correct = 0
    for ex in problems:
        answer = solve_with_majority_vote(
            ex['problem'], model, tokenizer, n_samples=n_samples)
        if answer == int(ex['answer']):
            correct += 1
    return correct / len(problems)
```

## Tips
- Answers are 5-digit integers → validate EVERY prediction is in [0, 99999]
- More samples (n=64) beats better model with n=1 — spend compute on sampling
- Coverage matters: ensure training data spans Algebra, Combinatorics, Geometry, Number Theory
- Chain-of-thought quality > quantity — filter noisy traces by log probability before training
- For AIMO3: 44/50 teams on public LB; grand prize requires 47/50 — every problem counts
