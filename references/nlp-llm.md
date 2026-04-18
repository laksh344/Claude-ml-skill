# NLP / Text Competitions

## Overview

Types: text classification, named entity recognition, question answering,
summarization, entity linking (SNOMED), LLM output detection, RAG, multi-label.

Winning trend: **decoder models (Qwen, Llama, Gemma, DeepSeek) now dominate**,
but encoder models (DeBERTa) still win when labeled data is abundant.

## Text Classification with DeBERTa

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn

MODEL = "microsoft/deberta-v3-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len=512):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_classes)

# Training with gradient accumulation (for large models on small GPU)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
ACCUM_STEPS = 4

for batch_idx, batch in enumerate(train_loader):
    outputs = model(**{k: v.cuda() for k, v in batch.items() if k != 'label'},
                    labels=batch['label'].cuda())
    loss = outputs.loss / ACCUM_STEPS
    loss.backward()
    
    if (batch_idx + 1) % ACCUM_STEPS == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## RAG Pipeline (Retrieval-Augmented Generation)

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# Build vector index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # or "BAAI/bge-large-en"
docs = ["doc text 1", "doc text 2", ...]

embeddings = embed_model.encode(docs, batch_size=64, normalize_embeddings=True)
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product = cosine for normalized
index.add(embeddings.astype(np.float32))

# Query
def retrieve(query: str, k: int = 5) -> list[str]:
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb.astype(np.float32), k)
    return [docs[i] for i in indices[0]]

# Generate with context (ColBERT used by Zindi Telecom winner)
def rag_answer(query, model, tokenizer):
    context_docs = retrieve(query, k=3)
    context = "\n\n".join(context_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return generate(model, tokenizer, prompt)
```

## Detecting AI-Generated Text

```python
# Ensemble of detectors (winning approach in LLM detection comps)
# Used: Mistral-7B + DeBERTa + Llama ensemble

class EnsembleDetector:
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, texts):
        all_preds = []
        for model, tokenizer in self.models:
            preds = batch_predict(model, tokenizer, texts)
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)  # average ensemble
```

## Multi-label NLP (e.g., topic classification)

```python
# BCE loss for multi-label
criterion = nn.BCEWithLogitsLoss()

# Threshold tuning per class
def find_best_thresholds(y_true, y_prob):
    from sklearn.metrics import f1_score
    thresholds = []
    for cls in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(y_true[:, cls], y_prob[:, cls] > t)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds
```

## Synthetic Data for NLP

```python
import anthropic

client = anthropic.Anthropic()

def generate_synthetic_examples(label: str, n: int = 50) -> list[str]:
    """Generate training examples for underrepresented classes."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Generate {n} diverse text examples labeled as '{label}'.
Each example should be on a separate line. Vary the style, length, and phrasing.
Output only the examples, one per line."""
        }]
    )
    return response.content[0].text.strip().split('\n')
```

## Tips
- DeBERTa-v3-large is the go-to encoder for classification with labeled data
- For low-resource / zero-shot: use instruction-tuned decoder (Llama-3, Qwen2.5)
- Always try pseudo-labeling on unlabeled test data for semi-supervised boost
- Entity linking (SNOMED competition): dictionary-based approach beat LLMs!
- Check if competition has external data restriction before using synthetic data
