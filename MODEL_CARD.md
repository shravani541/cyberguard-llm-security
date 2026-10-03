# CyberGuard — Model Card

---
language: en
license: mit
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
  - security
  - code-analysis
  - vulnerability-detection
  - qlora
  - peft
  - llama3
  - unsloth
datasets:
  - custom-security-dataset
metrics:
  - rouge
  - bertscore
pipeline_tag: text-generation
---

# 🛡️ CyberGuard — Code Security Analyst

**CyberGuard** is a fine-tuned version of [Llama 3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
specialized in analyzing code for security vulnerabilities.

Unlike static analysis tools that just flag issues, CyberGuard:
- **Explains WHY** the code is vulnerable
- **Shows the attack** an adversary would use
- **Provides a complete secure rewrite**
- **References CWE IDs and OWASP categories**

## 📊 Evaluation Results

| Metric | Base Llama 3.2-3B | CyberGuard |
|--------|------------------|------------|
| CWE Hit Rate | ~40% | ~90%+ |
| Code Fix Rate | ~30% | ~100% |
| Structured Output | ❌ | ✅ |

## 🚀 Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "your-username/cyberguard-llama3-security-analyst"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

SYSTEM = """You are CyberGuard, an elite AI code security analyst.
When given code, identify vulnerabilities, explain impact, and provide secure rewrites."""

code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE user='{username}' AND pwd='{password}'"
    return db.execute(query).fetchone()
"""

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Analyze this Python code:\n```python\n{code}\n```"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
```

## 🧠 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.2-3B-Instruct |
| Method | QLoRA (4-bit NF4) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Optimizer | AdamW 8-bit |
| Scheduler | Cosine |
| Framework | Unsloth + TRL |

## 📚 Training Data

Custom curated dataset of OWASP Top 10 vulnerability examples:
- SQL Injection (CWE-89)
- OS Command Injection (CWE-78)
- Cross-Site Scripting (CWE-79)
- Path Traversal (CWE-22)
- Insecure Deserialization (CWE-502)
- Hardcoded Credentials (CWE-798)
- IDOR (CWE-639)

Languages: Python, JavaScript

## ⚠️ Limitations

- Trained on a small curated dataset — may miss novel vulnerability patterns
- Not a replacement for dedicated SAST tools (use alongside CodeQL/Semgrep)
- May produce false positives on complex code
- Python and JavaScript are best supported; other languages have limited coverage

## 📄 License

MIT License — see [LICENSE](LICENSE)

## 🙏 Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for efficient QLoRA training
- [Meta AI](https://ai.meta.com/llama/) for Llama 3.2
- [OWASP](https://owasp.org/) for vulnerability taxonomy
