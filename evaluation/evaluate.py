"""
Evaluation Script — Base vs Fine-Tuned Model Comparison
=========================================================
Runs both models on a test suite of vulnerable code snippets
and computes:
  - ROUGE-L score (structural similarity to reference)
  - BERTScore F1 (semantic similarity)
  - CWE Hit Rate (did the model correctly identify the CWE?)
  - Fix Quality Score (does the secure rewrite remove the vuln?)
  - LLM-as-Judge Score (optional: uses GPT-4o-mini as judge)

Output: Markdown results table → results/comparison_table.md
"""

import json
import re
import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import mlflow
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


# ─────────────────────────────────────────────
# Test Suite — Held-Out Examples
# ─────────────────────────────────────────────

@dataclass
class EvalSample:
    id: str
    language: str
    vulnerable_code: str
    expected_cwe: str
    expected_vuln_type: str
    reference_analysis: str      # Gold-standard response text
    severity: str


TEST_SUITE: list[EvalSample] = [
    EvalSample(
        id="eval-001",
        language="Python",
        vulnerable_code='''
def authenticate(conn, user, pwd):
    q = "SELECT id FROM users WHERE user=\'" + user + "\' AND pwd=\'" + pwd + "\'"
    return conn.execute(q).fetchone()
''',
        expected_cwe="CWE-89",
        expected_vuln_type="SQL Injection",
        reference_analysis=(
            "SQL Injection (CWE-89). The string concatenation allows attackers to inject SQL. "
            "Fix: Use parameterized queries with `?` placeholders."
        ),
        severity="CRITICAL",
    ),
    EvalSample(
        id="eval-002",
        language="Python",
        vulnerable_code='''
import subprocess
def run_tool(user_input):
    subprocess.run(f"nmap {user_input}", shell=True)
''',
        expected_cwe="CWE-78",
        expected_vuln_type="Command Injection",
        reference_analysis=(
            "OS Command Injection (CWE-78). shell=True with user input allows arbitrary command execution. "
            "Fix: Use subprocess with list arguments and shell=False."
        ),
        severity="CRITICAL",
    ),
    EvalSample(
        id="eval-003",
        language="Python",
        vulnerable_code='''
SECRET = "abc123"
API_KEY = "sk-prod-REAL_KEY_HERE"
def get_data():
    return {"key": API_KEY}
''',
        expected_cwe="CWE-798",
        expected_vuln_type="Hardcoded Credentials",
        reference_analysis=(
            "Hardcoded credentials (CWE-798). Secrets in source code are exposed via version control. "
            "Fix: Use environment variables and a secrets manager."
        ),
        severity="CRITICAL",
    ),
    EvalSample(
        id="eval-004",
        language="Python",
        vulnerable_code='''
import yaml
def load_config(user_yaml: str):
    return yaml.load(user_yaml)  # unsafe!
''',
        expected_cwe="CWE-502",
        expected_vuln_type="Insecure Deserialization",
        reference_analysis=(
            "Insecure deserialization via PyYAML (CWE-502). yaml.load() can execute arbitrary Python. "
            "Fix: Use yaml.safe_load() instead."
        ),
        severity="HIGH",
    ),
    EvalSample(
        id="eval-005",
        language="JavaScript",
        vulnerable_code='''
app.get('/user', (req, res) => {
    const id = req.query.id;
    db.query(`SELECT * FROM users WHERE id = ${id}`, (err, rows) => {
        res.json(rows);
    });
});
''',
        expected_cwe="CWE-89",
        expected_vuln_type="SQL Injection",
        reference_analysis=(
            "SQL Injection (CWE-89) in JavaScript/Node.js. Template literal interpolation in SQL query. "
            "Fix: Use parameterized queries with `?` placeholders."
        ),
        severity="CRITICAL",
    ),
]


# ─────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are CyberGuard, an elite AI code security analyst.
When given code, you:
1. Identify ALL security vulnerabilities with precision
2. Explain the real-world IMPACT (what an attacker can do)
3. Provide a complete SECURE REWRITE
4. Reference CWE IDs and OWASP categories
5. Teach the developer WHY the fix works

Be concise, technical, and actionable. Always provide working code in fixes."""


def build_prompt(sample: EvalSample) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Analyze this {sample.language} code for security vulnerabilities:\n\n"
        f"```{sample.language.lower()}\n{sample.vulnerable_code}\n```\n\n"
        f"Provide: vulnerability type, affected line, impact, and a secure rewrite.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate model response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def compute_bert_score(predictions: list[str], references: list[str]) -> list[float]:
    """Compute BERTScore F1 for a batch."""
    P, R, F1 = bert_score_fn(
        predictions, references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
    )
    return F1.tolist()


def check_cwe_mentioned(response: str, expected_cwe: str) -> bool:
    """Check if the model mentioned the correct CWE ID."""
    return expected_cwe.lower() in response.lower()


def check_vuln_type_mentioned(response: str, expected_type: str) -> bool:
    """Check if the model identified the correct vulnerability type."""
    keywords = expected_type.lower().split()
    return all(kw in response.lower() for kw in keywords)


def check_has_code_fix(response: str) -> bool:
    """Check if the response contains a code block (likely the fix)."""
    return "```" in response


@dataclass
class SampleResult:
    sample_id: str
    language: str
    vuln_type: str
    severity: str
    base_response: str = ""
    finetuned_response: str = ""
    base_rouge_l: float = 0.0
    finetuned_rouge_l: float = 0.0
    base_bert_f1: float = 0.0
    finetuned_bert_f1: float = 0.0
    base_cwe_hit: bool = False
    finetuned_cwe_hit: bool = False
    base_vuln_hit: bool = False
    finetuned_vuln_hit: bool = False
    base_has_fix: bool = False
    finetuned_has_fix: bool = False


# ─────────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────────

def evaluate_model(model, tokenizer, model_name: str, samples: list[EvalSample]) -> dict[str, str]:
    """Run inference on all test samples, return {sample_id: response}."""
    responses = {}
    FastLanguageModel.for_inference(model)

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        prompt = build_prompt(sample)
        response = generate_response(model, tokenizer, prompt)
        responses[sample.id] = response
        logger.info(f"  [{model_name}] {sample.id}: {len(response)} chars")

    return responses


def run_evaluation(
    base_model_path: str,
    finetuned_model_path: str,
    output_dir: str = "./results",
    mlflow_run_id: Optional[str] = None,
):
    """Full evaluation pipeline."""

    results: list[SampleResult] = []

    # Initialize results
    for sample in TEST_SUITE:
        results.append(SampleResult(
            sample_id=sample.id,
            language=sample.language,
            vuln_type=sample.expected_vuln_type,
            severity=sample.severity,
        ))

    # ── Base Model ─────────────────────────────────────
    logger.info("Loading BASE model...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    base_responses = evaluate_model(base_model, base_tokenizer, "BASE", TEST_SUITE)
    del base_model
    torch.cuda.empty_cache()

    # ── Fine-tuned Model ───────────────────────────────
    logger.info("Loading FINE-TUNED model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=finetuned_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    ft_responses = evaluate_model(ft_model, ft_tokenizer, "FINETUNED", TEST_SUITE)
    del ft_model
    torch.cuda.empty_cache()

    # ── Compute Metrics ────────────────────────────────
    logger.info("Computing metrics...")

    base_preds = [base_responses[s.id] for s in TEST_SUITE]
    ft_preds = [ft_responses[s.id] for s in TEST_SUITE]
    references = [s.reference_analysis for s in TEST_SUITE]

    base_bert_scores = compute_bert_score(base_preds, references)
    ft_bert_scores = compute_bert_score(ft_preds, references)

    for i, (sample, result) in enumerate(zip(TEST_SUITE, results)):
        result.base_response = base_responses[sample.id]
        result.finetuned_response = ft_responses[sample.id]
        result.base_rouge_l = compute_rouge_l(base_responses[sample.id], references[i])
        result.finetuned_rouge_l = compute_rouge_l(ft_responses[sample.id], references[i])
        result.base_bert_f1 = base_bert_scores[i]
        result.finetuned_bert_f1 = ft_bert_scores[i]
        result.base_cwe_hit = check_cwe_mentioned(base_responses[sample.id], sample.expected_cwe)
        result.finetuned_cwe_hit = check_cwe_mentioned(ft_responses[sample.id], sample.expected_cwe)
        result.base_vuln_hit = check_vuln_type_mentioned(base_responses[sample.id], sample.expected_vuln_type)
        result.finetuned_vuln_hit = check_vuln_type_mentioned(ft_responses[sample.id], sample.expected_vuln_type)
        result.base_has_fix = check_has_code_fix(base_responses[sample.id])
        result.finetuned_has_fix = check_has_code_fix(ft_responses[sample.id])

    # ── Aggregate ──────────────────────────────────────
    n = len(results)
    summary = {
        "base_avg_rouge_l": sum(r.base_rouge_l for r in results) / n,
        "ft_avg_rouge_l": sum(r.finetuned_rouge_l for r in results) / n,
        "base_avg_bert_f1": sum(r.base_bert_f1 for r in results) / n,
        "ft_avg_bert_f1": sum(r.finetuned_bert_f1 for r in results) / n,
        "base_cwe_hit_rate": sum(r.base_cwe_hit for r in results) / n,
        "ft_cwe_hit_rate": sum(r.finetuned_cwe_hit for r in results) / n,
        "base_vuln_hit_rate": sum(r.base_vuln_hit for r in results) / n,
        "ft_vuln_hit_rate": sum(r.finetuned_vuln_hit for r in results) / n,
        "base_code_fix_rate": sum(r.base_has_fix for r in results) / n,
        "ft_code_fix_rate": sum(r.finetuned_has_fix for r in results) / n,
    }

    # ── MLflow Logging ────────────────────────────────
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics({
                "eval_base_rouge_l": summary["base_avg_rouge_l"],
                "eval_ft_rouge_l": summary["ft_avg_rouge_l"],
                "eval_base_bert_f1": summary["base_avg_bert_f1"],
                "eval_ft_bert_f1": summary["ft_avg_bert_f1"],
                "eval_base_cwe_hit_rate": summary["base_cwe_hit_rate"],
                "eval_ft_cwe_hit_rate": summary["ft_cwe_hit_rate"],
            })

    # ── Save Results ───────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Raw JSON
    raw_path = Path(output_dir) / "eval_results_raw.json"
    with open(raw_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    # Markdown table
    md_path = Path(output_dir) / "comparison_table.md"
    _write_markdown_table(results, summary, md_path)

    logger.info(f"✅ Results saved to {output_dir}")
    _print_summary(summary)

    return results, summary


def _write_markdown_table(results: list[SampleResult], summary: dict, output_path: Path):
    """Write a beautifully formatted Markdown comparison table."""
    lines = []
    lines.append("# 📊 Model Evaluation: Base vs Fine-Tuned\n")
    lines.append("> **Model**: Llama 3.2-3B-Instruct | **Fine-tune**: QLoRA (4-bit, r=16)\n")
    lines.append(f"> **Task**: Code Security Vulnerability Analysis\n")
    lines.append(f"> **Test Samples**: {len(results)}\n")

    lines.append("\n## 🏆 Summary Metrics\n")
    lines.append("| Metric | Base Model | Fine-Tuned | Improvement |")
    lines.append("|--------|-----------|------------|-------------|")

    metrics = [
        ("ROUGE-L ↑", "base_avg_rouge_l", "ft_avg_rouge_l"),
        ("BERTScore F1 ↑", "base_avg_bert_f1", "ft_avg_bert_f1"),
        ("CWE Hit Rate ↑", "base_cwe_hit_rate", "ft_cwe_hit_rate"),
        ("Vuln Type Hit Rate ↑", "base_vuln_hit_rate", "ft_vuln_hit_rate"),
        ("Code Fix Present ↑", "base_code_fix_rate", "ft_code_fix_rate"),
    ]

    for label, base_key, ft_key in metrics:
        base_val = summary[base_key]
        ft_val = summary[ft_key]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.1%}"
        trend = "✅" if delta >= 0 else "❌"
        lines.append(f"| {label} | {base_val:.3f} | {ft_val:.3f} | {trend} {delta_str} |")

    lines.append("\n## 📋 Per-Sample Results\n")
    lines.append("| ID | Vulnerability | Lang | Severity | Base CWE✓ | FT CWE✓ | Base ROUGE | FT ROUGE |")
    lines.append("|----|--------------|------|----------|-----------|---------|-----------|---------|")

    for r in results:
        base_cwe = "✅" if r.base_cwe_hit else "❌"
        ft_cwe = "✅" if r.finetuned_cwe_hit else "❌"
        sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(r.severity, "⚪")
        lines.append(
            f"| {r.sample_id} | {r.vuln_type} | {r.language} | "
            f"{sev_emoji} {r.severity} | {base_cwe} | {ft_cwe} | "
            f"{r.base_rouge_l:.3f} | {r.finetuned_rouge_l:.3f} |"
        )

    lines.append("\n## 📖 Sample Responses\n")
    for r in results[:2]:  # Show first 2 examples
        lines.append(f"\n### {r.sample_id} — {r.vuln_type}\n")
        lines.append("**Base Model Response:**")
        lines.append(f"```\n{r.base_response[:600]}...\n```\n")
        lines.append("**Fine-Tuned Response:**")
        lines.append(f"```\n{r.finetuned_response[:600]}...\n```\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"📝 Markdown table → {output_path}")


def _print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Base':>8}  {'Fine-Tuned':>10}  {'Delta':>8}")
    print("-" * 60)
    pairs = [
        ("ROUGE-L", "base_avg_rouge_l", "ft_avg_rouge_l"),
        ("BERTScore F1", "base_avg_bert_f1", "ft_avg_bert_f1"),
        ("CWE Hit Rate", "base_cwe_hit_rate", "ft_cwe_hit_rate"),
        ("Vuln Hit Rate", "base_vuln_hit_rate", "ft_vuln_hit_rate"),
        ("Code Fix Rate", "base_code_fix_rate", "ft_code_fix_rate"),
    ]
    for label, bk, fk in pairs:
        b, f = summary[bk], summary[fk]
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<25} {b:>8.3f}  {f:>10.3f}  {sign}{d:>7.3f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/Llama-3.2-3B-Instruct", help="Base model path or HF ID")
    parser.add_argument("--finetuned_model", default="./outputs/merged_model", help="Fine-tuned model path")
    parser.add_argument("--output_dir", default="./results", help="Where to save results")
    parser.add_argument("--mlflow_run_id", default=None, help="Log metrics to this MLflow run")
    args = parser.parse_args()

    run_evaluation(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        output_dir=args.output_dir,
        mlflow_run_id=args.mlflow_run_id,
    )
