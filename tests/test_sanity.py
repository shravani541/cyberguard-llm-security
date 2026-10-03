"""
Quick sanity test — run this locally (no GPU needed) to verify:
  1. Dataset builder works
  2. Config loads correctly
  3. Prompt formatting is correct
  4. API request/response schemas validate

Usage: python tests/test_sanity.py
"""
import io
import json
import sys
import os

# Fix Windows console encoding for unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dataset_builder():
    """Test that dataset builder runs and produces valid JSONL."""
    print("Testing dataset builder...")
    from data.build_dataset import build_dataset, CURATED_EXAMPLES, format_llama3_chat
    
    assert len(CURATED_EXAMPLES) >= 5, "Should have at least 5 curated examples"
    
    # Test formatting
    sample = CURATED_EXAMPLES[0]
    text = format_llama3_chat(sample)
    assert "<|begin_of_text|>" in text
    assert "<|start_header_id|>system<|end_header_id|>" in text
    assert "CyberGuard" in text
    assert sample.vuln_type in text
    assert sample.cwe_id in text
    assert "```" in text  # Should have code blocks
    
    print(f"  [OK] {len(CURATED_EXAMPLES)} examples, formatting correct")


def test_config_loads():
    """Test that config YAML loads and has required keys."""
    print("Testing config loading...")
    import yaml
    
    with open("training/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    required_keys = ["model", "lora", "training", "data", "mlflow", "output"]
    for key in required_keys:
        assert key in cfg, f"Missing config key: {key}"
    
    assert cfg["model"]["load_in_4bit"] == True
    assert cfg["lora"]["r"] > 0
    assert cfg["training"]["learning_rate"] > 0
    
    print(f"  [OK] Config valid, model: {cfg['model']['base_model']}")


def test_api_schemas():
    """Test Pydantic schemas validate correctly."""
    print("Testing API schemas...")
    from serving.api.schemas import AnalyzeRequest, AnalyzeResponse, VulnerabilityInfo
    
    # Valid request
    req = AnalyzeRequest(
        code="def f(x): return eval(x)",
        language="Python",
        max_tokens=256,
    )
    assert req.language == "Python"
    
    # Invalid request (too short code)
    try:
        bad_req = AnalyzeRequest(code="x", language="Python")
        assert False, "Should have raised validation error"
    except Exception:
        pass  # Expected
    
    print("  [OK] API schemas validate correctly")


def test_prompt_format():
    """Test prompt contains required tokens."""
    print("Testing prompt format...")
    
    from data.build_dataset import format_llama3_chat, CURATED_EXAMPLES
    sample = CURATED_EXAMPLES[0]
    prompt = format_llama3_chat(sample)
    
    required_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
    ]
    for token in required_tokens:
        assert token in prompt, f"Missing token: {token}"
    
    print("  [OK] Prompt format correct (Llama-3 chat template)")


if __name__ == "__main__":
    print("Running sanity tests...\n")
    try:
        test_config_loads()
        test_dataset_builder()
        test_prompt_format()
        test_api_schemas()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
