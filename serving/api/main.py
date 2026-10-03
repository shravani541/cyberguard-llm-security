"""
FastAPI Inference Server — CyberGuard Security Analyst
=======================================================
Wraps the vLLM inference engine in a REST API.
Supports: single analysis, batch analysis, streaming SSE

Endpoints:
  POST /analyze          — Analyze one code snippet
  POST /analyze/batch    — Analyze multiple snippets
  GET  /analyze/stream   — Server-Sent Events streaming
  GET  /health           — Health check
  GET  /model/info       — Model metadata
"""

import asyncio
import json
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTIL", "0.85"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cyberguard-api")

# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are CyberGuard, an elite AI code security analyst.
When given code, you:
1. Identify ALL security vulnerabilities with precision
2. Explain the real-world IMPACT (what an attacker can do)
3. Provide a complete SECURE REWRITE
4. Reference CWE IDs and OWASP categories
5. Teach the developer WHY the fix works

Be concise, technical, and actionable. Always provide working code in fixes."""


# ─────────────────────────────────────────────
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    ModelInfoResponse,
    VulnerabilityInfo,
)


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────

llm_engine: Optional[LLM] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, clean up at shutdown."""
    global llm_engine
    logger.info(f"Loading model from {MODEL_PATH}...")
    llm_engine = LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="float16",
        trust_remote_code=True,
        # vLLM quantization: works with AWQ/GPTQ quantized models
        # quantization="awq",  # Uncomment if using AWQ quantized model
    )
    logger.info("✅ Model loaded and ready!")
    yield
    # Cleanup
    del llm_engine
    logger.info("Model unloaded.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="CyberGuard — AI Code Security Analyst",
    description=(
        "Fine-tuned Llama 3.2-3B model specialized in identifying and explaining "
        "code vulnerabilities (SQL injection, XSS, command injection, etc.) "
        "with OWASP references and secure code rewrites."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def format_prompt(code: str, language: str) -> str:
    """Build the full Llama-3 chat prompt."""
    user_content = (
        f"Analyze this {language} code for security vulnerabilities:\n\n"
        f"```{language.lower()}\n{code}\n```\n\n"
        f"Provide: vulnerability type, affected line, impact, and a secure rewrite."
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_vulnerabilities(text: str) -> list[VulnerabilityInfo]:
    """Simple regex-based extraction of structured vulnerability info."""
    import re
    vulns = []

    # Try to extract CWE IDs
    cwe_matches = re.findall(r"CWE-\d+", text, re.IGNORECASE)
    vuln_types = re.findall(
        r"(SQL Injection|XSS|Cross-Site Scripting|Command Injection|Path Traversal|"
        r"IDOR|Hardcoded|Deserialization|SSRF|XXE|Buffer Overflow)",
        text, re.IGNORECASE
    )
    severities = re.findall(r"\b(CRITICAL|HIGH|MEDIUM|LOW)\b", text)
    owasp_matches = re.findall(r"A\d{2}:\d{4}[^,\n]*", text)

    for i in range(max(1, len(cwe_matches))):
        vulns.append(VulnerabilityInfo(
            vuln_type=vuln_types[i] if i < len(vuln_types) else None,
            cwe_id=cwe_matches[i] if i < len(cwe_matches) else None,
            severity=severities[0] if severities else None,
            owasp=owasp_matches[i] if i < len(owasp_matches) else None,
        ))

    return vulns if vulns else [VulnerabilityInfo()]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy" if llm_engine else "loading",
        model_loaded=llm_engine is not None,
        model_path=MODEL_PATH,
        gpu_memory_util=GPU_MEMORY_UTILIZATION,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    return ModelInfoResponse(
        model_path=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        quantization="4-bit QLoRA merged to float16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_code(request: AnalyzeRequest):
    """
    Analyze a single code snippet for security vulnerabilities.
    Returns structured analysis with CWE IDs, impact, and secure rewrite.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    request_id = random_uuid()
    prompt = format_prompt(request.code, request.language)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    start = time.perf_counter()
    outputs = llm_engine.generate([prompt], sampling_params)
    latency_ms = (time.perf_counter() - start) * 1000

    output = outputs[0]
    generated_text = output.outputs[0].text.strip()
    tokens_generated = len(output.outputs[0].token_ids)

    vulnerabilities = extract_vulnerabilities(generated_text)

    logger.info(
        f"[{request_id[:8]}] Analyzed {request.language} code | "
        f"{tokens_generated} tokens | {latency_ms:.1f}ms"
    )

    return AnalyzeResponse(
        request_id=request_id,
        model=os.path.basename(MODEL_PATH),
        language=request.language,
        analysis=generated_text,
        vulnerabilities_detected=vulnerabilities,
        latency_ms=round(latency_ms, 2),
        tokens_generated=tokens_generated,
    )


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["Analysis"])
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze multiple code snippets in one efficient batched call.
    vLLM handles batching internally for maximum GPU utilization.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if len(request.samples) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.samples)} exceeds maximum {MAX_BATCH_SIZE}"
        )

    prompts = [format_prompt(s.code, s.language) for s in request.samples]
    sampling_params_list = [
        SamplingParams(
            temperature=s.temperature,
            top_p=s.top_p,
            max_tokens=s.max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        for s in request.samples
    ]

    # vLLM: all prompts in a single forward pass for efficiency
    start = time.perf_counter()
    all_outputs = llm_engine.generate(prompts, sampling_params_list[0])  # shared params
    total_latency = (time.perf_counter() - start) * 1000

    results = []
    for i, (output, sample) in enumerate(zip(all_outputs, request.samples)):
        generated = output.outputs[0].text.strip()
        tokens = len(output.outputs[0].token_ids)
        results.append(AnalyzeResponse(
            request_id=random_uuid(),
            model=os.path.basename(MODEL_PATH),
            language=sample.language,
            analysis=generated,
            vulnerabilities_detected=extract_vulnerabilities(generated),
            latency_ms=round(total_latency / len(request.samples), 2),
            tokens_generated=tokens,
        ))

    return BatchAnalyzeResponse(
        results=results,
        total_latency_ms=round(total_latency, 2),
        batch_size=len(request.samples),
    )


@app.get("/analyze/stream", tags=["Analysis"])
async def analyze_stream(
    code: str,
    language: str = "Python",
    max_tokens: int = 512,
):
    """
    Stream the security analysis token-by-token using Server-Sent Events.
    Connect with EventSource for real-time output.
    """
    if not llm_engine:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = format_prompt(code, language)

    async def token_generator() -> AsyncGenerator[str, None]:
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
        )
        # vLLM async streaming
        async for output in await asyncio.to_thread(
            llm_engine.generate, [prompt], sampling_params
        ):
            token = output.outputs[0].text
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0)  # Yield control

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,          # Disable reload in production
    )
