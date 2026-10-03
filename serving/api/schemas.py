"""
Pydantic schemas for CyberGuard API requests and responses.
"""
from typing import Optional
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    code: str = Field(..., description="Code snippet to analyze", min_length=10)
    language: str = Field(default="Python", description="Programming language")
    max_tokens: int = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    model_config = {"json_schema_extra": {
        "example": {
            "code": "def login(user, pwd):\n    q = f\"SELECT * FROM users WHERE user='{user}' AND pwd='{pwd}'\"\n    return db.execute(q).fetchone()",
            "language": "Python",
            "max_tokens": 512,
        }
    }}

class VulnerabilityInfo(BaseModel):
    vuln_type: Optional[str] = None
    cwe_id: Optional[str] = None
    severity: Optional[str] = None
    owasp: Optional[str] = None

class AnalyzeResponse(BaseModel):
    request_id: str
    model: str
    language: str
    analysis: str
    vulnerabilities_detected: list[VulnerabilityInfo]
    latency_ms: float
    tokens_generated: int

class BatchAnalyzeRequest(BaseModel):
    samples: list[AnalyzeRequest] = Field(..., max_length=8)

class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
    total_latency_ms: float
    batch_size: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    gpu_memory_util: float

class ModelInfoResponse(BaseModel):
    model_path: str
    max_model_len: int
    quantization: str
    tensor_parallel_size: int
