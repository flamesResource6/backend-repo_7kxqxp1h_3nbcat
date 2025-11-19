import os
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from database import db

# App setup
app = FastAPI(title="AI Website Builder API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth setup
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    plan: str = Field("free", pattern="^(free|premium|admin)$")

class LoginRequest(BaseModel):
    username: str
    password: str

class UserPublic(BaseModel):
    id: str
    name: str
    email: str
    plan: str

class ProjectFile(BaseModel):
    path: str
    content: str

class ProjectCreate(BaseModel):
    prompt: str
    framework: str = Field(..., pattern="^(react|next)$")
    database: str = Field(..., pattern="^(mongodb|postgres|mysql|sqlite|firebase|supabase)$")
    template: Optional[str] = None

class Project(BaseModel):
    id: str
    user_id: str
    prompt: str
    framework: str
    database: str
    template: Optional[str]
    premium: bool
    status: str
    created_at: datetime
    updated_at: datetime
    analytics: Dict[str, Any] = {}
    files: Dict[str, List[ProjectFile]] = {}
    deployment_url: Optional[str] = None
    build_logs: List[str] = []
    version: int = 1

# Utility functions

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Custom token extractor to support header or `?token=` query param
async def get_token_from_request(request: Request) -> str:
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1]
    token = request.query_params.get("token")
    if token:
        return token
    raise HTTPException(status_code=401, detail="Not authenticated")


def get_current_user(token: str = Depends(get_token_from_request)) -> dict:
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # Try ObjectId and string id
    user = None
    try:
        from bson import ObjectId
        user = db["user"].find_one({"_id": ObjectId(user_id)})
    except Exception:
        user = None
    if not user:
        user = db["user"].find_one({"_id": user_id})
    if not user:
        raise credentials_exception
    user["id"] = str(user.get("_id", ""))
    return user


def is_premium(user: dict) -> bool:
    return user.get("plan") in {"premium", "admin"}


def daily_generation_count(user_id: str) -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return db["generation"].count_documents({"user_id": user_id, "created_at": {"$gte": start}})


# Simple template generators

def generate_react_frontend(prompt: str, premium: bool) -> Dict[str, str]:
    comments = "// Premium documentation generated\n" if premium else ""
    index_html = """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
    <title>Generated App</title><script defer src="/main.js"></script></head><body><div id="root"></div></body></html>"""
    app_js = f"""
{comments}import React from 'react';
import ReactDOM from 'react-dom/client';
function App() {{
  return (
    <div style={{fontFamily:'Inter, sans-serif',padding:20}}>
      <h1>Generated React App</h1>
      <p>Prompt: {prompt}</p>
      <p>This app was generated automatically.</p>
    </div>
  );
}}
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
"""
    return {"index.html": index_html, "main.js": app_js}


def generate_next_frontend(prompt: str, premium: bool) -> Dict[str, str]:
    comments = "// Premium documentation generated\n" if premium else ""
    page_js = f"""
{comments}export default function Home() {{
  return (
    <main style={{fontFamily:'Inter, sans-serif',padding:20}}>
      <h1>Generated Next.js App</h1>
      <p>Prompt: {prompt}</p>
      <p>This app was generated automatically.</p>
    </main>
  );
}}
"""
    return {"pages/index.js": page_js}


def generate_fastapi_backend(prompt: str, premium: bool) -> Dict[str, str]:
    comments = "# Premium documentation generated\n" if premium else ""
    main_py = f"""
{comments}from fastapi import FastAPI
app = FastAPI()
@app.get('/')
def root():
    return {{'message':'Generated backend for: {prompt}'}}
"""
    return {"main.py": main_py}


def generate_db_schema(db_choice: str, premium: bool) -> Dict[str, str]:
    if db_choice == "mongodb":
        schema = """from pydantic import BaseModel\nclass Item(BaseModel):\n    title: str\n    description: str | None = None\n"""
    elif db_choice == "postgres":
        schema = """CREATE TABLE items (id SERIAL PRIMARY KEY, title TEXT NOT NULL, description TEXT);\n"""
    elif db_choice == "mysql":
        schema = """CREATE TABLE items (id INT AUTO_INCREMENT PRIMARY KEY, title VARCHAR(255) NOT NULL, description TEXT);\n"""
    elif db_choice == "sqlite":
        schema = """CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, description TEXT);\n"""
    elif db_choice == "firebase":
        schema = """// Firestore rules and collection structure for items\n"""
    else:  # supabase
        schema = """-- SQL for Supabase: items table\nCREATE TABLE items (id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, title TEXT NOT NULL, description TEXT);\n"""
    if premium:
        schema = f"""-- Premium schema notes\n{schema}"""
    return {"schema": schema}


def synthesize_project(prompt: str, framework: str, db_choice: str, premium: bool) -> Dict[str, Any]:
    # Assemble code files for a simple but working scaffold
    if framework == "react":
        fe = generate_react_frontend(prompt, premium)
    else:
        fe = generate_next_frontend(prompt, premium)
    be = generate_fastapi_backend(prompt, premium)
    db_files = generate_db_schema(db_choice, premium)

    pages = len(fe)
    analytics = {
        "pages": pages,
        "stack": {"frontend": framework, "backend": "fastapi", "database": db_choice},
    }
    return {
        "files": {
            "frontend": [ProjectFile(path=k, content=v) for k, v in fe.items()],
            "backend": [ProjectFile(path=k, content=v) for k, v in be.items()],
            "database": [ProjectFile(path=k, content=v) for k, v in db_files.items()],
        },
        "analytics": analytics,
    }


# Auth routes
@app.post("/auth/register", response_model=UserPublic)
def register(user: UserCreate):
    existing = db["user"].find_one({"email": user.email.lower()})
    if existing:
        raise HTTPException(400, "Email already registered")
    doc = {
        "name": user.name,
        "email": user.email.lower(),
        "password": get_password_hash(user.password),
        "plan": user.plan,
        "created_at": datetime.now(timezone.utc),
    }
    res = db["user"].insert_one(doc)
    return UserPublic(id=str(res.inserted_id), name=doc["name"], email=doc["email"], plan=doc["plan"])


@app.post("/auth/login", response_model=Token)
def login(payload: LoginRequest):
    user = db["user"].find_one({"email": payload.username.lower()})
    if not user or not verify_password(payload.password, user.get("password", "")):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": str(user["_id"])})
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserPublic)
def me(current=Depends(get_current_user)):
    return UserPublic(id=str(current.get("_id", current.get("id", ""))), name=current["name"], email=current["email"], plan=current.get("plan", "free"))


# Upload and PDF text extraction
@app.post("/files/upload")
def upload_file(file: UploadFile = File(...), current=Depends(get_current_user)):
    content = file.file.read()
    text = None
    if file.filename.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        except Exception:
            text = None
    doc = {
        "user_id": str(current.get("_id", current.get("id", ""))),
        "filename": file.filename,
        "size": len(content),
        "type": file.content_type,
        "text": text,
        "created_at": datetime.now(timezone.utc),
    }
    db["upload"].insert_one(doc)
    return {"ok": True, "text_preview": (text or "").strip()[:1000]}


# AI generation
@app.post("/ai/generate")
def ai_generate(payload: ProjectCreate, current=Depends(get_current_user)):
    premium = is_premium(current)
    # Free user limit
    if not premium:
        count = daily_generation_count(str(current.get("_id", current.get("id", ""))))
        if count >= 3:
            raise HTTPException(429, "Daily generation limit reached. Upgrade to premium for unlimited generations.")
    # Priority processing
    if not premium:
        time.sleep(1.5)

    synthesis = synthesize_project(payload.prompt, payload.framework, payload.database, premium)
    now = datetime.now(timezone.utc)
    user_id = str(current.get("_id", current.get("id", "")))
    proj_doc = {
        "user_id": user_id,
        "prompt": payload.prompt,
        "framework": payload.framework,
        "database": payload.database,
        "template": payload.template,
        "premium": premium,
        "status": "ready",
        "created_at": now,
        "updated_at": now,
        "analytics": synthesis["analytics"],
        "files": {
            "frontend": [f.model_dump() for f in synthesis["files"]["frontend"]],
            "backend": [f.model_dump() for f in synthesis["files"]["backend"]],
            "database": [f.model_dump() for f in synthesis["files"]["database"]],
        },
        "version": 1,
        "build_logs": [f"Generated at {now.isoformat()}"]
    }
    res = db["project"].insert_one(proj_doc)
    db["generation"].insert_one({"user_id": user_id, "project_id": str(res.inserted_id), "created_at": now})
    proj_doc["id"] = str(res.inserted_id)
    return proj_doc


# Projects CRUD and dashboard
@app.get("/projects")
def list_projects(q: Optional[str] = None, sort: Optional[str] = "-created_at", current=Depends(get_current_user)):
    user_id = str(current.get("_id", current.get("id", "")))
    query: Dict[str, Any] = {"user_id": user_id}
    if q:
        query["prompt"] = {"$regex": q, "$options": "i"}
    sort_field = (sort or "-created_at")
    direction = -1 if sort_field.startswith("-") else 1
    field = sort_field[1:] if sort_field.startswith("-") else sort_field
    cursor = db["project"].find(query).sort(field, direction)
    items = []
    for p in cursor:
        p["id"] = str(p["_id"]) 
        items.append(p)
    return {"items": items}


@app.get("/projects/{project_id}")
def get_project(project_id: str, current=Depends(get_current_user)):
    from bson import ObjectId
    user_id = str(current.get("_id", current.get("id", "")))
    p = db["project"].find_one({"_id": ObjectId(project_id), "user_id": user_id})
    if not p:
        raise HTTPException(404, "Project not found")
    p["id"] = str(p["_id"]) 
    return p


class UpdateFiles(BaseModel):
    frontend: Optional[List[ProjectFile]] = None
    backend: Optional[List[ProjectFile]] = None
    database: Optional[List[ProjectFile]] = None


@app.post("/projects/{project_id}/rebuild")
def rebuild_project(project_id: str, changes: UpdateFiles, current=Depends(get_current_user)):
    from bson import ObjectId
    user_id = str(current.get("_id", current.get("id", "")))
    p = db["project"].find_one({"_id": ObjectId(project_id), "user_id": user_id})
    if not p:
        raise HTTPException(404, "Project not found")
    files = p.get("files", {})
    if changes.frontend is not None:
        files["frontend"] = [f.model_dump() for f in changes.frontend]
    if changes.backend is not None:
        files["backend"] = [f.model_dump() for f in changes.backend]
    if changes.database is not None:
        files["database"] = [f.model_dump() for f in changes.database]
    version = p.get("version", 1) + 1
    now = datetime.now(timezone.utc)
    db["project"].update_one({"_id": p["_id"]}, {"$set": {"files": files, "updated_at": now, "version": version}, "$push": {"build_logs": f"Rebuilt at {now.isoformat()}"}})
    return {"ok": True, "version": version}


@app.get("/projects/{project_id}/download")
async def download_zip(project_id: str, current=Depends(get_current_user)):
    from bson import ObjectId
    user_id = str(current.get("_id", current.get("id", "")))
    p = db["project"].find_one({"_id": ObjectId(project_id), "user_id": user_id})
    if not p:
        raise HTTPException(404, "Project not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        # frontend
        for f in p["files"].get("frontend", []):
            z.writestr(f"generated/frontend/{f['path']}", f["content"].encode("utf-8"))
        # backend
        for f in p["files"].get("backend", []):
            z.writestr(f"generated/backend/{f['path']}", f["content"].encode("utf-8"))
        # database
        for f in p["files"].get("database", []):
            z.writestr(f"generated/database/{f['path']}", f["content"].encode("utf-8"))
        # meta
        z.writestr("generated/README.txt", f"Generated from prompt: {p['prompt']}\nStack: {p['framework']} + FastAPI + {p['database']}\n")
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    filename = f"project_{project_id}.zip"
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/projects/{project_id}/deploy")
def deploy_project(project_id: str, current=Depends(get_current_user)):
    if not is_premium(current):
        raise HTTPException(403, "Deployment is a premium feature")
    from bson import ObjectId
    user_id = str(current.get("_id", current.get("id", "")))
    p = db["project"].find_one({"_id": ObjectId(project_id), "user_id": user_id})
    if not p:
        raise HTTPException(404, "Project not found")
    # Simulated deployment URL
    url = f"https://deploy.example.com/{project_id}"
    now = datetime.now(timezone.utc)
    db["project"].update_one({"_id": p["_id"]}, {"$set": {"deployment_url": url, "updated_at": now}, "$push": {"build_logs": f"Deployed at {now.isoformat()} to {url}"}})
    db["deployment"].insert_one({"project_id": project_id, "user_id": user_id, "url": url, "created_at": now})
    return {"ok": True, "url": url}


# Billing (placeholder)
@app.post("/billing/checkout")
def checkout(provider: str = Form("stripe"), plan: str = Form("premium"), current=Depends(get_current_user)):
    if plan != "premium":
        raise HTTPException(400, "Only premium plan supported here")
    # Simulate session url
    session_url = f"https://billing.example.com/checkout?provider={provider}&user={current.get('_id', current.get('id',''))}"
    return {"checkout_url": session_url}


# Analytics
@app.get("/analytics/summary")
def analytics_summary(current=Depends(get_current_user)):
    total_projects = db["project"].count_documents({"user_id": str(current.get("_id", current.get("id", "")))})
    pages = 0
    users = db["user"].count_documents({}) if current.get("plan") == "admin" else None
    for p in db["project"].find({"user_id": str(current.get("_id", current.get("id", "")))}):
        analytics = p.get("analytics", {})
        pages += analytics.get("pages", 0)
    return {"projects": total_projects, "pages": pages, "users": users}


# Helper endpoints
@app.get("/")
def root():
    return {"message": "AI Website Builder API is running"}


@app.get("/test")
def test_database():
    ok = bool(db)
    collections = []
    try:
        collections = db.list_collection_names()[:10] if db else []
    except Exception:
        collections = []
    return {"backend": "ok", "database": "ok" if ok else "missing", "collections": collections}
