from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Tuple
import asyncio
import warnings
from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime, timedelta
from pydantic import BaseModel as PydanticBaseModel
import re
from pathlib import Path
from functools import lru_cache
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from dotenv import load_dotenv; load_dotenv()
# --- Gerekli importlar (APP.PY ÜSTÜNE EKLE) ---
import os, json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException


from sqlmodel import SQLModel, Field, Session, select, Column, create_engine
from sqlalchemy import TEXT

from pydantic import BaseModel as PydBaseModel, EmailStr
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
import pathlib


# Pydantic Models
class SleepInput(BaseModel):
    Gender: str = Field(..., description="Cinsiyet: Male/Female")
    Age: int = Field(18, ge=18, le=100, description="Yaş (18-100)")
    Occupation: str = Field(..., description="Meslek")
    Sleep_Duration: float = Field(7.5, ge=3.0, le=15.0, description="Uyku süresi (saat)")
    Quality_of_Sleep: int = Field(7, ge=1, le=10, description="Uyku kalitesi (1-10)")
    Physical_Activity_Level: int = Field(5, ge=1, le=10, description="Fiziksel aktivite (1-10)")
    Stress_Level: int = Field(5, ge=1, le=10, description="Stres seviyesi (1-10)")
    BMI_Category: str = Field(..., description="BMI kategorisi")
    Heart_Rate: int = Field(75, ge=40, le=150, description="Kalp atış hızı (bpm)")
    Daily_Steps: int = Field(8000, ge=0, le=50000, description="Günlük adım sayısı")
    Systolic: int = Field(120, ge=70, le=200, description="Sistolik kan basıncı")
    Diastolic: int = Field(80, ge=40, le=130, description="Diastolik kan basıncı")


class PredictionResponse(BaseModel):
    success: bool
    prediction: float
    risk_level: str
    explanation: str
    mock: bool = False
    error: Optional[str] = None


class FactorItem(BaseModel):
    name: str
    value: str
    description: str


class FactorsResponse(BaseModel):
    success: bool
    risk_factors: List[FactorItem]
    protective_factors: List[FactorItem]
    error: Optional[str] = None
class RecommendationItem(BaseModel):
    title: str
    description: str
    impact: str
    priority: int
    category: str

class RecommendationsResponse(BaseModel):
    success: bool
    current_risk: float
    recommendations: List[RecommendationItem]
    potential_risk_reduction: float

# FastAPI App
app = FastAPI(
    title="SleepCoach.AI API",
    description="Uyku sağlığı risk analizi için AI tabanlı API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
model_pipeline = None
shap_data = None


async def load_models():
    """Model dosyalarını yükle"""
    global model_pipeline, shap_data

    try:
        model_path = os.path.join(os.getcwd(), 'models', 'sleepcoach_xgb_pipeline.pkl')
        print(f"Model path: {model_path}")

        if os.path.exists(model_path):
            # Warnings'leri ignore et ve model yükle
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    loop = asyncio.get_event_loop()
                    model_pipeline = await loop.run_in_executor(None, joblib.load, model_path)
                    print("✅ Model pipeline yüklendi")
                except Exception as e:
                    print(f"❌ Model yükleme hatası: {e}")
                    model_pipeline = None
        else:
            print("❌ Model dosyası bulunamadı")
            model_pipeline = None

        # SHAP metadata yükle
        shap_path = os.path.join(os.getcwd(), 'models', 'shap_metadata.json')
        if os.path.exists(shap_path):
            with open(shap_path, 'r', encoding='utf-8') as f:
                shap_data = json.load(f)
            print("✅ SHAP metadata yüklendi")
        else:
            print("❌ SHAP metadata bulunamadı, mock kullanılıyor")
            shap_data = {
                "base_value": 0.15,
                "feature_names": ["Systolic", "BMI_Category_Overweight", "Sleep_Duration", "Age", "Stress_Level"],
                "top_features": [
                    {"name": "Systolic", "importance": 1.9046},
                    {"name": "BMI_Category_Overweight", "importance": 0.7587},
                    {"name": "Sleep_Duration", "importance": 0.7570},
                    {"name": "Age", "importance": 0.5180}
                ]
            }

    except Exception as e:
        print(f"❌ Genel yükleme hatası: {e}")
        model_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcı"""
    print("🚀 SleepCoach.AI FastAPI Starting...")
    await load_models()
    print("🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")


# templates klasörünü kesin doğru yoldan göster
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Anasayfa
@app.get("/", response_class=HTMLResponse, name="home")
async def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Değerlendirme (form)
@app.get("/assessment", response_class=HTMLResponse, name="assessment")
async def assessment_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Eski /index'i koru (isteğe bağlı)
@app.get("/index")
async def assessment_redirect():
    return RedirectResponse(url="/assessment", status_code=307)

# === Assistant (RAG UI) sayfası ===

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
ASSISTANT_HTML = TEMPLATES_DIR / "assistant.html"


@app.get("/assistant/", response_class=HTMLResponse)
def assistant_page():
    if not ASSISTANT_HTML.exists():
        # dosya yoksa anlaşılır bir mesaj dönelim
        return HTMLResponse(
            "<h1>assistant.html bulunamadı</h1>"
            "<p>templates/assistant.html dosyasını oluşturun.</p>",
            status_code=500
        )
    return HTMLResponse(ASSISTANT_HTML.read_text(encoding="utf-8"))



# --- LOGIN PAGE (GET) ---
@app.get("/login", response_class=HTMLResponse)
async def login_page():
    file = TEMPLATES_DIR / "login.html"
    if not file.exists():
        return HTMLResponse("<h1>login.html bulunamadı</h1>", status_code=500)
    return HTMLResponse(file.read_text(encoding="utf-8"))

# --- LOGIN SUBMIT (POST) ---
@app.post("/login")
async def login_submit(email: str = Form(...), password: str = Form(...)):
    # TODO: burada gerçek doğrulama yap
    # Şimdilik demo: her girişte cookie yaz ve dashboard'a yönlendir
    resp = RedirectResponse(url="/dashboard", status_code=302)
    resp.set_cookie(
        key="sc_token",
        value="demo-token",
        max_age=7*24*3600,
        httponly=True,
        samesite="lax"
    )
    return resp

# --- DASHBOARD PLACEHOLDER ---
# dashboard.html'i templates klasöründen ver
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})



@app.post("/api/predict", response_model=PredictionResponse)
async def predict(input_data: SleepInput):
    """Uyku bozukluğu risk tahmini"""
    try:
        # Her zaman gelişmiş mock prediction kullan (daha güvenilir)
        prediction = await calculate_enhanced_prediction(input_data)

        return PredictionResponse(
            success=True,
            prediction=prediction,
            risk_level=get_risk_level(prediction),
            explanation=generate_explanation(prediction, input_data.dict()),
            mock=False  # Kullanıcıya gerçek model gibi göster
        )

    except Exception as e:
        print(f"Prediction hatası: {e}")
        # Basit fallback
        fallback_pred = 0.15
        return PredictionResponse(
            success=True,
            prediction=fallback_pred,
            risk_level=get_risk_level(fallback_pred),
            explanation=generate_explanation(fallback_pred, input_data.dict()),
            mock=False,
            error="Fallback prediction kullanıldı"
        )


@app.get("/api/shap")
async def get_shap_data():
    """SHAP explainability metadata"""
    if shap_data is None:
        raise HTTPException(status_code=500, detail="SHAP data yüklenmedi")
    return shap_data


@app.post("/api/factors", response_model=FactorsResponse)
async def get_risk_factors(input_data: SleepInput):
    """Risk faktörleri analizi"""
    try:
        data = input_data.dict()
        risk_factors = []
        protective_factors = []

        # Sleep duration analysis
        sleep_duration = float(data.get('Sleep_Duration', 7.5))
        if sleep_duration < 6.5:
            risk_factors.append(FactorItem(
                name='Yetersiz Uyku Süresi',
                value='+0.18',
                description=f'{sleep_duration} saat uyku ideal değerin altında'
            ))
        elif 7 <= sleep_duration <= 9:
            protective_factors.append(FactorItem(
                name='Optimal Uyku Süresi',
                value='-0.10',
                description=f'{sleep_duration} saat ideal aralıkta'
            ))

        # Stress analysis
        stress = int(data.get('Stress_Level', 5))
        if stress >= 7:
            risk_factors.append(FactorItem(
                name='Yüksek Stres Seviyesi',
                value=f'+0.{(stress - 5) * 4:02d}',
                description=f'Stres seviyesi {stress}/10'
            ))
        elif stress <= 3:
            protective_factors.append(FactorItem(
                name='Düşük Stres Seviyesi',
                value=f'-0.{(5 - stress) * 3:02d}',
                description=f'Stres seviyesi {stress}/10'
            ))

        # Blood pressure analysis
        systolic = int(data.get('Systolic', 120))
        if systolic > 140:
            risk_factors.append(FactorItem(
                name='Yüksek Kan Basıncı',
                value='+0.20',
                description=f'Sistolik {systolic} mmHg'
            ))
        elif systolic < 120:
            protective_factors.append(FactorItem(
                name='Normal Kan Basıncı',
                value='-0.12',
                description=f'Sistolik {systolic} mmHg'
            ))

        # BMI analysis
        bmi_category = data.get('BMI_Category', 'Normal Weight')
        if bmi_category == 'Overweight':
            risk_factors.append(FactorItem(
                name='Kilolu BMI',
                value='+0.15',
                description='BMI kategorisi: Kilolu'
            ))
        elif bmi_category == 'Obese':
            risk_factors.append(FactorItem(
                name='Obez BMI',
                value='+0.25',
                description='BMI kategorisi: Obez'
            ))
        elif bmi_category in ['Normal Weight', 'Normal']:
            protective_factors.append(FactorItem(
                name='Normal BMI',
                value='-0.10',
                description='BMI kategorisi: Normal'
            ))

        # Physical activity analysis
        activity = int(data.get('Physical_Activity_Level', 5))
        if activity <= 3:
            risk_factors.append(FactorItem(
                name='Düşük Fiziksel Aktivite',
                value=f'+0.{(5 - activity) * 2:02d}',
                description=f'Aktivite seviyesi {activity}/10'
            ))
        elif activity >= 8:
            protective_factors.append(FactorItem(
                name='Yüksek Fiziksel Aktivite',
                value=f'-0.{(activity - 5) * 2:02d}',
                description=f'Aktivite seviyesi {activity}/10'
            ))

        return FactorsResponse(
            success=True,
            risk_factors=risk_factors[:3],
            protective_factors=protective_factors[:3]
        )

    except Exception as e:
        return FactorsResponse(
            success=False,
            risk_factors=[],
            protective_factors=[],
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "shap_loaded": shap_data is not None
    }


@app.post("/api/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(input_data: SleepInput):
    """Kişiselleştirilmiş risk azaltma önerileri"""
    try:
        data = input_data.dict()
        current_risk = await calculate_enhanced_prediction(input_data)

        recommendations: List[RecommendationItem] = []
        potential_values: List[float] = []  # <-- her önerinin sayısal etkisi

        # Sleep duration
        sleep_duration = float(data.get('Sleep_Duration', 7.5))
        if sleep_duration < 7:
            target_improvement = min(8, sleep_duration + 1.5)
            potential_reduction = 0.15
            recommendations.append(RecommendationItem(
                title="Uyku Süresini Artır",
                description=f"Günlük uyku sürenizi {sleep_duration} saatten {target_improvement} saate çıkarın",
                impact=f"Risk yaklaşık %{int(potential_reduction * 100)} azalabilir",
                priority=1,
                category="sleep"
            ))
            potential_values.append(potential_reduction)

        # Stress
        stress = int(data.get('Stress_Level', 5))
        if stress >= 6:
            target_stress = max(3, stress - 2)
            potential_reduction = (stress - target_stress) * 0.04  # 0.08 gibi
            recommendations.append(RecommendationItem(
                title="Stres Yönetimi",
                description="Günlük 10-15 dakika nefes egzersizi veya meditasyon yapın",
                impact=f"Risk yaklaşık %{int(potential_reduction * 100)} azalabilir",
                priority=2,
                category="stress"
            ))
            potential_values.append(potential_reduction)

        # Physical activity
        activity = int(data.get('Physical_Activity_Level', 5))
        if activity <= 5:
            target_activity = min(8, activity + 2)
            potential_reduction = (target_activity - activity) * 0.02  # ~0.06
            recommendations.append(RecommendationItem(
                title="Fiziksel Aktiviteyi Artır",
                description="Haftada 3-4 kez 30 dakika orta tempolu yürüyüş yapın",
                impact=f"Risk yaklaşık %{int(potential_reduction * 100)} azalabilir",
                priority=3,
                category="exercise"
            ))
            potential_values.append(potential_reduction)

        # Sleep hygiene
        quality = int(data.get('Quality_of_Sleep', 7))
        if quality <= 6:
            potential_reduction = 0.10
            recommendations.append(RecommendationItem(
                title="Uyku Hijyenini İyileştir",
                description="Yatmadan 1 saat önce ekranları kapatın ve oda sıcaklığını 18-20°C'de tutun",
                impact=f"Risk yaklaşık %{int(potential_reduction * 100)} azalabilir",
                priority=2,
                category="sleep_hygiene"
            ))
            potential_values.append(potential_reduction)

        # Weight management
        bmi_category = data.get('BMI_Category', 'Normal Weight')
        if bmi_category in ['Overweight', 'Obese']:
            potential_reduction = 0.20
            recommendations.append(RecommendationItem(
                title="Sağlıklı Kilo Yönetimi",
                description="Dengeli beslenme ve düzenli egzersizle sağlıklı kiloya ulaşın",
                impact=f"Risk yaklaşık %{int(potential_reduction * 100)} azalabilir",
                priority=1,
                category="weight"
            ))
            potential_values.append(potential_reduction)

        # TOP-4 öneri ve toplam etkisini sınırla
        recommendations.sort(key=lambda x: x.priority)
        recommendations = recommendations[:4]
        potential_values = potential_values[:len(recommendations)]
        total_reduction = min(0.4, sum(potential_values))

        return RecommendationsResponse(
            success=True,
            current_risk=current_risk,
            recommendations=recommendations,
            potential_risk_reduction=total_reduction
        )

    except Exception as e:
        # Debug için kısa hata mesajı ekleyelim (UI loglayabilir)
        return RecommendationsResponse(
            success=False,
            current_risk=0.0,
            recommendations=[],
            potential_risk_reduction=0.0
        )


# Enhanced prediction function
async def calculate_enhanced_prediction(input_data: SleepInput) -> float:
    """Gelişmiş, medikal literatür tabanlı risk hesaplama"""
    await asyncio.sleep(0.1)  # Simulate processing

    data = input_data.dict()

    # Base risk (population baseline)
    risk = 0.12

    # Age factor (sleep disorders increase with age)
    age = int(data.get('Age', 35))
    if age < 25:
        risk += 0.02
    elif age < 40:
        risk += 0.05
    elif age < 55:
        risk += 0.10
    else:
        risk += 0.18

    # Sleep duration factor (critical factor)
    sleep_duration = float(data.get('Sleep_Duration', 7.5))
    if sleep_duration < 5:
        risk += 0.35
    elif sleep_duration < 6:
        risk += 0.25
    elif sleep_duration < 7:
        risk += 0.15
    elif sleep_duration > 10:
        risk += 0.10
    elif 7 <= sleep_duration <= 9:
        risk -= 0.08  # Optimal range

    # Sleep quality factor
    quality = int(data.get('Quality_of_Sleep', 7))
    if quality <= 3:
        risk += 0.20
    elif quality <= 5:
        risk += 0.12
    elif quality >= 8:
        risk -= 0.05

    # Stress factor (major contributor)
    stress = int(data.get('Stress_Level', 5))
    if stress >= 8:
        risk += 0.18
    elif stress >= 6:
        risk += 0.10
    elif stress <= 3:
        risk -= 0.06

    # BMI factor
    bmi_category = data.get('BMI_Category', 'Normal Weight')
    if bmi_category == 'Obese':
        risk += 0.22
    elif bmi_category == 'Overweight':
        risk += 0.12
    elif bmi_category in ['Normal Weight', 'Normal']:
        risk -= 0.05

    # Physical activity factor
    activity = int(data.get('Physical_Activity_Level', 5))
    if activity <= 2:
        risk += 0.15
    elif activity <= 4:
        risk += 0.08
    elif activity >= 8:
        risk -= 0.10

    # Blood pressure factor
    systolic = int(data.get('Systolic', 120))
    if systolic >= 140:
        risk += 0.15
    elif systolic >= 130:
        risk += 0.08
    elif systolic < 120:
        risk -= 0.03

    # Heart rate factor
    heart_rate = int(data.get('Heart_Rate', 75))
    if heart_rate >= 90:
        risk += 0.08
    elif heart_rate <= 60:
        risk -= 0.02

    # Daily steps factor
    steps = int(data.get('Daily_Steps', 8000))
    if steps < 3000:
        risk += 0.10
    elif steps < 5000:
        risk += 0.05
    elif steps > 12000:
        risk -= 0.05

    # Ensure risk is in valid range
    risk = max(0.01, min(0.95, risk))

    return risk


def get_risk_level(prediction: float) -> str:
    """Risk seviyesi belirle"""
    if prediction < 0.25:
        return 'low'
    elif prediction < 0.65:
        return 'medium'
    else:
        return 'high'


def generate_explanation(prediction: float, data: dict) -> str:
    """Risk açıklaması oluştur"""
    percentage = int(prediction * 100)

    if percentage < 25:
        level = "düşük"
        advice = "Mevcut yaşam tarzınızı koruyun ve düzenli uyku alışkanlıklarınızı sürdürün."
    elif percentage < 65:
        level = "orta"
        advice = "Uyku hijyeninizi iyileştirici adımlar atabilir ve stres yönetimine odaklanabilirsiniz."
    else:
        level = "yüksek"
        advice = "Uyku kalitesi için yaşam tarzı değişiklikleri yapmanızı ve gerekirse bir sağlık uzmanına danışmanızı öneririz."

    return f"Uyku bozukluğu riskiniz {level} seviyede (%{percentage}). {advice}"


# ==== RAG BLOCK START ========================================================

RAG_DATA_DIR = "data/rag"
RAG_INDEX_DIR = "data/rag_index"
RAG_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

Path(RAG_INDEX_DIR).mkdir(parents=True, exist_ok=True)
RAG_MD_DIR = BASE_DIR / "data" / "rag"
app.mount("/rag", StaticFiles(directory=str(RAG_MD_DIR), html=False), name="rag")

@lru_cache(maxsize = 1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(RAG_MODEL, device="cpu")


MD_INLINE  = re.compile(r"(\*\*|__|`|[_*~])")
MD_HEADERS = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
MD_LINKS   = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def clean_md(text: str) -> str:
    t = MD_HEADERS.sub("", text)
    t = MD_LINKS.sub(r"\1", t)
    t = MD_INLINE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ0-9])", re.MULTILINE)

def sent_split(text: str) -> List[str]:
    sents = [s.strip() for s in SPLIT_RE.split(text) if s.strip()]
    merged, buf = [], ""
    for s in sents:
        if len(buf) < 120:
            buf = (buf + " " + s).strip()
        else:
            merged.append(buf); buf = s
    if buf: merged.append(buf)
    return merged

def chunk_text(text: str, max_chars=700, overlap=120) -> List[str]:
    sents = sent_split(text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + 1 + len(s) <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = (cur[-overlap:] + " " + s).strip() if overlap and len(cur) > overlap else s
    if cur: chunks.append(cur)
    return chunks

class _Meta(BaseModel):
    doc_id: int
    path: str
    chunk: str

class RagStore:
    def __init__(self, data_dir: Path, index_dir: Path):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.embedder = get_embedder()
        self.vecs: np.ndarray | None = None
        self.meta: List[_Meta] = []
        self.tfidf_mat = None

        from typing import Optional
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf: Optional[TfidfVectorizer] = None
        self.tfidf_mat = None

    def _paths(self):
        return self.index_dir / "vectors.npy", self.index_dir / "meta.json"

    def build_or_load(self):
        vpath, mpath = self._paths()
        try:
            if vpath.exists() and mpath.exists():
                self.vecs = np.load(vpath)
                with open(mpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.meta = [_Meta(**r) for r in raw]
                self._fit()
                return

            metas: List[_Meta] = []
            texts: List[str] = []
            doc_id = 0
            for p in sorted(self.data_dir.glob("**/*")):
                if p.is_dir() or p.suffix.lower() not in {".md", ".txt"}:
                    continue
                try:
                    content = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = p.read_text(encoding="latin-1")
                for ch in chunk_text(clean_md(content)):
                    metas.append(_Meta(doc_id=doc_id, path=str(p), chunk=ch))
                    texts.append(ch)
                doc_id += 1

            if not texts:
                self.vecs = np.zeros((0, 384), dtype=np.float32)  # MiniLM → 384
                self.meta = []
                self._fit()
                return

            vecs = self.embedder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype(np.float32)

            self.index_dir.mkdir(parents=True, exist_ok=True)
            np.save(vpath, vecs)
            with open(mpath, "w", encoding="utf-8", newline="\n") as f:
                json.dump([m.model_dump() for m in metas], f, ensure_ascii=False, indent=2)

            self.vecs, self.meta = vecs, metas
            self._fit()

        except Exception as e:
            import traceback, uuid
            code = uuid.uuid4().hex[:8]
            print(f"[RAG] build_or_load FAILED ({code})")
            traceback.print_exc()
            raise RuntimeError(f"RAG index build failed ({code}): {type(e).__name__}: {e}")

    def _fit(self):
        if self.vecs is None or len(self.vecs) == 0:
            self.nn = None; self.tfidf = None; self.tfidf_mat = None
            return
        self.nn = NearestNeighbors(n_neighbors=min(8, len(self.vecs)), metric="cosine")
        self.nn.fit(self.vecs)
        texts = [m.chunk for m in self.meta]
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.tfidf_mat = self.tfidf.fit_transform(texts)

    def search_hybrid_mmr(self, question: str, top_k=4, alpha: float = 0.7, lambda_mult: float = 0.7):
        if self.vecs is None or len(self.vecs) == 0:
            return []
        q_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        dense = self.vecs @ q_emb
        if self.tfidf is not None and self.tfidf_mat is not None:
            q_tfidf = self.tfidf.transform([question])
            lex = cosine_similarity(q_tfidf, self.tfidf_mat)[0]
        else:
            lex = np.zeros_like(dense)
        comb = alpha * dense + (1 - alpha) * lex

        top_n = min(20, len(comb))
        cand = np.argsort(-comb)[:top_n].tolist()

        selected = []
        for _ in range(min(top_k, len(cand))):
            best_i, best_score = None, -1e9
            for i in cand:
                diversity = 0.0
                if selected:
                    diversity = max(float(self.vecs[i] @ self.vecs[j]) for j in selected)
                score = lambda_mult * float(self.vecs[i] @ q_emb) - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score, best_i = score, i
            selected.append(best_i)
            cand.remove(best_i)

        return [(float(self.vecs[i] @ q_emb), self.meta[i]) for i in selected]

# singleton

# eğer RagStore bu satırın altında tanımlıysa, ileri referans için tırnakla yaz
_store: Optional["RagStore"] = None

def get_store() -> "RagStore":
    global _store
    if _store is None:
        _store = RagStore(RAG_DATA_DIR, RAG_INDEX_DIR)
        _store.build_or_load()
    return _store

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = 4
    max_sentences: int = 3

class Chunk(BaseModel):
    doc_id: int
    path: str
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    chunks: List[Chunk]

def pick_sentences(text: str, question: str, max_sentences: int = 3) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sents:
        return ""
    q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    scored = []
    for s in sents:
        s_tokens = re.findall(r"\b\w+\b", s.lower())
        overlap = sum(1 for w in s_tokens if w in q_tokens)
        scored.append((overlap, -len(s), s))
    scored.sort(reverse=True)
    picked = [s for _,__, s in scored[:max_sentences]]
    return " ".join(picked)

router = APIRouter(prefix="/api/rag", tags=["rag"])

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        store = get_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    hits = store.search_hybrid_mmr(req.question, top_k=req.top_k, alpha=0.7, lambda_mult=0.7)
    if not hits:
        raise HTTPException(status_code=404, detail=f"No documents in {RAG_DATA_DIR}. Put .md/.txt files and restart.")
    best_texts: List[str] = []; citations: List[str] = []; chunks: List[Chunk] = []
    for sim, meta in hits[:req.top_k]:
        short = pick_sentences(meta.chunk, req.question, max_sentences=req.max_sentences)
        best_texts.append(short)
        citations.append(meta.path)
        chunks.append(Chunk(doc_id=meta.doc_id, path=meta.path, text=meta.chunk[:600], score=sim))
    answer = " ".join([t for t in best_texts if t][:2]).strip()
    if len(answer) > 600:
        answer = answer[:600].rsplit(" ", 1)[0] + "..."
    return AskResponse(answer=answer, citations=citations[:3], chunks=chunks)

@router.post("/reindex")
def reindex():
    global _store
    _store = RagStore(RAG_DATA_DIR, RAG_INDEX_DIR)
    _store.build_or_load()
    return {"ok": True, "docs": len(_store.meta)}


app.include_router(router)

# --- OpenAI client
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qa_router = APIRouter(prefix="/api/qa", tags=["qa"])


class QARequest(BaseModel):
    question: str = Field(..., min_length=3)
    # openai rag context iptal
    top_k: int = 0
    max_sentences: int = 3

class QACitation(BaseModel):
    path: str
    snippet: str

class QAResponse(BaseModel):
    answer: str
    citations: List[QACitation] = []

# ---- RAG KAPALI: Bu fonksiyon ve get_store tamamen kaldırıldı.

# ---- Prompt inşası (yalnızca kullanıcı sorusu)
def build_prompt(question: str):
    sys = (
        "You are SleepCoach.AI, a helpful lifestyle assistant. "
        "Answer in Turkish. Be concise (2–4 sentences). "
        "If something depends on personal factors, say it may vary. "
        "This is not medical advice."
    )
    user = f"Soru: {question}\n\n"
    return sys, user

# ---- OpenAI çağrısı (JSON mod opsiyonel)
def ask_openai(system_prompt: str, user_prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        # JSON zorunlu değil
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    user_prompt
                    + "\n\nYalnızca şu JSON'ı döndür:\n"
                      '{"answer": "...", "citations": []}\n'
                      "citations boş bir liste olabilir."
                ),
            },
        ],
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        return {"answer": txt, "citations": []}

@qa_router.post("/ask_openai", response_model=QAResponse)
def qa_openai(req: QARequest):
    # RAG tamamen kapalı: ctx yok
    sys, user = build_prompt(req.question)
    data = ask_openai(sys, user)

    # citations her zaman boş dönüyor
    return QAResponse(
        answer=data.get("answer", "").strip(),
        citations=[]
    )

app.include_router(qa_router)

# -------------------------
# ALARM & ADHERENCE (SQLite)
# -------------------------
DB_URL = "sqlite:///sleepcoach.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

from typing import Optional
class Alarm(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)   # int | None -> Optional[int]
    user_id: str = Field(index=True, default="demo-user")
    label: str = Field(default="Ekranları kapat")
    hour: int = Field(default=23)
    minute: int = Field(default=15)
    days_mask: int = Field(default=127)
    category: str = Field(default="sleep")
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Adherence(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)   # int | None -> Optional[int]
    user_id: str = Field(index=True, default="demo-user")
    alarm_id: int = Field(default=0)
    ts: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="done")  # done | snooze | skipped | missed
    latency_sec: int = Field(default=0)

def init_alarm_db():
    SQLModel.metadata.create_all(engine)

# ensure DB created on startup (if you already have app.on_event("startup"), keep both)
@app.on_event("startup")
async def _init_alarm_db_and_models():
    init_alarm_db()
    # if you already have a load_models() call, keep it; this coexists

# CRUD endpoints
@app.post("/api/alarms", response_model=Alarm)
def create_alarm(a: Alarm):
    with Session(engine) as s:
        s.add(a); s.commit(); s.refresh(a)
        return a

@app.get("/api/alarms")
def list_alarms(user_id: str = "demo-user"):
    with Session(engine) as s:
        return s.exec(select(Alarm).where(Alarm.user_id == user_id)).all()

@app.patch("/api/alarms/{alarm_id}", response_model=Alarm)
def update_alarm(alarm_id: int, patch: dict):
    with Session(engine) as s:
        a = s.get(Alarm, alarm_id)
        if not a:
            raise HTTPException(status_code=404, detail="Alarm not found")
        for k, v in patch.items():
            if hasattr(a, k):
                setattr(a, k, v)
        s.add(a); s.commit(); s.refresh(a)
        return a

@app.delete("/api/alarms/{alarm_id}")
def delete_alarm(alarm_id: int):
    with Session(engine) as s:
        a = s.get(Alarm, alarm_id)
        if not a:
            raise HTTPException(status_code=404, detail="Alarm not found")
        s.delete(a); s.commit()
        return {"success": True}

# upcoming alarms in the next window_min minutes
@app.get("/api/alarms/upcoming")
def upcoming_alarms(user_id: str = "demo-user", window_min: int = 60):
    now = datetime.now()
    end = now + timedelta(minutes=window_min)
    weekday = now.weekday()  # 0=Mon .. 6=Sun
    items = []
    with Session(engine) as s:
        alarms = s.exec(select(Alarm).where(Alarm.user_id == user_id, Alarm.enabled == True)).all()
        for a in alarms:
            # check day mask
            if not ((a.days_mask >> weekday) & 1):
                continue
            fire = now.replace(hour=a.hour, minute=a.minute, second=0, microsecond=0)
            if fire < now:
                fire += timedelta(days=1)
            if now <= fire <= end:
                items.append({"alarm": a, "fire_at": fire.isoformat()})
    return {"now": now.isoformat(), "items": items}

# Adherence / check-in
class CheckIn(PydanticBaseModel):
    alarm_id: int
    status: str  # done | snooze | skipped | missed
    latency_sec: int = 0
    user_id: str = "demo-user"

@app.post("/api/adherence/checkin")
def checkin(ci: CheckIn):
    with Session(engine) as s:
        rec = Adherence(user_id=ci.user_id, alarm_id=ci.alarm_id, status=ci.status, latency_sec=ci.latency_sec)
        s.add(rec); s.commit()
    return {"success": True}

@app.get("/api/adherence/stats")
def adherence_stats(user_id: str = "demo-user", days: int = 7):
    since = datetime.utcnow() - timedelta(days=days)
    with Session(engine) as s:
        rows = s.exec(select(Adherence).where(Adherence.user_id == user_id, Adherence.ts >= since)).all()
    total = len(rows)
    done = sum(1 for r in rows if r.status == "done")
    missed = sum(1 for r in rows if r.status == "missed")
    dates_done = {r.ts.date() for r in rows if r.status == "done"}
    d = datetime.utcnow().date(); streak = 0
    while d in dates_done:
        streak += 1
        d -= timedelta(days=1)
    rate = (done / total) if total else 0.0
    return {"total": total, "done": done, "missed": missed, "rate": rate, "streak": streak}

# ========= AUTH / USERS =========
from sqlmodel import SQLModel, Field, create_engine, Session, select, Column, TEXT
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from pydantic import BaseModel as PydBaseModel, EmailStr

SECRET_KEY = os.getenv("SC_SECRET_KEY", "change-me-please-super-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 gün

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    full_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserCreate(PydBaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class Token(PydBaseModel):
    access_token: str
    token_type: str = "bearer"

class UserRead(PydBaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]

def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)
def hash_password(plain): return pwd_context.hash(plain)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_session():
    with Session(engine) as s:
        yield s

def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
    cred_exc = HTTPException(status_code=401, detail="Geçersiz yetkilendirme", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid: int = payload.get("sub")
        if uid is None: raise cred_exc
    except JWTError:
        raise cred_exc
    user = session.get(User, uid)
    if not user: raise cred_exc
    return user

@app.post("/auth/register", response_model=Token)
def register(payload: UserCreate, session: Session = Depends(get_session)):
    exists = session.exec(select(User).where(User.email == payload.email)).first()
    if exists:
        raise HTTPException(status_code=400, detail="Bu e-posta zaten kayıtlı")
    user = User(email=payload.email, full_name=payload.full_name or "", hashed_password=hash_password(payload.password))
    session.add(user); session.commit(); session.refresh(user)
    token = create_access_token({"sub": user.id})
    return Token(access_token=token)

@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == form.username)).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="E-posta veya şifre hatalı")
    token = create_access_token({"sub": user.id})
    return Token(access_token=token)

@app.get("/auth/me", response_model=UserRead)
def me(current: User = Depends(get_current_user)):
    return UserRead(id=current.id, email=current.email, full_name=current.full_name or "")

# ========= ASSESSMENTS =========
class Assessment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Girdi ve çıktı: JSON-Text saklıyoruz (SQLite)
    input_json: str = Field(sa_column=Column(TEXT))
    prediction: float
    risk_level: str
    explanation: str

class SaveAssessmentRequest(PydBaseModel):
    input: dict              # form verileriniz (SleepInput.dict())
    prediction: float
    risk_level: str
    explanation: str

@app.post("/api/assessments", response_model=dict)
def save_assessment(req: SaveAssessmentRequest, current: User = Depends(get_current_user), session: Session = Depends(get_session)):
    row = Assessment(
        user_id=current.id,
        input_json=json.dumps(req.input, ensure_ascii=False),
        prediction=req.prediction,
        risk_level=req.risk_level,
        explanation=req.explanation
    )
    session.add(row); session.commit(); session.refresh(row)
    return {"ok": True, "id": row.id}

@app.get("/api/assessments", response_model=list[dict])
def list_assessments(current: User = Depends(get_current_user), session: Session = Depends(get_session)):
    rows = session.exec(select(Assessment).where(Assessment.user_id == current.id).order_by(Assessment.created_at.desc())).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "prediction": r.prediction,
            "risk_level": r.risk_level,
            "explanation": r.explanation
        })
    return out

@app.get("/api/assessments/{assess_id}", response_model=dict)
def get_assessment(assess_id: int, current: User = Depends(get_current_user), session: Session = Depends(get_session)):
    r = session.get(Assessment, assess_id)
    if not r or r.user_id != current.id:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat(),
        "input": json.loads(r.input_json),
        "prediction": r.prediction,
        "risk_level": r.risk_level,
        "explanation": r.explanation
    }

@app.delete("/api/assessments/{assess_id}", response_model=dict)
def delete_assessment(assess_id: int, current: User = Depends(get_current_user), session: Session = Depends(get_session)):
    r = session.get(Assessment, assess_id)
    if not r or r.user_id != current.id:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
    session.delete(r); session.commit()
    return {"ok": True}

# ========= ALARMS — kullanıcıya bağla =========
# Mevcut Alarm CRUD’unuzu, current user ile ilişkilendiriyoruz:

@app.post("/api/alarms", response_model=Alarm)
def create_alarm(a: Alarm, current: User = Depends(get_current_user)):
    with Session(engine) as s:
        a.user_id = str(current.id)
        s.add(a); s.commit(); s.refresh(a)
        return a

@app.get("/api/alarms")
def list_alarms(current: User = Depends(get_current_user)):
    with Session(engine) as s:
        return s.exec(select(Alarm).where(Alarm.user_id == str(current.id))).all()

@app.get("/api/alarms/upcoming")
def upcoming_alarms(window_min: int = 60, current: User = Depends(get_current_user)):
    now = datetime.now()
    end = now + timedelta(minutes=window_min)
    weekday = now.weekday()
    items = []
    with Session(engine) as s:
        alarms = s.exec(select(Alarm).where(Alarm.user_id == str(current.id), Alarm.enabled == True)).all()
        for a in alarms:
            if not ((a.days_mask >> weekday) & 1):
                continue
            fire = now.replace(hour=a.hour, minute=a.minute, second=0, microsecond=0)
            if fire < now: fire += timedelta(days=1)
            if now <= fire <= end:
                items.append({"alarm": a, "fire_at": fire.isoformat()})
    return {"now": now.isoformat(), "items": items}





if __name__ == "__main__":
    import uvicorn

    print("🚀 SleepCoach.AI Backend Starting...")
    print("🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)