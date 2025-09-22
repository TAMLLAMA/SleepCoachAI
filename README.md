# SleepCoach.AI

Kişisel verilerinizi tarayıcıdan çıkarmadan (edge ML) uyku riski ve kalite tahmini yapan, açıklanabilirlik (Why?) ve RAG tabanlı kısa yanıtlar veren mini web uygulaması.

## ✨ Özellikler
- **Risk Modeli (Sınıflandırma)**: XGBoost + ONNX; AUROC, PR-AUC, F1 ile doğrulandı.
- **Kalite / Süre Regresyonu**: XGBoostRegressor + ONNX; MAE ve MAPE raporlanır.
- **SHAP Açıklanabilirlik**: Global/yerel önemler; UI’da “En etkili 3 faktör” ve bar grafik.
  - Global: Dataset genelinde özellik katkıları (özet bar/importance).
  - Lokal: Tek bir tahmin için “↑/↓ faktörler” ve doğal dil açıklama.
- **Edge Inference**: ONNX Runtime Web ile tarayıcıda tahmin (gizlilik + düşük gecikme).
- **RAG mini-asistan (TR)**: Markdown bilgi tabanından hibrit+MMR arama ve kısa cevap.
- **Plan & Alarm**: 7 günlük mikro-adım planı, bildirimli hatırlatıcı/uyum takibi.
- **Kullanıcı Hesabı**: Değerlendirme kaydetme ve panelde listeleme.
- **Mini Chatbot**: OpenAI destekli yardımcı; uyku hijyeni ve yaşam tarzı sorularına yanıt.

### SHAP nasıl çalışır?
Model her tahminde, temel (base) skordan sapmayı açıklayan SHAP değerleri üretir. Pozitif SHAP → riski artıran, negatif SHAP → riski azaltan katkı. Arayüzde:
- **Top-3 etken** listesi (etiketli, okunur Türkçe açıklamalar),
- **Bar grafikte önem sırası** (SHAP |value|),
- **Metinsel özet** (ör. “Stres ↑, Adım ↓, BMI ↑ risk puanını yükseltti.”)
gösterilir.

> Modeller, Kaggle üzerinden tarafımızca eğitildi.

---

## 1) Hızlı Kurulum

> Python 3.9+ önerilir.

```bash
# 1) Kaynak kodu alın
git clone <repo-url>
cd SleepCoach

# 2) Sanal ortam
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3) Bağımlılıklar
pip install -U pip
pip install -r requirements.txt
```
## 2) Ortam Değişkenleri (.env)
Proje kök dizininde bir .env dosyası oluşturun:
```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o-mini    # ör: gpt-4o-mini, gpt-4.1, o3-mini vb.
```
## 3) Çalıştırma
```bash
# FastAPI dev sunucusu
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```