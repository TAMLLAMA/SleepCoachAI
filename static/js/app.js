// SleepCoach.AI - Main JavaScript Application
class SleepCoachApp {
    constructor() {
        this.currentPrediction = null;
        this.shapData = null;
        this.charts = {};
        this.isRealTimeMode = false;
        this.lastResult = null;
        this.init();
    }

    async init() {
        console.log('🚀 SleepCoach.AI Starting...');

        this.setupSliders();
        this.setupForm();
        this.loadShapData();

        this.bindSaveButton();

        // Auto-fill demo data after 2 seconds
        setTimeout(() => {
            this.loadDemoData();
            this.predict();
        }, 2000);
    }

    bindSaveButton() {
    const btn = document.getElementById('saveAssessBtn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
      try {
        // Sonuç yoksa otomatik tahmin çalıştır
        if (!this.lastResult) {
          await this.predict();
        }
        if (!this.lastResult) {
          alert('Tahmin oluşturulamadı. Lütfen tekrar deneyin.');
          return;
        }
        await saveAssessment(this.lastResult);
      } catch (e) {
        console.error(e);
        alert('Kaydetme sırasında bir hata oluştu.');
      }
    });
  }

    setupSliders() {
        const sliders = document.querySelectorAll('.slider');

        sliders.forEach(slider => {
            const valueElement = document.getElementById(slider.id + 'Value');

            // Initial value display
            this.updateSliderValue(slider, valueElement);

            // Real-time updates
            slider.addEventListener('input', (e) => {
                this.updateSliderValue(e.target, valueElement);

                // Auto-predict if we're in real-time mode
                if (this.isRealTimeMode) {
                    this.debouncedPredict();
                }
            });
        });
    }

    updateSliderValue(slider, valueElement) {
        let value = parseFloat(slider.value);
        let displayValue = value;

        // Format based on slider type
        switch(slider.id) {
            case 'Age':
                displayValue = Math.round(value);
                break;
            case 'Sleep_Duration':
                displayValue = value.toFixed(1);
                break;
            case 'Daily_Steps':
                displayValue = (value * 1000).toLocaleString(); // Convert to actual steps
                slider.setAttribute('data-value', value * 1000); // Store real value
                break;
            case 'Heart_Rate':
            case 'Systolic':
            case 'Diastolic':
                displayValue = Math.round(value);
                break;
            default:
                displayValue = Math.round(value);
        }

        valueElement.textContent = displayValue;
    }

    setupForm() {
        const form = document.getElementById('sleepForm');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.predict();
        });

        // Setup debounced prediction for real-time mode
        this.debouncedPredict = this.debounce(() => this.predict(), 1000);
    }

    async loadShapData() {
        try {
            const response = await fetch('/api/shap');
            if (response.ok) {
                this.shapData = await response.json();
                console.log('✅ SHAP data loaded');
            }
        } catch (error) {
            console.log('⚠️ SHAP data not available, using mock data');
            this.shapData = {
                base_value: 0.15,
                top_features: [
                    { name: "Systolic", importance: 1.9046 },
                    { name: "BMI_Category_Overweight", importance: 0.7587 },
                    { name: "Sleep_Duration", importance: 0.7570 },
                    { name: "Age", importance: 0.5180 }
                ]
            };
        }
    }

    loadDemoData() {
        // Load friendly demo data
        const demoValues = {
            'Gender': 'Male',
            'BMI_Category': 'Normal Weight',
            'Occupation': 'Software Engineer',
            'Age': '32',
            'Sleep_Duration': '7.2',
            'Quality_of_Sleep': '6',
            'Physical_Activity_Level': '4',
            'Stress_Level': '6',
            'Heart_Rate': '78',
            'Daily_Steps': '6.5', // This will show as 6500 steps
            'Systolic': '125',
            'Diastolic': '82'
        };

        Object.entries(demoValues).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                element.value = value;

                // Update slider displays
                if (element.classList.contains('slider')) {
                    const valueElement = document.getElementById(key + 'Value');
                    this.updateSliderValue(element, valueElement);
                }
            }
        });

        console.log('📝 Demo data loaded');
    }

   async predict() {
  const formData = this.getFormData();
  if (!this.validateFormData(formData)) return;

  this.showLoading();
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(formData)
    });

    const ct = res.headers.get('content-type') || '';
    if (!res.ok) {
      const text = await res.text().catch(()=>'(no body)');
      console.error('predict non-OK:', res.status, text);
      alert(`Sunucu hatası (${res.status}). Ayrıntı için konsola bakın.`);
      return;
    }
    if (!ct.includes('application/json')) {
      const text = await res.text().catch(()=>'(no body)');
      console.error('predict non-JSON:', res.status, text);
      alert('Sunucudan beklenmeyen yanıt (JSON değil).');
      return;
    }

    const result = await res.json();
    if (result.success) {
      this.lastResult = {
        input: formData,
        prediction: result.prediction,
        risk_level: result.risk_level,
        explanation: result.explanation
      };

      const saveBtn = document.getElementById('saveAssessBtn');
      if (saveBtn) saveBtn.disabled = false;

      await this.displayResults(result);
      await this.loadFactors(formData);
      await this.loadRecommendations(formData);
      this.isRealTimeMode = true;
    } else {
      console.error('predict logical error:', result);
      alert(result.error || 'Tahmin başarısız.');
    }
  } catch (err) {
    console.error('predict fetch error:', err);
    alert('Ağ/parse hatası. Konsola bakın.');
  } finally {
    this.hideLoading();
  }
}



    getFormData() {
        const formData = {};
        const form = document.getElementById('sleepForm');
        const formElements = form.elements;

        for (let element of formElements) {
            if (element.name && element.value !== '') {
                let value = element.value;

                // Convert values based on field type
                if (element.type === 'range') {
                    if (element.id === 'Daily_Steps') {
                        value = parseFloat(element.value) * 1000; // Convert to actual steps
                    } else {
                        value = parseFloat(element.value);
                    }
                }

                formData[element.name] = value;
            }
        }

        return formData;
    }

    validateFormData(data) {
        const required = ['Gender', 'BMI_Category', 'Occupation'];

        for (let field of required) {
            if (!data[field]) {
                this.showError(`Lütfen ${field} alanını doldurun`);
                return false;
            }
        }

        return true;
    }

    async displayResults(result) {
        // Hide initial message, show results
        document.getElementById('initialMessage').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'block';

        // Update risk percentage and level
        const percentage = Math.round(result.prediction * 100);
        document.getElementById('riskPercentage').textContent = `${percentage}%`;

        // Update risk level with friendly messaging
        const riskLevel = document.getElementById('riskLevel');
        const confidenceText = document.getElementById('confidenceText');

        riskLevel.className = 'risk-level';

        if (result.risk_level === 'low') {
            riskLevel.classList.add('risk-low');
            riskLevel.textContent = 'Düşük Risk 😊';
            confidenceText.textContent = 'Harika! Uyku alışkanlıklarınız oldukça iyi görünüyor.';
        } else if (result.risk_level === 'medium') {
            riskLevel.classList.add('risk-medium');
            riskLevel.textContent = 'Orta Risk 🤔';
            confidenceText.textContent = 'Birkaç küçük değişiklikle daha iyi uyuyabilirsiniz.';
        } else {
            riskLevel.classList.add('risk-high');
            riskLevel.textContent = 'Yüksek Risk 😟';
            confidenceText.textContent = 'Uyku kalitenizi iyileştirmek için adımlar atabiliriz.';
        }

        // Update explanation with friendly tone
        const explanationText = document.getElementById('explanationText');
        explanationText.textContent = this.getFriendlyExplanation(result);

        // Create risk chart
        this.createRiskChart(result.prediction);

        // Create SHAP chart if available
        if (this.shapData) {
            this.createShapChart();
        }

        this.hideLoading();
    }

    getFriendlyExplanation(result) {
        const percentage = Math.round(result.prediction * 100);

        if (result.mock) {
            return `Bu tahmin demo amaçlı hesaplandı (%${percentage} risk). Gerçek model yüklendiğinde daha doğru sonuçlar alacaksınız. ${result.explanation}`;
        }

        let explanation = result.explanation || '';

        // Make explanation more friendly
        if (result.risk_level === 'low') {
            explanation += ' Mevcut yaşam tarzınızı sürdürmeye devam edin! 🌟';
        } else if (result.risk_level === 'medium') {
            explanation += ' Endişe etmeyin - küçük değişikliklerle büyük farklar yaratabilirsiniz! 💪';
        } else {
            explanation += ' Ama panik yapmanıza gerek yok - birlikte çözümler bulabiliriz! 🤝';
        }

        return explanation;
    }

    async loadFactors(formData) {
        try {
            const response = await fetch('/api/factors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                this.displayFactors(result.risk_factors, result.protective_factors);
            }
        } catch (error) {
            console.error('Factors loading error:', error);
        }
    }

    displayFactors(riskFactors, protectiveFactors) {
        // Display risk factors
        const riskList = document.getElementById('riskFactorsList');
        if (riskFactors && riskFactors.length > 0) {
            riskList.innerHTML = riskFactors.map(factor => `
                <div class="factor-item">
                    <div class="factor-content">
                        <div class="factor-name">${factor.name}</div>
                        <div class="factor-desc">${factor.description}</div>
                    </div>
                    <div class="factor-value risk-value">${factor.value}</div>
                </div>
            `).join('');
        } else {
            riskList.innerHTML = '<div class="factor-placeholder">Harika! Önemli risk faktörü tespit edilmedi. 🎉</div>';
        }

        // Display protective factors
        const protectiveList = document.getElementById('protectiveFactorsList');
        if (protectiveFactors && protectiveFactors.length > 0) {
            protectiveList.innerHTML = protectiveFactors.map(factor => `
                <div class="factor-item">
                    <div class="factor-content">
                        <div class="factor-name">${factor.name}</div>
                        <div class="factor-desc">${factor.description}</div>
                    </div>
                    <div class="factor-value protective-value">${factor.value}</div>
                </div>
            `).join('');
        } else {
            protectiveList.innerHTML = '<div class="factor-placeholder">Koruyucu faktörleri güçlendirmek için öneriler geliştirebiliriz! 🛡️</div>';
        }
    }

    async loadRecommendations(formData) {
        try {
            const response = await fetch('/api/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                this.displayRecommendations(result);
            }
        } catch (error) {
            console.error('Recommendations loading error:', error);
        }
    }

    displayRecommendations(data) {
        const section = document.getElementById('recommendationsSection');
        const currentRiskValue = document.getElementById('currentRiskValue');
        const potentialImprovement = document.getElementById('potentialImprovement');
        const recommendationsList = document.getElementById('recommendationsList');

        // Show section
        section.style.display = 'block';

        // Update current risk
        currentRiskValue.textContent = `${Math.round(data.current_risk * 100)}%`;

        // Update potential improvement
        const improvement = Math.round(data.potential_risk_reduction * 100);
        potentialImprovement.textContent = `Potansiyel iyileştirme: %${improvement} azalma`;

        // Display recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            recommendationsList.innerHTML = data.recommendations.map(rec => `
                <div class="recommendation-item priority-${rec.priority}">
                    <div class="recommendation-header">
                        <div class="recommendation-title">${rec.title}</div>
                        <div class="recommendation-impact">${rec.impact}</div>
                    </div>
                    <div class="recommendation-description">${rec.description}</div>
                </div>
            `).join('');
        } else {
            recommendationsList.innerHTML = '<div class="recommendation-placeholder">Harika! Şu anda önerebileceğimiz iyileştirme alanı bulunmuyor.</div>';
        }
    }

    createRiskChart(prediction) {
        const canvas = document.getElementById('riskChart');
        const ctx = canvas.getContext('2d');

        // Destroy existing chart
        if (this.charts.risk) {
            this.charts.risk.destroy();
        }

        const percentage = prediction * 100;

        // Create doughnut chart
        this.charts.risk = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [
                        this.getRiskColor(prediction),
                        '#E5E7EB'
                    ],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    createShapChart() {
        const canvas = document.getElementById('shapChart');
        const ctx = canvas.getContext('2d');

        // Canvas boyutunu sabit ayarla
        canvas.height = 300;
        canvas.style.height = '300px';

        // Destroy existing chart
        if (this.charts.shap) {
            this.charts.shap.destroy();
        }

        const features = this.shapData.top_features.slice(0, 5); // Top 5

        this.charts.shap = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features.map(f => f.name.replace(/_/g, ' ')),
                datasets: [{
                    label: 'Feature Importance',
                    data: features.map(f => f.importance),
                    backgroundColor: 'rgba(74, 144, 226, 0.8)',
                    borderColor: 'rgba(74, 144, 226, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // Bu çok önemli
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: Math.max(...features.map(f => f.importance)) * 1.1, // Dinamik max değer
                        title: {
                            display: true,
                            text: 'Importance'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Features'
                        }
                    }
                }
            }
        });
    }

    getRiskColor(prediction) {
        if (prediction < 0.3) return '#50C878'; // Green
        if (prediction < 0.7) return '#FFA726'; // Orange
        return '#FF6B6B'; // Red
    }

    showLoading() {
        const btn = document.getElementById('predictBtn');
        const btnText = btn.querySelector('.btn-text');
        const btnLoading = btn.querySelector('.btn-loading');

        btn.disabled = true;
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline';
    }

    hideLoading() {
        const btn = document.getElementById('predictBtn');
        const btnText = btn.querySelector('.btn-text');
        const btnLoading = btn.querySelector('.btn-loading');

        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }

    showError(message) {
        alert(`Hata: ${message}`); // In production, use a better error display
        this.hideLoading();
    }

    // Utility function for debouncing
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Enhanced CSS for factors display
const additionalCSS = `
.factor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #F1F5F9;
}

.factor-item:last-child {
    border-bottom: none;
}

.factor-content {
    flex: 1;
}

.factor-name {
    font-weight: 600;
    color: #1F2937;
    margin-bottom: 4px;
}

.factor-desc {
    font-size: 0.85em;
    color: #6B7280;
}

.factor-value {
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.9em;
    min-width: 60px;
    text-align: center;
}

.risk-value {
    background: #FEE2E2;
    color: #DC2626;
}

.protective-value {
    background: #DCFCE7;
    color: #16A34A;
}

/* Recommendations Section Styles */
.recommendations-section {
    margin-top: 25px;
    padding: 20px;
    background: linear-gradient(135deg, #f6f9fc 0%, #eef2f7 100%);
    border-radius: 12px;
    border-left: 4px solid var(--primary-blue);
}

.recommendations-section h3 {
    margin-bottom: 15px;
    color: var(--text-dark);
}

.current-risk-info {
    display: flex;
    justify-content: space-between;
    margin-bottom: 20px;
    padding: 12px;
    background: white;
    border-radius: 8px;
    font-size: 0.9em;
}

.potential-improvement {
    color: var(--success-green);
    font-weight: 600;
}

.recommendations-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.recommendation-item {
    background: white;
    padding: 16px;
    border-radius: 10px;
    border-left: 4px solid transparent;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.recommendation-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.recommendation-item.priority-1 { border-left-color: #FF6B6B; }
.recommendation-item.priority-2 { border-left-color: #FFA726; }
.recommendation-item.priority-3 { border-left-color: #4A90E2; }

.recommendation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.recommendation-title {
    font-weight: 600;
    color: #1F2937;
    font-size: 1em;
}

.recommendation-impact {
    background: #50C878;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 500;
}

.recommendation-description {
    color: #6B7280;
    font-size: 0.9em;
    line-height: 1.4;
}

.recommendation-placeholder {
    text-align: center;
    color: #6B7280;
    padding: 20px;
    font-style: italic;
}
`;

// Add additional CSS
const style = document.createElement('style');
style.textContent = additionalCSS;
document.head.appendChild(style);

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SleepCoachApp();

    console.log('🎉 SleepCoach.AI Ready!');
    console.log('💡 Features: Real-time prediction, SHAP explainability, friendly UX, recommendations');
});

// ----- Alarm UI + Polling -----
async function requestNotifPermission() {
  if (Notification.permission !== "granted") {
    try { await Notification.requestPermission(); } catch {}
  }
}

function showAlarmNotification(a) {
  const body = `${a.label} (${String(a.hour).padStart(2,'0')}:${String(a.minute).padStart(2,'0')})`;
  if (Notification.permission === "granted") {
    new Notification("SleepCoach.AI ⏰", { body });
  } else {
    alert(`⏰ ${body}`);
  }
  // sesi başlat + overlay aç
  alarmPlayer.start({ pattern: "beep-beep", a: 920, b: 620 });
  openAlarmOverlay(body);
}

async function loadAlarms() {
  try {
    const r = await fetch('/api/alarms');
    const list = await r.json();
    const ul = document.getElementById('alarmList');
    ul.innerHTML = '';
    list.forEach(a => {
      const li = document.createElement('li');
      li.style.cssText = "display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9";
      li.innerHTML = `
        <div><b>${a.label}</b> — ${String(a.hour).padStart(2,'0')}:${String(a.minute).padStart(2,'0')}
          <span style="margin-left:8px;padding:2px 8px;border-radius:10px;background:${a.enabled?'#DCFCE7':'#FEE2E2'};font-size:.8em;">
            ${a.enabled?'Açık':'Kapalı'}
          </span>
        </div>
        <div style="display:flex;gap:6px;">
          <button data-id="${a.id}" class="toggleAlarm">${a.enabled ? 'Kapat' : 'Aç'}</button>
          <button data-id="${a.id}" class="delAlarm">Sil</button>
        </div>`;
      ul.appendChild(li);
    });

    // toggle
    ul.querySelectorAll('.toggleAlarm').forEach(b => b.onclick = async (e) => {
      const id = e.target.getAttribute('data-id');
      const aResp = await fetch(`/api/alarms`);
      const all = await aResp.json();
      const ai = all.find(x => String(x.id) === String(id));
      if (!ai) return;
      await fetch(`/api/alarms/${id}`, {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({enabled: !ai.enabled})
      });
      await loadAlarms();
    });

    // delete
    ul.querySelectorAll('.delAlarm').forEach(b => b.onclick = async (e) => {
      const id = e.target.getAttribute('data-id');
      await fetch(`/api/alarms/${id}`, {method:'DELETE'});
      await loadAlarms();
    });

    // adherence stats
    const st = await (await fetch('/api/adherence/stats?days=7')).json();
    document.getElementById('adherenceStats').textContent =
      `Son 7 gün: ${Math.round(st.rate*100)}% uyum, streak: ${st.streak}`;
  } catch (err) {
    console.error('loadAlarms err', err);
  }
}

async function createAlarm() {
  const a = {
    label: document.getElementById('alarmLabel').value || 'Hatırlatma',
    hour: Number(document.getElementById('alarmHour').value),
    minute: Number(document.getElementById('alarmMinute').value),
    days_mask: Number(document.getElementById('alarmDays').value),
    enabled: true,
    category: "sleep",
    user_id: "demo-user"
  };
  await fetch('/api/alarms', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(a)
  });
  await loadAlarms();
}

let pollTimer = null;
const scheduled = new Set(); // aynı alarmı iki kez planlamayı engelle

async function pollUpcoming() {
  try {
    const r = await fetch('/api/alarms/upcoming');
    const { items } = await r.json(); // [{ alarm:{...}, fire_at: ISO8601 }]
    const now = Date.now();

    items.forEach(it => {
      const fireAtMs = new Date(it.fire_at).getTime();
      const delta = fireAtMs - now;

      // Önümüzdeki 90 sn içinde çalacaksa tek seferlik timeout kur
      if (delta > 0 && delta < 90_000) {
        const key = `${it.alarm.id}-${fireAtMs}`;
        if (scheduled.has(key)) return;
        scheduled.add(key);

        setTimeout(async () => {
          showAlarmNotification(it.alarm);
          // (opsiyonel) otomatik checkin
          try {
            await fetch('/api/adherence/checkin', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify({alarm_id: it.alarm.id, status: "done", latency_sec: 0})
            });
          } catch {}
          try { await loadAlarms(); } catch {}
        }, delta);
      }
    });
  } catch (err) {
    console.debug('pollUpcoming error', err);
  } finally {
    // 30 sn'de bir kontrol
    pollTimer = setTimeout(pollUpcoming, 30_000);
  }
}

// wire UI buttons after DOM ready
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('createAlarmBtn').addEventListener('click', createAlarm);

  // Test: sadece izin istesin, alarm çaldırmasın
  document.getElementById('testAlarmBtn').addEventListener('click', async () => {
    await requestNotifPermission();
    alert('Bildirim izni verildi ✅ Kurduğun alarmlar saatinde çalacak.');
  });

  requestNotifPermission();
  loadAlarms();
  pollUpcoming();
});

// (İsteğe bağlı: kullanmıyorsan kaldırabilirsin)
function playAlarmTone(seconds=5, freq=880) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  osc.connect(gain);
  gain.connect(ctx.destination);
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.05);
  osc.start();
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + seconds);
  osc.stop(ctx.currentTime + seconds + 0.05);
}

// Tekil AudioContext ile sürekli "beep-beep" çalan oynatıcı
class AlarmPlayer {
  constructor(){
    this.ctx = null; this.gain = null;
    this.loopId = null; this.playing = false;
  }
  _ensureCtx(){
    if(!this.ctx){
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.gain = this.ctx.createGain();
      this.gain.connect(this.ctx.destination);
      this.gain.gain.value = 0.15; // genel ses seviyesi
    }
  }
  _beep(freq=880, ms=300){
    if(!this.playing) return;
    const o = this.ctx.createOscillator();
    o.type = "sine";
    o.frequency.value = freq;
    o.connect(this.gain);
    o.start();
    setTimeout(()=>{ try{ o.stop(); o.disconnect(); }catch{} }, ms);
  }
  start({pattern="beep-beep", a=880, b=660}={}){
    this._ensureCtx();
    this.playing = true;
    const tick = () => {
      if(!this.playing) return;
      if(pattern === "beep-beep"){
        this._beep(a, 300);
        setTimeout(()=> this._beep(b, 300), 400);
      } else {
        this._beep(a, 600);
      }
    };
    tick();
    this.loopId = setInterval(tick, 1200);
  }
  stop(){
    this.playing = false;
    if(this.loopId) clearInterval(this.loopId);
    if(this.gain){
      try{
        this.gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime+0.1);
      }catch{}
    }
  }
}
const alarmPlayer = new AlarmPlayer();

// === Overlay kontrolü ===
function openAlarmOverlay(text){
  const ov = document.getElementById('alarmOverlay');
  const txt = document.getElementById('alarmOverlayText');
  const btn = document.getElementById('alarmStopBtn');

  if (txt) txt.textContent = text || '';
  if (ov) ov.style.display = 'flex';

  if (btn) {
    btn.disabled = false;
    btn.textContent = 'Alarmı Kapat';
  }
}

function closeAlarmOverlay(){
  const ov = document.getElementById('alarmOverlay');
  if (ov) ov.style.display = 'none';
}

// === Overlay butonları DOM yüklendikten sonra bağlanmalı ===
document.addEventListener('DOMContentLoaded', () => {
  const stopBtn   = document.getElementById('alarmStopBtn');
  const snoozeBtn = document.getElementById('alarmSnoozeBtn');
  const muteBtn   = document.getElementById('alarmMuteBtn'); // varsa

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      try { alarmPlayer.stop(); } catch {}
      closeAlarmOverlay();
    });
  }

  if (snoozeBtn) {
    snoozeBtn.addEventListener('click', async () => {
      try { alarmPlayer.stop(); } catch {}
      closeAlarmOverlay();
      // 5 dk sonraya tek seferlik alarm
      try{
        const now = new Date();
        now.setMinutes(now.getMinutes() + 5);
        await fetch('/api/alarms', {
          method : 'POST',
          headers: { 'Content-Type':'application/json' },
          body   : JSON.stringify({
            label    : 'Erteleme',
            hour     : now.getHours(),
            minute   : now.getMinutes(),
            days_mask: 127,
            enabled  : true,
            category : 'sleep',
            user_id  : 'demo-user'
          })
        });
      } catch {}
    });
  }

  if (muteBtn) {
    muteBtn.addEventListener('click', () => {
      try {
        if (alarmPlayer && alarmPlayer.gain && alarmPlayer.ctx) {
         // iOS/Safari vs. için emin olmak adına:
          if (alarmPlayer.ctx.state === 'suspended') { alarmPlayer.ctx.resume().catch(()=>{}); }
          const ctx = alarmPlayer.ctx;
          alarmPlayer.gain.gain.setValueAtTime(0.0001, ctx.currentTime);
          setTimeout(() => {

          try { alarmPlayer.gain.gain.setValueAtTime(0.15, ctx.currentTime); } catch {}
          }, 60_000);
        }
      } catch {}
    });
  }
});


const API = location.origin;
function authHeader(){
  const t = localStorage.getItem('sc_token');
  return t ? { Authorization: 'Bearer ' + t } : {};
}

async function saveAssessment(latest){
  // Token kontrolü
  const token = localStorage.getItem('sc_token');
  if(!token){
    // giriş yoksa login’e yönlendir
    location.href = '/login?redirect=/assessment';
    return;
  }

  const r = await fetch(API + '/api/assessments', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + token
    },
    body: JSON.stringify(latest)
  });

  if(r.ok){
    alert('✅ Değerlendirme kaydedildi. Panelden görebilirsiniz.');
    // isterseniz: location.href = '/dashboard';
  }else{
    const t = await r.text().catch(()=> '');
    console.error('save fail', t);
    alert('Kaydedilemedi. Lütfen tekrar deneyin.');
  }
}




