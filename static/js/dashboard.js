(function(){
  const fmtDT = (d) => new Intl.DateTimeFormat('tr-TR', { dateStyle:'medium', timeStyle:'short' }).format(new Date(d));
  const byId = (id) => document.getElementById(id);

  async function fetchJSON(url, token){
    const r = await fetch(url, { headers: token ? { Authorization: 'Bearer ' + token } : {} });
    if (!r.ok) throw new Error(r.status + ':' + r.statusText);
    return r.json();
  }

  function initials(name){
    const s = (name || '').trim().split(/\s+/).slice(0,2).map(x=>x[0]?.toUpperCase()||'').join('');
    return s || '🙂';
  }

  function renderProfile(me){
    // avatar
    const av = document.createElement('div');
    av.style.width = '48px';
    av.style.height = '48px';
    av.style.borderRadius = '999px';
    av.style.display = 'grid';
    av.style.placeItems = 'center';
    av.style.fontWeight = '800';
    av.style.color = '#fff';
    av.style.background = 'linear-gradient(135deg,#6c63ff,#a78bfa)';
    av.textContent = initials(me.full_name || me.email);
    const holder = byId('profAvatar');
    holder.replaceWith(av);

    // name + email
    const nameEl = document.createElement('div');
    nameEl.textContent = me.full_name || (me.email || '').split('@')[0] || 'Kullanıcı';
    nameEl.style.fontWeight = '800';
    nameEl.style.fontSize = '18px';

    const mailEl = document.createElement('div');
    mailEl.textContent = me.email || '—';
    mailEl.style.color = '#6b7280';
    mailEl.style.fontSize = '13px';
    const nHolder = byId('profName'); const eHolder = byId('profEmail');
    nHolder.replaceWith(nameEl); eHolder.replaceWith(mailEl);

    // üyelik tarihi (istatistik)
    if(me.created_at){
      byId('stMember').textContent = new Intl.DateTimeFormat('tr-TR', { dateStyle:'medium' }).format(new Date(me.created_at));
    } else {
      byId('stMember').textContent = '—';
    }

    // başlığı selamla
    byId('hello').textContent = `Merhaba, ${nameEl.textContent}`;
    byId('helloSub').textContent = 'Hesap özeti ve son aktiviteleriniz.';
  }

  function renderAssess(list){
    const wrap = byId('assessList');
    wrap.innerHTML = '';
    if(!list || list.length === 0){
      wrap.innerHTML = '<div class="muted">Henüz değerlendirme bulunamadı.</div>';
      byId('stAssess').textContent = '0';
      return;
    }
    byId('stAssess').textContent = String(list.total ?? list.length);
    list.slice(0,5).forEach(a=>{
      const row = document.createElement('div'); row.className = 'row';
      const left = document.createElement('div');
      const title = document.createElement('div'); title.className = 'title';
      title.textContent = a.title || `Değerlendirme #${a.id ?? ''}`;
      const meta = document.createElement('div'); meta.className = 'meta';
      const score = (a.score != null) ? ` · Puan: ${a.score}` : '';
      meta.textContent = `${fmtDT(a.created_at || a.date || Date.now())}${score}`;
      left.append(title, meta);
      const go = document.createElement('a'); go.className='btn'; go.textContent='Aç';
      go.href = `/assessment/${a.id ?? ''}`;
      row.append(left, go);
      wrap.append(row);
    });
  }

  function renderAlarms(list){
    const wrap = byId('alarmList');
    wrap.innerHTML = '';
    if(!list || list.length === 0){
      wrap.innerHTML = '<div class="muted">Aktif alarm yok.</div>';
      byId('stAlarms').textContent = '0';
      return;
    }
    byId('stAlarms').textContent = String(list.total ?? list.length);
    list.slice(0,5).forEach(a=>{
      const row = document.createElement('div'); row.className='row';
      const left = document.createElement('div');
      const title = document.createElement('div'); title.className='title';
      title.textContent = a.label || 'Alarm';
      const meta = document.createElement('div'); meta.className='meta';
      const hh = String(a.hour ?? 0).padStart(2,'0');
      const mm = String(a.minute ?? 0).padStart(2,'0');
      meta.textContent = `${hh}:${mm} · ${a.days?.join(', ') || 'Tek sefer'}`;
      left.append(title, meta);
      const go = document.createElement('a'); go.className='btn'; go.textContent='Düzenle';
      go.href = '/dashboard#alarms';
      row.append(left, go);
      wrap.append(row);
    });
  }

  async function main(){
    const token = localStorage.getItem('sc_token');
    if(!token){
      location.replace('/login?redirect=/dashboard');
      return;
    }

    let me = null;
    try {
      me = await fetchJSON('/auth/me', token);
    } catch (e) {
      // token bozuk → çıkış
      localStorage.removeItem('sc_token');
      location.replace('/login?redirect=/dashboard');
      return;
    }

    renderProfile(me || {});

    // Veri kaynaklarını paralel çek
    // Not: API uçların yoksa sorun değil—catch ile boş gösteririz.
    const reqAssess = fetchJSON('/api/assessments?limit=5', token).catch(()=>[]);
    const reqAlarms = fetchJSON('/api/alarms?limit=5', token).catch(()=>[]);

    const [assess, alarms] = await Promise.all([reqAssess, reqAlarms]);

    try { renderAssess(Array.isArray(assess) ? assess : (assess.items ?? assess)); } catch { renderAssess([]); }
    try { renderAlarms(Array.isArray(alarms) ? alarms : (alarms.items ?? alarms)); } catch { renderAlarms([]); }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main);
  else main();
})();
