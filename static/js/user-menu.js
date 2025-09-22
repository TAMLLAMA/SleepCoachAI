(function () {
  const init = async () => {
    const slot = document.getElementById('user-slot');
    if (!slot) return;

    const token = localStorage.getItem('sc_token');
    if (!token) return;

    let me;
    try {
      const r = await fetch('/auth/me', { headers: { Authorization: 'Bearer ' + token } });
      if (!r.ok) throw new Error('unauth');
      me = await r.json();
    } catch {
      return; // geçersiz token -> fallback kalsın
    }

    const display = (me.full_name && me.full_name.trim()) || (me.email || '').split('@')[0] || 'Kullanıcı';
    const initials = display.trim().split(/\s+/).slice(0,2).map(s => s[0]?.toUpperCase() || '').join('') || '🙂';

    // --- Güvenli DOM kur ---
    slot.innerHTML = '';               // temizle
    slot.classList.add('user-slot');   // pozisyonlama için (CSS'te vardır)

    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'user-chip';
    chip.id = 'userChip';
    chip.setAttribute('aria-haspopup', 'menu');
    chip.setAttribute('aria-expanded', 'false');

    const av = document.createElement('span'); av.className = 'avatar'; av.textContent = initials;
    const nm = document.createElement('span'); nm.className = 'name'; nm.textContent = display;
    const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.setAttribute('width','16'); svg.setAttribute('height','16'); svg.setAttribute('viewBox','0 0 24 24');
    svg.innerHTML = '<path d="M7 10l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/>';
    chip.append(av,nm,svg);

    const menu = document.createElement('div');
    menu.className = 'user-menu';
    menu.id = 'userMenu';
    menu.role = 'menu';
    menu.innerHTML = `
      <a href="/dashboard" role="menuitem" tabindex="-1"><span class="ico">🏠</span>Panel</a>
      <a href="/dashboard#assessments" role="menuitem" tabindex="-1"><span class="ico">📄</span>Değerlendirmelerim</a>
      <a href="/dashboard#alarms" role="menuitem" tabindex="-1"><span class="ico">⏰</span>Alarmlarım</a>
      <a href="/dashboard#profile" role="menuitem" tabindex="-1"><span class="ico">👤</span>Profilim</a>
      <a href="/dashboard#settings" role="menuitem" tabindex="-1"><span class="ico">⚙️</span>Ayarlar</a>
      <hr>
      <button type="button" id="logoutBtn" class="danger" role="menuitem" tabindex="-1">
        <span class="ico">🚪</span>Çıkış Yap
      </button>
    `;

    slot.append(chip, menu);

    const items = Array.from(menu.querySelectorAll('[role="menuitem"]'));

    const openMenu = () => {
      slot.classList.add('open');            // CSS: .user-slot.open .user-menu{display:block}
      chip.setAttribute('aria-expanded','true');
      // ilk iteme odak
      const first = items[0];
      first && first.focus();
      // ok animasyonu için istersen chip’e expanded’a göre class verilebilir
    };

    const closeMenu = () => {
      slot.classList.remove('open');
      chip.setAttribute('aria-expanded','false');
      chip.focus();
    };

    const toggleMenu = () => (slot.classList.contains('open') ? closeMenu() : openMenu());

    // --- Etkileşimler ---
    chip.addEventListener('click', e => { e.stopPropagation(); toggleMenu(); });
    chip.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault(); openMenu();
      }
    });

    // Menü içi klavye
    menu.addEventListener('keydown', e => {
      const i = items.indexOf(document.activeElement);
      if (e.key === 'Escape') { e.preventDefault(); closeMenu(); }
      else if (e.key === 'ArrowDown') {
        e.preventDefault(); (items[i+1] || items[0]).focus();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault(); (items[i-1] || items[items.length-1]).focus();
      } else if (e.key === 'Tab') {
        // menü dışına tab’lenirse kapat
        requestAnimationFrame(() => {
          if (!menu.contains(document.activeElement)) closeMenu();
        });
      }
    });

    // Dışarı tıklayınca kapat
    document.addEventListener('click', e => { if (!slot.contains(e.target)) closeMenu(); });

    // Çıkış
    menu.querySelector('#logoutBtn')?.addEventListener('click', () => {
      localStorage.removeItem('sc_token');
      location.reload();
    });
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
