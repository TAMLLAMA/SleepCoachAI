(function () {
  function initCarousel(carousel) {
    if (!carousel) return;

    // 1) Track oluştur ve kartları içine taşı
    let track = carousel.querySelector(".carousel__track");
    if (!track) {
      track = document.createElement("div");
      track.className = "carousel__track";
      const cards = Array.from(carousel.children).filter(el => el.classList.contains("card"));
      cards.forEach(el => track.appendChild(el)); // orijinal set
      carousel.appendChild(track);
    }

    // 2) Kesintisiz döngü için bir set daha klonla
    const originals = Array.from(track.children);
    originals.forEach(node => {
      const clone = node.cloneNode(true);
      clone.setAttribute("aria-hidden", "true");
      track.appendChild(clone);
    });

    // 3) Hıza göre süre (px/sn). data-speed ile özelleştir.
    const speed = Number(carousel.dataset.speed || 60); // px/s
    function recompute() {
      const total = Array.from(track.children)
        .reduce((sum, el) => sum + el.getBoundingClientRect().width, 0);
      const duration = total / speed;
      track.style.setProperty("--marquee-duration", `${duration}s`);
    }
    recompute();
    let rid = null;
    window.addEventListener("resize", () => {
      if (rid) cancelAnimationFrame(rid);
      rid = requestAnimationFrame(recompute);
    });
  }

  function boot() {
    document.querySelectorAll(".carousel").forEach(initCarousel);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();