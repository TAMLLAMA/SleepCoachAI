(function(){
  const API = window.API_BASE || "http://localhost:8000"; // ör: "http://localhost:8000"
  const root = document.getElementById("sc-chat-root");
  if (!root) return;

  // UI
  root.insertAdjacentHTML("beforeend", `
    <button id="sc-launch" aria-label="Sohbeti aç">
      <span class="sc-emoji">🤖</span>
    </button>
    <div id="sc-panel" role="dialog" aria-label="SleepCoach Sohbet">
      <div class="sc-head">
        <h4>SleepCoach Asistan</h4>
        <button class="sc-close" title="Kapat" aria-label="Kapat">×</button>
      </div>
      <div class="sc-body" id="sc-body"></div>
      <div class="sc-input">
        <input id="sc-input" type="text" placeholder="Sorunuzu yazın… (Enter)">
        <button id="sc-send" class="sc-send">Gönder</button>
      </div>
    </div>
  `);

  const launch = document.getElementById("sc-launch");
  const panel  = document.getElementById("sc-panel");
  const body   = document.getElementById("sc-body");
  const input  = document.getElementById("sc-input");
  const send   = document.getElementById("sc-send");
  const close  = panel.querySelector(".sc-close");

  function toggle(open){
    if (open === undefined) open = !panel.classList.contains("sc-open");
    panel.classList.toggle("sc-open", open);
    if (open) setTimeout(()=> input.focus(), 120);
  }
  launch.addEventListener("click", ()=> toggle(true));
  close .addEventListener("click", ()=> toggle(false));

  // helpers
  function addMsg(text, who){
    const el = document.createElement("div");
    el.className = "sc-msg " + (who === "user" ? "user" : "bot");
    el.textContent = text;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
    return el;
  }
  function addTyping(){
    const el = document.createElement("div");
    el.className = "sc-msg bot";
    el.innerHTML = `<div class="sc-typing"><span></span><span></span><span></span></div>`;
    body.appendChild(el);
    body.scrollTop = body.scrollHeight;
    return el;
  }

  // ilk selamlama
  addMsg("Merhaba! Uyku, stres, kafein, mavi ışık gibi konularda kısa ve kaynaklı yanıtlar verebilirim. Nasıl yardımcı olabilirim?", "bot");

  async function ask(q){
    if (!q.trim()) return;
    addMsg(q, "user");
    input.value = "";
    send.disabled = true;

    const typing = addTyping();
    try{
      const res = await fetch(API + "/api/qa/ask_openai", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ question:q, top_k:3, max_sentences:2 })
      });
      if(!res.ok) throw new Error(await res.text());
      const data = await res.json();

      typing.remove();
      addMsg(data.answer || "—", "bot");

      if (Array.isArray(data.citations) && data.citations.length){
        const refs = data.citations.map(c => c.path || c).slice(0,3).join(" · ");
        addMsg("Kaynaklar: " + refs, "bot");
      }
    }catch(err){
      typing.remove();
      console.error(err);
      addMsg("Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.", "bot");
    }finally{
      send.disabled = false;
      body.scrollTop = body.scrollHeight;
    }
  }

  send.addEventListener("click", ()=> ask(input.value));
  input.addEventListener("keydown", (e)=>{
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
  });
})();
