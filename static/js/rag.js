(function () {
  const API_BASE  = window.API_BASE      || "http://localhost:8000";
  const DOCS_BASE = window.RAG_DOCS_BASE || "/rag/"; // .md dosyalarının servis edildiği yol

  const $ = (id) => document.getElementById(id);

  const $form    = $("ragForm");
  const $btn     = $("ragBtn");
  const $q       = $("q");
  const $status  = $("ragStatus");
  const $result  = $("ragResult");
  const $answer  = $("ragAnswer");
  const $cits    = $("ragCitations");
  const $chunks  = $("ragChunks");

  // Modal elemanları
  const $mdModal   = $("mdModal");
  const $mdTitle   = $("mdTitle");
  const $mdBody    = $("mdBody");
  const $mdClose   = $("mdCloseBtn");
  const $mdOpenNew = $("mdOpenNew");

  if ($mdClose) $mdClose.addEventListener("click", ()=> $mdModal.style.display="none");

  if (!$form || !$btn || !$q) return;

  $form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = ($q.value || "").trim();
    if (!question) {
      setStatus("Lütfen bir soru yazın.");
      return;
    }
    await askRag(question);
  });

  $q.addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const question = ($q.value || "").trim();
      if (!question) return;
      await askRag(question);
    }
  });

  async function askRag(question) {
    setLoading(true, "Aranıyor…");
    try {
      const res = await fetch(API_BASE + "/api/rag/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4, max_sentences: 2 }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(res.status + " " + res.statusText + ": " + txt);
      }
      const data = await res.json();

      if ($answer) $answer.textContent = data.answer || "—";

      // Kaynakları modal ile açan linkler olarak render et
      renderCitations(data.citations);

      if ($chunks) $chunks.textContent = JSON.stringify(data.chunks || [], null, 2);
      if ($result) $result.style.display = "block";
      setLoading(false, "");
    } catch (err) {
      console.error(err);
      setLoading(false, "Bir hata oluştu. Konsolu kontrol edin.");
      alert(String(err));
    }
  }

  function setLoading(on, msg) {
    if ($btn) {
      $btn.disabled = on;
      $btn.style.opacity = on ? "0.7" : "1";
    }
    setStatus(msg || "");
  }
  function setStatus(msg) { if ($status) $status.textContent = msg; }

  // Yardımcılar: dosya adı → etiket ve URL
  function fileBase(p) {
    const fixed = String(p).replace(/\\/g, "/");
    return fixed.split("/").pop() || fixed;
  }
  function trTitleCase(str) {
    const small = new Set(["ve","ile","veya","ya","da","de","ki","mi"]);
    return str.split(/\s+/).map((w,i)=>{
      const lw = w.toLocaleLowerCase("tr-TR");
      if (i>0 && small.has(lw)) return lw;
      return lw.charAt(0).toLocaleUpperCase("tr-TR")+lw.slice(1);
    }).join(" ");
  }
  function prettyLabelFromFile(file) {
    let name = file.replace(/\.md$/i,"").replace(/[_-]+/g," ").trim();
    name = name.replace(/\bcbt i\b/gi,"CBT-I").replace(/\bcbti\b/gi,"CBT-I");
    return trTitleCase(name);
  }

  // Kaynakları listele (modal ile aç)
  function renderCitations(list) {
    if (!$cits) return;
    $cits.innerHTML = "";

    (list || []).forEach((it)=>{
      const raw = typeof it === "string" ? it : (it && it.path) ? it.path : "";
      if (!raw) return;

      const base  = fileBase(raw);                       // "kafein_ve_uyku.md"
      const label = prettyLabelFromFile(base);
      const url   = DOCS_BASE + encodeURIComponent(base);// "/rag/kafein_ve_uyku.md"

      const li = document.createElement("li");
      const a  = document.createElement("a");
      a.href   = "#";
      a.textContent = label;

      a.addEventListener("click", async (e)=>{
        e.preventDefault();
        try {
          const res = await fetch(url);
          if (!res.ok) throw new Error(await res.text());
          const md = await res.text();

          // Markdown -> HTML (marked) + sanitize (DOMPurify)
          const rawHtml = marked.parse(md);
          const safeHtml = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;

          if ($mdTitle) $mdTitle.textContent = label;
          if ($mdBody)  $mdBody.innerHTML = safeHtml;
          if ($mdOpenNew) { $mdOpenNew.href = url; $mdOpenNew.style.display = "inline"; }

          if ($mdModal) $mdModal.style.display = "block";
        } catch (err) {
          console.error(err);
          alert("Dosya yüklenemedi: " + err);
        }
      });

      li.appendChild(a);
      $cits.appendChild(li);
    });
  }

  // Debug için global yardımcı (opsiyonel)
  window.__RAG__ = {
    ask: (q) => askRag(q),
    setApiBase: (u) => (window.API_BASE = u),
    setDocsBase: (u) => (window.RAG_DOCS_BASE = u),
  };
})();
