(() => {
  const fileInput = document.getElementById("file");
  const domainSelect = document.getElementById("domain");
  const runBtn = document.getElementById("run");
  const statusEl = document.getElementById("status");
  const resultCard = document.getElementById("result-card");
  const summaryEl = document.getElementById("summary");
  const traceEl = document.getElementById("trace");

  let chosenBlob = null;
  let chosenName = null;

  document.querySelectorAll(".example").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const src = btn.dataset.src;
      const domain = btn.dataset.domain;
      domainSelect.value = domain;
      try {
        const r = await fetch(src);
        chosenBlob = await r.blob();
        chosenName = src.split("/").pop();
        setStatus(`Loaded example: ${chosenName} (domain=${domain}).`);
        document.querySelectorAll(".example").forEach((b) => b.classList.remove("selected"));
        btn.classList.add("selected");
      } catch (e) {
        setStatus(`Could not load example: ${e}`, true);
      }
    });
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (!f) return;
    chosenBlob = f;
    chosenName = f.name;
    document.querySelectorAll(".example").forEach((b) => b.classList.remove("selected"));
    setStatus(`Selected upload: ${f.name} (${(f.size / 1024).toFixed(0)} KB).`);
  });

  runBtn.addEventListener("click", async () => {
    if (!chosenBlob) {
      setStatus("Pick an example or upload an image first.", true);
      return;
    }
    runBtn.disabled = true;
    setStatus("Running cascade… (cold-start may take 30–60 s)");
    resultCard.hidden = true;

    const fd = new FormData();
    fd.append("file", chosenBlob, chosenName || "image");
    fd.append("domain", domainSelect.value);

    const t0 = performance.now();
    try {
      const r = await fetch("/predict", { method: "POST", body: fd });
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { raw: text }; }
      if (!r.ok) {
        setStatus(`HTTP ${r.status}: ${data.detail || text}`, true);
        return;
      }
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      renderResult(data, elapsed);
      setStatus(`Done in ${elapsed} s (server-reported ${data.elapsed_ms} ms).`);
    } catch (e) {
      setStatus(`Request failed: ${e}`, true);
    } finally {
      runBtn.disabled = false;
    }
  });

  function setStatus(msg, isErr = false) {
    statusEl.textContent = msg;
    statusEl.className = isErr ? "err" : "ok";
  }

  function renderResult(data, elapsed) {
    const decision = data.decision || "unknown";
    const stopped = data.stopped_at_layer ?? "?";
    const cls = data.class ? ` (${data.class})` : "";
    summaryEl.innerHTML = `
      <p><strong>Decision:</strong> <span class="decision decision-${decision}">${decision}${cls}</span></p>
      <p><strong>Stopped at layer:</strong> ${stopped} of 3</p>
      <p><strong>Wall time:</strong> ${elapsed} s</p>
    `;
    traceEl.textContent = JSON.stringify(data.trace || data, null, 2);
    resultCard.hidden = false;
  }
})();
