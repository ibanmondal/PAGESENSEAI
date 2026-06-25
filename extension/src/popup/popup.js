// Popup controller. Talks to the background service worker via chrome.runtime.
// Kept framework-free so the extension loads with zero build step.

const $ = (id) => document.getElementById(id);
const send = (msg) => new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));

function setStatus(state, text) {
  const el = $("status");
  el.className = "status " + state;
  el.textContent = text;
}

function showAnswer(resp) {
  $("error").classList.add("hidden");
  const box = $("answer");
  box.classList.remove("hidden");
  $("model").textContent = "model: " + resp.model_used;
  $("conf").textContent = "confidence: " + (resp.confidence ?? 0).toFixed(2);
  $("answer-text").textContent = resp.answer;

  const cit = $("citations");
  cit.innerHTML = "<h4>Sources</h4>";
  (resp.citations || []).forEach((c, i) => {
    const a = document.createElement("a");
    a.className = "cite";
    a.href = c.url;
    a.target = "_blank";
    a.innerHTML = `<b>[${i + 1}] ${(c.score ?? 0).toFixed(2)}</b> ${c.title}<br/>${c.snippet}`;
    cit.appendChild(a);
  });
}

function showError(msg) {
  $("answer").classList.add("hidden");
  const e = $("error");
  e.classList.remove("hidden");
  e.textContent = msg;
}

async function ingest(kind) {
  setStatus("busy", "indexing…");
  const type = kind === "all" ? "INGEST_ALL" : "INGEST_ACTIVE";
  const resp = await send({ type });
  if (resp?.ok) {
    setStatus("ok", "indexed");
    $("indexed-count").textContent = `${resp.ingested_chunks} chunks · ${resp.indexed_tabs} tabs`;
  } else {
    setStatus("err", "error");
    showError(resp?.error || "Ingest failed.");
  }
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) return;
  setStatus("busy", "thinking…");
  $("ask-btn").disabled = true;
  const mode = $("mode").value;
  const resp = await send({ type: "ASK", question, mode });
  $("ask-btn").disabled = false;
  if (resp?.ok) {
    setStatus("ok", "answered");
    showAnswer(resp);
  } else {
    setStatus("err", "error");
    showError(resp?.error || "Query failed. Did you index a tab first?");
  }
}

$("ingest-active").addEventListener("click", () => ingest("active"));
$("ingest-all").addEventListener("click", () => ingest("all"));
$("ask-btn").addEventListener("click", ask);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});

// settings: just configure backend URL via a prompt for now (roadmap: real UI)
$("settings").addEventListener("click", async () => {
  const cur = (await chrome.storage.local.get("pagesense_backend")).pagesense_backend || "http://localhost:8000";
  const next = prompt("Backend URL:", cur);
  if (next) await chrome.storage.local.set({ pagesense_backend: next.trim() });
});
