// Popup controller. Talks to the background service worker via chrome.runtime.
// Kept framework-free so the extension loads with zero build step.

const $ = (id) => document.getElementById(id);
const send = (msg) => new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));

let sessionId = null;
let chatHistory = [];

async function init() {
  sessionId = await send({ type: "GET_SESSION" });
  if (sessionId) {
    const data = await chrome.storage.local.get(`chat_${sessionId}`);
    if (data[`chat_${sessionId}`]) {
      chatHistory = data[`chat_${sessionId}`];
      chatHistory.forEach(msg => renderMessage(msg));
    }
  }
}

function setStatus(state, text) {
  const el = $("status");
  el.className = "status " + state;
  const textEl = $("status-text");
  if (textEl) textEl.textContent = text;
}

function scrollToBottom() {
  const container = $("chat-container");
  container.scrollTop = container.scrollHeight;
}

function renderMessage(msg) {
  const container = $("chat-container");
  const div = document.createElement("div");
  
  if (msg.role === "error") {
    div.className = "message error-bubble";
    div.textContent = msg.content;
  } else if (msg.role === "user") {
    div.className = "message user";
    div.textContent = msg.content;
  } else if (msg.role === "assistant") {
    div.className = "message assistant";
    
    // Meta (model & conf)
    if (msg.model_used || msg.confidence !== undefined) {
      const meta = document.createElement("div");
      meta.className = "meta";
      if (msg.model_used) {
        const m = document.createElement("span");
        m.className = "chip";
        m.textContent = "model: " + msg.model_used;
        meta.appendChild(m);
      }
      if (msg.confidence !== undefined && msg.confidence !== null) {
        const c = document.createElement("span");
        c.className = "chip";
        c.textContent = "confidence: " + msg.confidence.toFixed(2);
        meta.appendChild(c);
      }
      div.appendChild(meta);
    }
    
    // Text
    const text = document.createElement("div");
    text.id = "answer-text";
    text.textContent = msg.content;
    div.appendChild(text);
    
    // Citations
    if (msg.citations && msg.citations.length > 0) {
      const cit = document.createElement("div");
      cit.className = "citations";
      cit.innerHTML = "<h4>Sources</h4>";
      msg.citations.forEach((c, i) => {
        const a = document.createElement("a");
        a.className = "cite";
        a.href = c.url;
        a.target = "_blank";
        a.innerHTML = `<b>[${i + 1}] ${(c.score ?? 0).toFixed(2)}</b> ${c.title}<br/>${c.snippet}`;
        cit.appendChild(a);
      });
      div.appendChild(cit);
    }
  }
  
  container.appendChild(div);
  scrollToBottom();
}

async function appendAndSave(msg) {
  chatHistory.push(msg);
  renderMessage(msg);
  if (sessionId) {
    await chrome.storage.local.set({ [`chat_${sessionId}`]: chatHistory });
  }
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
    appendAndSave({ role: "error", content: resp?.error || "Ingest failed." });
  }
}

async function ask() {
  const questionInput = $("question");
  const question = questionInput.value.trim();
  if (!question) return;
  
  questionInput.value = "";
  await appendAndSave({ role: "user", content: question });
  
  setStatus("busy", "thinking…");
  $("ask-btn").disabled = true;
  
  const mode = $("mode").value;
  const resp = await send({ type: "ASK", question, mode });
  
  $("ask-btn").disabled = false;
  if (resp?.ok) {
    setStatus("ok", "answered");
    await appendAndSave({
      role: "assistant",
      content: resp.answer,
      model_used: resp.model_used,
      confidence: resp.confidence,
      citations: resp.citations
    });
  } else {
    setStatus("err", "error");
    await appendAndSave({ role: "error", content: resp?.error || "Query failed. Did you index a tab first?" });
  }
}

$("ingest-active").addEventListener("click", () => ingest("active"));
$("ingest-all").addEventListener("click", () => ingest("all"));
$("ask-btn").addEventListener("click", ask);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { 
    e.preventDefault();
    ask();
  }
});

// settings: just configure backend URL via a prompt for now (roadmap: real UI)
$("settings").addEventListener("click", async () => {
  const cur = (await chrome.storage.local.get("pagesense_backend")).pagesense_backend || "http://localhost:8000";
  const next = prompt("Backend URL:", cur);
  if (next) await chrome.storage.local.set({ pagesense_backend: next.trim() });
});

// Initialize on load
init();
