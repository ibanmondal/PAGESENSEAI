/**
 * Background service worker (Manifest V3) — the bridge between the popup/UI
 * and the Python backend. All agent/retrieval work happens server-side; this
 * worker only orchestrates extraction + HTTP. State is ephemeral (MV3 workers
 * are destroyed when idle), so we persist session id in chrome.storage.
 *
 * Backend base URL is configurable; defaults to local FastAPI dev server.
 */
const DEFAULT_BACKEND = "http://localhost:8000";

async function getBackend() {
  const { pagesense_backend } = await chrome.storage.local.get("pagesense_backend");
  return pagesense_backend || DEFAULT_BACKEND;
}

async function getSession() {
  const { pagesense_session } = await chrome.storage.local.get("pagesense_session");
  if (pagesense_session) return pagesense_session;
  const sid = "sess_" + Math.random().toString(36).slice(2, 10);
  await chrome.storage.local.set({ pagesense_session: sid });
  return sid;
}

async function extractFromTab(tab) {
  // Inject extractor if needed, then ask it to extract.
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["src/content/extractor.js"],
    });
  } catch (e) {
    // chrome:// pages and similar can't be scripted; skip gracefully.
    return null;
  }
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_PAGE" }, (resp) => {
      if (chrome.runtime.lastError || !resp?.ok) {
        resolve(null);
        return;
      }
      resolve(resp.tab);
    });
  });
}

async function ingestTabs(tabIds) {
  const backend = await getBackend();
  const sessionId = await getSession();
  const tabs = [];
  for (const id of tabIds) {
    const tab = await chrome.tabs.get(id);
    if (!tab.url || /^chrome:|edge:|about:/.test(tab.url)) continue;
    const extracted = await extractFromTab(tab);
    if (extracted && extracted.text && extracted.text.length > 50) {
      tabs.push({
        tab_id: hashUrl(extracted.url),
        url: extracted.url,
        title: extracted.title || tab.title,
        text: extracted.text,
        source_type: extracted.source_type || "web",
      });
    }
  }
  if (tabs.length === 0) {
    return { ok: false, error: "No extractable tabs." };
  }
  const res = await fetch(`${backend}/api/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, tabs }),
  });
  if (!res.ok) {
    return { ok: false, error: `Backend ${res.status}` };
  }
  const data = await res.json();
  return { ok: true, sessionId, ...data };
}

async function ask(question, mode = "ask") {
  const backend = await getBackend();
  const sessionId = await getSession();
  const res = await fetch(`${backend}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, mode }),
  });
  if (!res.ok) {
    return { ok: false, error: `Backend ${res.status}` };
  }
  return { ok: true, ...(await res.json()) };
}

// Simple string hash for stable tab ids.
function hashUrl(url) {
  let h = 0;
  for (let i = 0; i < url.length; i++) {
    h = (h << 5) - h + url.charCodeAt(i);
    h |= 0;
  }
  return "tab_" + Math.abs(h).toString(36);
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "INGEST_ALL") {
    chrome.tabs.query({}, (tabs) => {
      ingestTabs(tabs.map((t) => t.id)).then(sendResponse);
    });
    return true;
  }
  if (msg?.type === "INGEST_ACTIVE") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      ingestTabs(tabs.map((t) => t.id)).then(sendResponse);
    });
    return true;
  }
  if (msg?.type === "ASK") {
    ask(msg.question, msg.mode).then(sendResponse);
    return true;
  }
  if (msg?.type === "GET_SESSION") {
    getSession().then(sendResponse);
    return true;
  }
});

chrome.runtime.onInstalled.addListener(() => {
  console.log("PageSense AI installed.");
});
