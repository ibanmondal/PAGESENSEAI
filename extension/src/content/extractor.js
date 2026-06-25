/**
 * Content extractor — runs in the page, reads main content, removes boilerplate.
 *
 * Strategy: prefer <article>/<main> elements; fall back to the <body> but strip
 * nav/footer/script/style/aside/ad. This is the "good enough" heuristic; a
 * learned boilerplate-detection model is a documented ROADMAP milestone.
 * The result is a clean text string shipped to the backend for chunking.
 */

function cleanNode(node) {
  // Clone so we don't mutate the live DOM.
  const clone = node.cloneNode(true);
  const drop = clone.querySelectorAll(
    "script, style, noscript, nav, footer, header, aside, form, " +
    "iframe, svg, [role='navigation'], [role='banner'], [aria-hidden='true'], " +
    "[class*='ad'], [class*='advert'], [class*='sidebar'], [class*='cookie']"
  );
  drop.forEach((el) => el.remove());
  return clone;
}

function extractMain() {
  // Preferred containers in priority order.
  const candidates = [
    document.querySelector("article"),
    document.querySelector("main"),
    document.querySelector("[role='main']"),
    document.querySelector("#content, .content, .post, .article, .entry-content"),
  ].filter(Boolean);

  const root = candidates[0] || document.body;
  const cleaned = cleanNode(root);

  // Preserve some structure: paragraphs/headers/list items as line breaks.
  const parts = [];
  cleaned.querySelectorAll("h1,h2,h3,h4,p,li,pre,blockquote,td").forEach((el) => {
    const text = el.textContent.replace(/\s+/g, " ").trim();
    if (text.length > 0) parts.push(text);
  });

  let text = parts.join("\n\n");
  if (text.length < 120) {
    // Fallback: whole cleaned body if structured extraction was too sparse.
    text = (cleaned.textContent || "").replace(/\s+/g, " ").trim();
  }
  return text;
}

// Respond to extraction requests from the background/popup.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "EXTRACT_PAGE") {
    const selection = window.getSelection()?.toString().trim() || "";
    const text = extractMain();
    sendResponse({
      ok: true,
      tab: {
        url: location.href,
        title: document.title,
        text,
        selection,
        source_type: detectSourceType(location),
      },
    });
  }
  return true; // keep the message channel open for async sendResponse
});

function detectSourceType(loc) {
  const host = loc.hostname;
  if (host.includes("github.com")) return "github";
  if (host.includes("youtube.com") || host.includes("youtu.be")) return "youtube";
  if (loc.pathname.toLowerCase().endsWith(".pdf")) return "pdf";
  return "web";
}
