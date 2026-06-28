/**
 * PDF Content Extractor
 * Relies on pdf.min.js being injected into the page before this script runs.
 */

// Tell pdf.js to use the local worker we bundled.
pdfjsLib.GlobalWorkerOptions.workerSrc = chrome.runtime.getURL('src/lib/pdf.worker.min.js');

async function extractPDFText() {
  try {
    // Fetch the PDF binary from the current URL
    const response = await fetch(location.href);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const arrayBuffer = await response.arrayBuffer();

    // Parse the PDF
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    let fullText = "";
    const numPages = pdf.numPages;

    // Extract text from each page
    for (let i = 1; i <= numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(" ");
      fullText += pageText + "\n\n";
    }

    const cleanedText = fullText.replace(/\s+/g, " ").trim();
    return cleanedText;
  } catch (error) {
    console.error("PageSense AI: Failed to parse PDF:", error);
    return null;
  }
}

// Respond to extraction requests from the background/popup.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "EXTRACT_PAGE") {
    extractPDFText().then(text => {
      sendResponse({
        ok: !!text,
        tab: text ? {
          url: location.href,
          // document.title is sometimes just the filename in the PDF viewer
          title: document.title || location.pathname.split('/').pop(),
          text: text,
          selection: "", // Selection is extremely difficult to capture accurately from the native PDF viewer
          source_type: "pdf",
        } : null
      });
    });
    return true; // keep the message channel open for async sendResponse
  }
});
