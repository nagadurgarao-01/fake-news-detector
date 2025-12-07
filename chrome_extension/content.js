
function isHeadline(node) {
  // Prevent re-processing
  if (node.dataset.processed === "true") return false;

  const text = node.innerText.trim();

  // Adjusted thresholds to catch shorter headlines
  return (
    text &&
    text.length > 25 &&
    text.split(/\s+/).length >= 4
  );
}

async function checkHeadline(text) {
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
      }),
    });

    if (!response.ok) {
      console.error("API error:", response.statusText);
      return null;
    }

    return await response.json();
  } catch (err) {
    console.error("Failed to contact detection server:", err);
    return null;
  }
}

async function scanNode(node) {
  if (node.dataset.processed === "true") return;
  node.dataset.processed = "true"; // Mark as processed immediately

  const result = await checkHeadline(node.innerText.trim());
  if (result) {
    const isFake = result.prediction === "Fake";
    // Apply visual style
    node.style.border = `2px solid ${isFake ? "red" : "green"}`;
    node.style.backgroundColor = isFake ? "rgba(255, 0, 0, 0.1)" : "rgba(0, 255, 0, 0.1)"; // Transparent background
    node.style.borderRadius = "4px";
    node.style.cursor = "help";
    node.title = `🧠 AI Analysis: ${result.prediction} (${(result.confidence_score * 100).toFixed(1)}%)`;
  }
}

function scanHeadlines() {
  // Added 'li', 'b', 'strong', 'figcaption' to catch lists, bold text, and captions
  const nodes = document.querySelectorAll("h1, h2, h3, h4, h5, h6, p, span, a, div, li, b, strong, figcaption");
  nodes.forEach((node) => {
    // Only process if it's a headline and hasn't been processed
    if (isHeadline(node)) scanNode(node);
  });
}

function observeDynamicContent() {
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === 1) { // Element node
          // Check the node itself
          if (node.matches && node.matches("h1, h2, h3, h4, h5, h6, p, span, a, div, li, b, strong, figcaption")) {
            if (isHeadline(node)) scanNode(node);
          }
          // Check children
          const elements = node.querySelectorAll("h1, h2, h3, h4, h5, h6, p, span, a, div, li, b, strong, figcaption");
          elements.forEach((el) => {
            if (isHeadline(el)) scanNode(el);
          });
        }
      }
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

scanHeadlines();
observeDynamicContent();
