/**
 * Client-side markdown link converter for stubborn LLMs
 * Converts [text](url) markdown links to clickable HTML
 */
export function convertMarkdownLinks(text) {
  if (!text || typeof text !== 'string') return text;
  const markdownLinkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  return text.replace(markdownLinkRegex, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

export function processMessageForDisplay(message) {
  if (!message) return message;
  if (message.includes('<a href=')) return message;
  return convertMarkdownLinks(message);
}

export function setupLinkConversion() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          const messageElements = node.querySelectorAll('.message-content, [data-message], .chat-message');
          messageElements.forEach((element) => {
            const originalText = element.textContent;
            if (originalText && originalText.includes('[') && originalText.includes('](')) {
              const processedHTML = processMessageForDisplay(originalText);
              if (processedHTML !== originalText) element.innerHTML = processedHTML;
            }
          });
        }
      });
    });
  });
  observer.observe(document.body, { childList: true, subtree: true });
  return observer;
}

