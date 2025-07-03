class PromptInterceptor {
  registered = false;

  constructor() {}

  // processText MUST be synchronous.
  register(processText, force) {
    if (force) {
      this.registered = false;
    }
    if (this.registered) {
      return;
    }
    const prompt = document.getElementById('prompt-textarea');
    if (prompt) {
      console.log("Prompt found");
      console.log(prompt)
      prompt.addEventListener('keydown', async (e) => {
        console.log("Prompt keydown", e.key);
        if (
          e.key === 'Enter' && !e.shiftKey ||
          e.key === 'Enter' && e.ctrlKey
        ) {
          for (const child of prompt.childNodes) {
            if (child.textContent) {
              const processedText = await processText(child.textContent);
              child.textContent = processedText;
            }
          }
        }
      }, /* useCapture */ true); // Use capture has to be true to catch the event before the built in handlers
      console.log("Prompt registered");
      this.registered = true;
    }
  }
}

function getNodeTypeName(nodeType) {
  const types = {
    [Node.ELEMENT_NODE]: 'ELEMENT_NODE',                  // 1
    [Node.ATTRIBUTE_NODE]: 'ATTRIBUTE_NODE',              // 2 (deprecated)
    [Node.TEXT_NODE]: 'TEXT_NODE',                        // 3
    [Node.CDATA_SECTION_NODE]: 'CDATA_SECTION_NODE',      // 4
    [Node.ENTITY_REFERENCE_NODE]: 'ENTITY_REFERENCE_NODE', // 5 (deprecated)
    [Node.ENTITY_NODE]: 'ENTITY_NODE',                    // 6 (deprecated)
    [Node.PROCESSING_INSTRUCTION_NODE]: 'PROCESSING_INSTRUCTION_NODE', // 7
    [Node.COMMENT_NODE]: 'COMMENT_NODE',                  // 8
    [Node.DOCUMENT_NODE]: 'DOCUMENT_NODE',                // 9
    [Node.DOCUMENT_TYPE_NODE]: 'DOCUMENT_TYPE_NODE',      // 10
    [Node.DOCUMENT_FRAGMENT_NODE]: 'DOCUMENT_FRAGMENT_NODE', // 11
    [Node.NOTATION_NODE]: 'NOTATION_NODE'                 // 12 (deprecated)
  };
  
  return types[nodeType] || 'UNKNOWN_NODE';
}

class MessageModifier {
  registeredMessages = {};

  constructor() {}

  async register(selector, processText) {
    const messages = document.querySelectorAll(selector);
    for (const message of messages) {
      await this.recursivelyProcessText(message, processText, message.getAttribute('data-message-id'));
    }
  }

  async recursivelyProcessText(node, processText, id) {
    if (node.nodeType === Node.TEXT_NODE && !this.registeredMessages[id]) {
      const after = await processText(node.textContent);
      node.textContent = after;
      this.registeredMessages[id] = true;
    }
    let i = 0;
    for (const child of node.childNodes) {
      await this.recursivelyProcessText(child, processText, id + `-${i++}`);
    }
  }
}

const promptInterceptor = new PromptInterceptor();
const messageModifier = new MessageModifier();

// Import WASM redactor
let wasmRedactor = null;

// Initialize WASM redactor
async function initializeRedactor() {
  if (wasmRedactor) {
    return wasmRedactor;
  }

  try {
    // Try WASM approach
    wasmRedactor = new WasmRedactor();
    const wasmSuccess = await wasmRedactor.initialize(chrome.runtime.getURL('wasm/build/'));
    
    if (wasmSuccess) {
      console.log('Using WASM redactor');
      return wasmRedactor;
    } else {
      throw new Error('WASM initialization failed');
    }
  } catch (error) {
    console.warn('WASM redactor failed, using JavaScript fallback:', error);
    
    // Fallback to simple JavaScript implementation
    wasmRedactor = {
      redact: (text) => text.replace(/Benito/g, "[REDACTED]"),
      restore: (text) => text.replace(/\[REDACTED\]/g, "Benito"),
      clearMappings: () => {},
      getStats: () => ({})
    };
    console.log('Using JavaScript fallback redactor');
    return wasmRedactor;
}
}

var placeholderSessionId = null;

// Custom UUID generator to avoid bundling dependencies. This is going to be ephemeral anyway;
// it will be discarded when the page redirects to chatgpt/c/new-session-id.
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Extract session ID from ChatGPT URL
function getSessionId(url) {
  const match = url.match(/\/c\/([a-f0-9-]+)/);
  return match ? match[1] : null;
}

// Wrapper functions that use the WASM redactor
const redact = (text) => {
  console.log("Redacting", text);
  var sessionId = getSessionId(document.location.href);
  if (!sessionId) {
    sessionId = generateUUID();
    placeholderSessionId = sessionId;
  }
  const result = wasmRedactor.redact(text, sessionId);
  return result;
}

const restore = (text) => {
  const sessionId = getSessionId(document.location.href);
  const result = wasmRedactor.restore(text, sessionId);
  return result;
}

var oldHref = document.location.href;

const observer = new MutationObserver(async (mutations) => {
  const locationChanged = document.location.href !== oldHref;
  if (locationChanged) {
    const oldHrefSessionId = getSessionId(oldHref);
    const newHrefSessionId = getSessionId(document.location.href);
    if (!oldHrefSessionId && !!placeholderSessionId && !!newHrefSessionId) {
      wasmRedactor.updateExtensionId(placeholderSessionId, newHrefSessionId);
      placeholderSessionId = null;
    }
    oldHref = document.location.href;
  }

  // Initialize redactor if not already done
  if (!wasmRedactor) {
    await initializeRedactor();
  }

  promptInterceptor.register((text) => {
    return redact(text);
  }, /* force */ locationChanged);
  
  messageModifier.register('[data-message-author-role="assistant"]', (text) => {
    return restore(text);
  });
  messageModifier.register('[data-message-author-role="user"]', (text) => {
    return restore(text);
  });
});

// Initialize redactor when script loads
initializeRedactor().then(() => {
  console.log('Redactor initialized on page load');
}).catch(error => {
  console.error('Failed to initialize redactor on page load:', error);
});

observer.observe(document.body, { childList: true, subtree: true });