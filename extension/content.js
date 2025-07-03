class PromptInterceptor {
  registered = false;

  constructor() {}

  // processText MUST be synchronous.
  register(processText) {
    if (this.registered) {
      return;
    }
    const prompt = document.getElementById('prompt-textarea');
    if (prompt) {
      console.log("Prompt found");
      prompt.addEventListener('keydown', async (e) => {
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
    // console.log("Recursively processing text", node.textContent, id, getNodeTypeName(node.nodeType), this.registeredMessages[id], node, node.childNodes);
    if (node.nodeType === Node.TEXT_NODE && !this.registeredMessages[id]) {
      // console.log("Registering message modifier for", id);
      // console.log("Before", node.textContent);
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

// Extract session ID from ChatGPT URL
function getSessionId() {
  const url = window.location.href;
  const match = url.match(/\/c\/([a-f0-9-]+)/);
  return match ? match[1] : 'default-session';
}

// Wrapper functions that use the WASM redactor
const redact = (text) => {
  const sessionId = getSessionId();
  console.log("🔍 About to redact:", text, "Session:", sessionId);
  const result = wasmRedactor.redact(text, sessionId);
  console.log("✅ Redaction result:", result);
  return result;
}

const restore = (text) => {
  const sessionId = getSessionId();
  console.log("🔍 About to restore:", text, "Session:", sessionId);
  const result = wasmRedactor.restore(text, sessionId);
  console.log("✅ Restoration result:", result);
  return result;
}

const observer = new MutationObserver(async (mutations) => {
  // console.log("Mutation observer triggered", mutations);
  
  // Initialize redactor if not already done
  if (!wasmRedactor) {
    await initializeRedactor();
  }
  
  promptInterceptor.register((text) => {
    return redact(text);
  });
  
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