class PromptInterceptor {
  registered = false;

  constructor() {}

  register(processText) {
    if (this.registered) {
      return;
    }
    const prompt = document.getElementById('prompt-textarea');
    if (prompt) {
      console.log("Prompt found");
      prompt.addEventListener('keydown', (e) => {
        if (
          e.key === 'Enter' && !e.shiftKey ||
          e.key === 'Enter' && e.ctrlKey
        ) {
          prompt.childNodes.forEach(child => {
            child.textContent = processText(child.textContent);
          });
        }
      }, true);
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

  register(selector, processText) {
    const messages = document.querySelectorAll(selector);
    messages.forEach(message => {
      this.recursivelyProcessText(message, processText, message.getAttribute('data-message-id'));
    });
  }

  recursivelyProcessText(node, processText, id) {
    // console.log("Recursively processing text", node.textContent, id, getNodeTypeName(node.nodeType), this.registeredMessages[id], node, node.childNodes);
    if (node.nodeType === Node.TEXT_NODE && !this.registeredMessages[id]) {
      // console.log("Registering message modifier for", id);
      // console.log("Before", node.textContent);
      const after = processText(node.textContent);
      node.textContent = after;
      this.registeredMessages[id] = true;
    }
    let i = 0;
    node.childNodes.forEach(child => {
      this.recursivelyProcessText(child, processText, id + `-${i++}`);
    });
  }
}

const promptInterceptor = new PromptInterceptor();
const messageModifier = new MessageModifier();

const redact = (text) => {
  return text.replace("Benito", "[REDACTED]");
}

const unredact = (text) => {
  return text.replace("[REDACTED]", "Benito");
}

const observer = new MutationObserver((mutations) => {
  console.log("Mutation observer triggered", mutations);
  promptInterceptor.register(redact);
  messageModifier.register('[data-message-author-role="assistant"]', unredact);
});

observer.observe(document.body, { childList: true, subtree: true });