class PromptInterceptor {
  constructor(processText) {
    this.processText = processText;
    this.cleanupEnterListener = null;
    this.cleanupButtonListener = null;
    this.cache = {};
  }

  setup() {
    const handleSendPrompt = () => {
      for (const child of prompt.childNodes) {
        if (child.textContent) {
          const processedText = this.processText(child.textContent);
          child.textContent = processedText;
        }
      }
    }

    const enterListener = (e) => {
      if (
        e.key === 'Enter' && !e.shiftKey ||
        e.key === 'Enter' && e.ctrlKey
      ) {
        handleSendPrompt();
      }
    }
    
    const prompt = document.getElementById('prompt-textarea');
    if (prompt) {
      // addEventListener is idempotent, so we don't need to check if the listener is already added.
      prompt.addEventListener('keydown', enterListener, /* useCapture */ true);
      this.cleanupEnterListener = () => {
        prompt.removeEventListener('keydown', enterListener, /* useCapture */ true);
      }
    }

    const button = document.getElementById('composer-submit-button');
    if (button) {
      // addEventListener is idempotent, so we don't need to check if the listener is already added.
      button.addEventListener('click', handleSendPrompt, /* useCapture */ true);
      this.cleanupButtonListener = () => {
        button.removeEventListener('click', handleSendPrompt, /* useCapture */ true);
      }
    }
  }

  cleanup() {
    if (this.cleanupEnterListener) {
      this.cleanupEnterListener();
      this.cleanupEnterListener = null;
    }
    if (this.cleanupButtonListener) {
      this.cleanupButtonListener();
      this.cleanupButtonListener = null;
    }
  }
}

class MessageModifier {
  constructor(processText) {
    this.processText = processText;
    this.seen = {};
    this.latestTimestamp = {};
    this.cache = {};
  }

  handleMutations(mutations, timestamp) {
    let affectedMessageIds = new Set();
    mutations.forEach(mutation => {
      let affectedAncestorMessageId = this.affectedMessageAncestor(mutation);
      if (affectedAncestorMessageId) {
        affectedMessageIds.add(affectedAncestorMessageId);
      }
      let affectedDescendantMessageIds = this.affectedMessageDescendants(mutation);
      affectedDescendantMessageIds.forEach(messageId => affectedMessageIds.add(messageId));
    })
    affectedMessageIds = Array.from(affectedMessageIds);
    affectedMessageIds = affectedMessageIds.filter(id => !id.includes('-modified'));
    for (const messageId of affectedMessageIds) {
      this.latestTimestamp[messageId] = this.latestTimestamp[messageId] ? Math.max(this.latestTimestamp[messageId], timestamp) : timestamp;
      let message = document.querySelector(`[data-message-id="${messageId}"]`);
      this.modifyMessage(message, timestamp, true);
    }
  }

  async modifyMessage(messageNode, timestamp, delay = false) {
    // Add a small delay to allow for any pending DOM updates
    if (delay) {
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    let messageId = messageNode.getAttribute('data-message-id');
    if (timestamp < this.latestTimestamp[messageId]) {
      console.log("Skipping due to updates")
      return;
    }
    
    let messageChanged = !this.seen[messageId] || this.seen[messageId] !== messageNode.innerText.length;
    let messageMayHaveRedactions = messageNode.innerText.match(/\[.*?\]/g) !== null;
    if (!messageChanged || !messageMayHaveRedactions) {
      console.log("Skipping due to no changes")
      return;
    }
    
    this.seen[messageId] = messageNode.innerText.length;
    
    let modifiedId = messageId + '-modified';
    let previous = document.querySelector(`[data-message-id="${modifiedId}"]`);
    let modified = messageNode.cloneNode(true);
    modified.style.display = previous ? previous.style.display : messageNode.style.display;
    modified.setAttribute('data-message-id', modifiedId);
    modified.setAttribute('data-timestamp', timestamp);
    await this.recursivelyProcessMessage(modified, timestamp, messageId);
  
    // Query again because asynchronous processing may have modified the DOM.
    let allPrevious = document.querySelectorAll(`[data-message-id="${modifiedId}"]`);
    let existsNewer = false;
    for (const node of allPrevious) {
      if (node.getAttribute('data-timestamp') < timestamp) {
        node.remove();
      } else {
        existsNewer = true;
      }
    }
    if (existsNewer) {
      return;
    }
    
    messageNode.parentElement.prepend(modified);
    
    // So we don't unnecessarily trigger a mutation event.
    if (messageNode.style.display !== 'none') {
      messageNode.style.display = 'none';
    }
  }

  async recursivelyProcessMessage(node, timestamp, messageId) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (!this.cache[messageId]) {
        this.cache[messageId] = {};
      }
      if (!this.cache[messageId][node.textContent]) {
        this.cache[messageId][node.textContent] = await this.processText(node.textContent, timestamp);
      }
      node.textContent = this.cache[messageId][node.textContent];
    }
    for (const child of node.childNodes) {
      await this.recursivelyProcessMessage(child, timestamp, messageId);
    }
  }

  affectedMessageAncestor(mutation) {
    let target = mutation.target;
    while (target && !target.hasAttribute('data-message-id')) {
      if (target == document.body) {
        return null;
      }
      target = target.parentElement;
    }
    return target.getAttribute('data-message-id');
  }

  affectedMessageDescendants(mutation) {
    let addedNodes = Array.from(mutation.addedNodes).filter(node => node.nodeType === Node.ELEMENT_NODE);
    let removedNodes = Array.from(mutation.removedNodes).filter(node => node.nodeType === Node.ELEMENT_NODE);
    let descendants = [...addedNodes, ...removedNodes].map(node => {
      return Array.from(node.querySelectorAll('[data-message-id]'))
    }).flat();
    return descendants.map(node => node.getAttribute('data-message-id'));
  }
}

async function setupPage(redact, restore) {
  const promptInterceptor = new PromptInterceptor(redact);
  const messageModifier = new MessageModifier(restore);

  promptInterceptor.setup();
  for (const message of document.querySelectorAll('[data-message-id]')) {
    if (!message.getAttribute('data-message-id').includes('-modified')) {
      await messageModifier.modifyMessage(message, Date.now());
    }
  }
  
  let elementObserver = new MutationObserver((mutations) => {
    if (mutations.reduce((acc, mutation) => {
      const hasButton = mutation.target.id === 'composer-submit-button' || !!mutation.target.querySelector('#composer-submit-button');
      const hasPrompt = mutation.target.id === 'prompt-textarea';
      return acc || hasPrompt || hasButton
    }, false)) {
      promptInterceptor.setup();
    }
    messageModifier.handleMutations(mutations, Date.now());
  });
  elementObserver.observe(document.body, { childList: true, subtree: true });

  return () => {
    promptInterceptor.cleanup();
    if (elementObserver) {
      elementObserver.disconnect();
    }
  }
}

class LocationTracker {
  constructor() {
    this.previousLocation = document.location.href;
    this.handleLocationChange = [];

    this.observer = new MutationObserver(async () => {
      const newLocation = document.location.href;
      if (newLocation !== this.previousLocation) {
        this.handleLocationChange.forEach(callback => callback(this.previousLocation, newLocation));
        this.previousLocation = newLocation;
      }
    });
    this.observer.observe(document.body, { childList: true, subtree: true });
  }

  onLocationChange(callback) {
    this.handleLocationChange.push(callback);
  }
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function getSessionId(url) {
  const match = url.match(/\/c\/([a-f0-9-]+)/);
  return match ? match[1] : null;
}


async function initialize() {
  const locationTracker = new LocationTracker();
  let wasmRedactor = await newWasmRedactor('wasm/build/');
  const makeRedact = (sessionId) => (text) => wasmRedactor.redact(text, sessionId);
  // TODO: Can we reduce initial loading time by running restore asynchronously?
  const makeRestore = (sessionId) => async (text) => {
    const response = await fetch(`http://localhost:16549/api/v1/chat/sessions/${sessionId}/restore`, {
      method: 'POST',
      body: JSON.stringify({ Message: text }),
    }).then(response => response.json()).then(data => data.Message);
    return response;
  }

  let sessionId = getSessionId(document.location.href) || generateUUID();
  let cleanupPage = await setupPage(makeRedact(sessionId), makeRestore(sessionId));

  locationTracker.onLocationChange(async (oldLocation, newLocation) => {
    cleanupPage();
    const oldSessionId = getSessionId(oldLocation);
    const newSessionId = getSessionId(newLocation);
    // In this event, it is likely that we just sent the first message in a new chat.
    if (!oldSessionId && newSessionId) {
      // sessionId is the placeholder sessionId, we need to update it to the actual sessionId.
      wasmRedactor.updateExtensionId(sessionId, newSessionId);
    }
    sessionId = newSessionId || generateUUID();
    cleanupPage = await setupPage(makeRedact(sessionId), makeRestore(sessionId));
  });
}

const INJECTION_MARKER_ID = "__my_extension_root__";

if (!document.getElementById(INJECTION_MARKER_ID)) {
  const marker = document.createElement("div");
  marker.id = INJECTION_MARKER_ID;
  marker.style.display = "none";
  document.documentElement.appendChild(marker);

  console.log("Injected content script");

  initialize();
} else {
  console.log("Script already injected, skipping.");
}