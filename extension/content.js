class PromptInterceptor {
  constructor(processText) {
    this.processText = processText;
    this.cleanupEnterListener = null;
    this.cleanupButtonListener = null;
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
    this.previousMessageId = null;
    this.seen = {}
  }

  setup() {
    for (const message of document.querySelectorAll('[data-message-author-role]')) {
      const messageId = message.getAttribute("data-message-id");
      const messageText = message.textContent;
      if (!this.seen[messageId] || this.seen[messageId] != messageText) {
        this.recursivelyProcessMessage(message, this.processText);
        this.seen[messageId] = messageText;
      }
    }
  }

  recursivelyProcessMessage(node, processText) {
    if (node.nodeType === Node.TEXT_NODE) {
      const after = processText(node.textContent);
      node.textContent = after;
    }
    let i = 0;
    for (const child of node.childNodes) {
      this.recursivelyProcessMessage(child, processText);
    }
  }

}

function setupPage(redact, restore) {
  const promptInterceptor = new PromptInterceptor(redact);
  const existingMessageModifier = new MessageModifier(restore);

  promptInterceptor.setup();
  existingMessageModifier.setup();
  
  let elementObserver = new MutationObserver(async (mutations) => {
    if (mutations.reduce((acc, mutation) => {
      const hasButton = mutation.target.id === 'composer-submit-button' || !!mutation.target.querySelector('#composer-submit-button');
      const hasPrompt = !!mutation.target.getAttribute("data-message-author-role");
      return acc || hasPrompt || hasButton
    }, false)) {
      promptInterceptor.setup();
    }
    existingMessageModifier.setup();
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
  const makeRestore = (sessionId) => (text) => wasmRedactor.restore(text, sessionId);

  let sessionId = getSessionId(document.location.href) || generateUUID();
  let cleanupPage = setupPage(makeRedact(sessionId), makeRestore(sessionId));

  locationTracker.onLocationChange((oldLocation, newLocation) => {
    cleanupPage();
    const oldSessionId = getSessionId(oldLocation);
    const newSessionId = getSessionId(newLocation);
    // In this event, it is likely that we just sent the first message in a new chat.
    if (!oldSessionId && newSessionId) {
      // sessionId is the placeholder sessionId, we need to update it to the actual sessionId.
      wasmRedactor.updateExtensionId(sessionId, newSessionId);
    }
    sessionId = newSessionId || generateUUID();
    cleanupPage = setupPage(makeRedact(sessionId), makeRestore(sessionId));
  });
}

if (!window.__MY_EXTENSION_ALREADY_INJECTED__) {
  window.__MY_EXTENSION_ALREADY_INJECTED__ = true;

  console.log("Injected content script");
  
  initialize();
} else {
  console.log("Script already injected, skipping.");
}
