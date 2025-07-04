class PromptInterceptor {
  constructor(processText, onSendPrompt) {
    this.processText = processText;
    this.cleanupFn = null;
    this.onSendPrompt = onSendPrompt;
  }

  setup() {
    if (this.cleanupFn) {
      return;
    }

    const prompt = document.getElementById('prompt-textarea');

    if (prompt) { 
      const listener = async (e) => {
        if (
          e.key === 'Enter' && !e.shiftKey ||
          e.key === 'Enter' && e.ctrlKey
        ) {
          for (const child of prompt.childNodes) {
            if (child.textContent) {
              const processedText = await this.processText(child.textContent);
              child.textContent = processedText;
            }
          }
          this.onSendPrompt();
        }
      }

      prompt.addEventListener('keydown', listener, /* useCapture */ true);
      this.cleanupFn = () => {
        prompt.removeEventListener('keydown', listener, /* useCapture */ true);
      }
    }
  }

  cleanup() {
    if (this.cleanupFn) {
      this.cleanupFn();
      this.cleanupFn = null;
    }
  }
}

class ExistingMessageModifier {
  constructor(processText) {
    this.processText = processText;
    this.previousMessageId = null;
    this.seen = new Set();
  }

  setup() {
    for (const message of document.querySelectorAll('[data-message-author-role]')) {
      const messageId = message.getAttribute("data-message-id");
      if (!this.seen.has(messageId) || (this.previousMessageId && this.previousMessageId === messageId)) {
        this.recursivelyProcessMessage(message, this.processText);
        this.seen.add(messageId);
        // We assume that changes for the same messageId come together.
        // Multiple mutations to the same message ID is an indication that ChatGPT is streaming the response.
        this.previousMessageId = messageId;
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

  // handleSendPrompt() {
  //   this.hasSentOnce = true;
  //   this.seenAfterLastSend.forEach(item => {
  //     console.log("adding", item);
  //     this.seenBeforeLastSend.add(item);
  //   });
  //   this.seenAfterLastSend.clear();
  // }
}

function isMessage(node) {
  let parent = node.parentElement;

  while (parent !== null) {
    if (parent.getAttribute('data-message-author-role')) {
      return true;
    }
    parent = parent.parentElement;
  }
  return false;
}

function modifyParent(node, processText) {
  let children = [];
  node.childNodes.forEach(child => {
    const lastChild = children[children.length - 1];
    if (lastChild && lastChild.nodeType === Node.TEXT_NODE) {
      lastChild.textContent = lastChild.textContent + child.textContent;
    } else {
      children.push(child);
    }
  });
  let changed = false;
  children.forEach(child => {
    if (child.nodeType === Node.TEXT_NODE) {
      const processed = processText(child.textContent);
      if (processed !== child.textContent) {
        child.textContent = processed;
        changed = true;
      }
    }
  });
  if (changed) {
    node.replaceChildren(...children);
  }
}

function handleTextChange(node, processText) {
  console.log("handleTextChange", node);
  if (!isMessage(node)) {
    return;
  }
  modifyParent(node.parentElement, processText);
}

function setupPage(redact, restore) {
  let newMessageObserver = null;
  let doneWithInitialElements = null;

  const existingMessageModifier = new ExistingMessageModifier(restore);
  const promptInterceptor = new PromptInterceptor(redact, () => {
    if (doneWithInitialElements) {
      doneWithInitialElements();
    }
  });
  
  let elementObserver = new MutationObserver(async (mut) => {
    console.log("elementObserver", mut);
    promptInterceptor.setup();
    existingMessageModifier.setup();
  });
  elementObserver.observe(document.body, { childList: true, subtree: true });


  doneWithInitialElements = () => {
    // elementObserver.disconnect();
    // elementObserver = null;

    // Restore sensitive data in new messages as they come.
    // newMessageObserver = new MutationObserver(async (mutations) => {
    //   console.log("newMessageObserver", mutations);
      // mutations.forEach(mutation => {
      //   handleTextChange(mutation.target, restore);
      // })
    // });
    // newMessageObserver.observe(document.body, { characterData: true, subtree: true });
  }

  return () => {
    promptInterceptor.cleanup();
    if (elementObserver) {
      elementObserver.disconnect();
    }
    if (newMessageObserver) {
      newMessageObserver.disconnect();
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
