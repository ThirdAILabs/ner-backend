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

function affectedMessageAncestor(mutation) {
  let target = mutation.target;
  while (target && !target.hasAttribute('data-message-id')) {
    if (target == document.body) {
      return null;
    }
    target = target.parentElement;
  }
  return target.getAttribute('data-message-id');
}

function affectedMessageDescendants(mutation) {
  let addedNodes = Array.from(mutation.addedNodes).filter(node => node.nodeType === Node.ELEMENT_NODE);
  let removedNodes = Array.from(mutation.removedNodes).filter(node => node.nodeType === Node.ELEMENT_NODE);
  let descendants = [...addedNodes, ...removedNodes].map(node => {
    return Array.from(node.querySelectorAll('[data-message-id]'))
  }).flat();
  return descendants.map(node => node.getAttribute('data-message-id'));
}

function recursivelyProcessMessage(node, processText) {
  if (node.nodeType === Node.TEXT_NODE) {
    const after = processText(node.textContent);
    node.textContent = after;
  }
  let i = 0;
  for (const child of node.childNodes) {
    recursivelyProcessMessage(child, processText);
  }
}

function modifyMessage(messageNode, processText) {
  let modifiedId = messageNode.getAttribute('data-message-id') + '-modified';
  let previous = document.querySelector(`[data-message-id="${modifiedId}"]`);

  if (processText(messageNode.innerText) === messageNode.innerText || (previous && processText(previous.innerText) === messageNode.innerText)) {
    console.log("Skipped");
    return;
  }
  
  let modified = messageNode.cloneNode(true);
  modified.style.display = previous ? previous.style.display : messageNode.style.display;
  modified.setAttribute('data-message-id', modifiedId);
  recursivelyProcessMessage(modified, processText);

  if (previous) {
    previous.remove();
  }
  messageNode.parentElement.prepend(modified);
  if (messageNode.style.display !== 'none') {
    messageNode.style.display = 'none';
  }
  // messageNode.style.height = '0px';
  // messageNode.style.overflow = 'hidden';
}

function setupPage(redact, restore) {
  const promptInterceptor = new PromptInterceptor(redact);
  // const existingMessageModifier = new MessageModifier(restore);

  promptInterceptor.setup();
  document.querySelectorAll('[data-message-id]').forEach(message => {
    console.log(message);
    modifyMessage(message, restore);
  });
  // existingMessageModifier.setup();
  
  let elementObserver = new MutationObserver(async (mutations) => {
    if (mutations.reduce((acc, mutation) => {
      const hasButton = mutation.target.id === 'composer-submit-button' || !!mutation.target.querySelector('#composer-submit-button');
      const hasPrompt = mutation.target.id === 'prompt-textarea';
      return acc || hasPrompt || hasButton
    }, false)) {
      promptInterceptor.setup();
    }
    // console.log(document.querySelectorAll('[data-message-id]'));
    // for (const message of document.querySelectorAll('[data-message-id]')) {
    //   // if (!message.getAttribute('data-message-id').includes('-modified')) {
    //   //   modifyMessage(message, restore);
    //   // }
    // }
    // existingMessageModifier.setup();
    let mutationMessages = mutations.map(mutation => {
      let message = {};
      let ancestor = affectedMessageAncestor(mutation);
      if (ancestor) {
        message.ancestor = ancestor;
      }
      let descendants = affectedMessageDescendants(mutation);
      if (descendants.length > 0) {
        message.descendants = descendants;
      }
      let self = mutation.target.getAttribute('data-message-id');
      if (self) {
        message.self = self;
      }
      return [mutation, message];
    }).filter(([mutation, message]) => Object.keys(message).length > 0);
    if (mutationMessages.length > 0) {
      console.log(mutations)
      console.log(mutationMessages);
      let allAffectedMessageIds = mutationMessages.map(([mutation, message]) => {
        return [message.ancestor, ...(message.descendants || []), message.self].filter(id => id);
      }).flat();
      allAffectedMessageIds = Array.from(new Set(allAffectedMessageIds));
      allAffectedMessageIds = allAffectedMessageIds.filter(id => !id.includes('-modified'));
      console.log(allAffectedMessageIds);
      let allAffectedMessages = allAffectedMessageIds.map(messageId => document.querySelector(`[data-message-id="${messageId}"]`));
      console.log(allAffectedMessages);
      allAffectedMessages.forEach(message => {
        modifyMessage(message, restore);
      });
    }
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
