let showRedaction = false;

// Global popup dimensions
const POPUP_WIDTH = 300;
const POPUP_HEIGHT = 300;

function addToggleButton() {
  if (document.querySelector('#toggle-redaction-slider')) {
    return;
  }

  const onToggle = () => {
    showRedaction = !showRedaction;
    for (const message of document.querySelectorAll('[data-message-id]')) {
      if (message.getAttribute('data-message-id').includes('-modified')) {
        message.style.display = showRedaction ? 'none' : '';
      } else {
        message.style.display = showRedaction ? '' : 'none';
      }
    }
  }
  
  // Create slider container
  const sliderContainer = document.createElement('div');
  sliderContainer.setAttribute("id", "toggle-redaction-slider");
  sliderContainer.style.cssText = `
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px;
  `;
  
  // Create label text
  const label = document.createElement('span');
  label.textContent = 'Show Redaction';
  label.style.cssText = `
    font-size: 14px;
    color: rgb(255, 255, 255);
    font-family: ui-sans-serif, -apple-system, system-ui, Segoe UI, Helvetica, Apple Color Emoji, Arial, sans-serif, Segoe UI Emoji, Segoe UI Symbol;
  `;
  
  // Create slider input
  const slider = document.createElement('input');
  slider.type = 'checkbox';
  slider.id = 'redaction-slider-input';
  slider.style.display = 'none';
  slider.checked = showRedaction;
  
  // Create slider visual
  const sliderVisual = document.createElement('label');
  sliderVisual.htmlFor = 'redaction-slider-input';
  sliderVisual.style.cssText = `
    position: relative;
    display: inline-block;
    width: 40px;
    height: 20px;
    background-color: ${showRedaction ? '#4CAF50' : '#ccc'};
    border-radius: 20px;
    cursor: pointer;
    transition: background-color 0.3s;
  `;
  
  // Create slider handle
  const sliderHandle = document.createElement('span');
  sliderHandle.style.cssText = `
    position: absolute;
    top: 2px;
    left: ${showRedaction ? '22px' : '2px'};
    width: 16px;
    height: 16px;
    background-color: white;
    border-radius: 50%;
    transition: left 0.3s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  `;
  
  sliderVisual.appendChild(sliderHandle);
  
  // Add event listener
  slider.addEventListener('change', () => {
    onToggle();
    sliderVisual.style.backgroundColor = showRedaction ? '#4CAF50' : '#ccc';
    sliderHandle.style.left = showRedaction ? '22px' : '2px';
  });
  
  // Assemble the slider
  sliderContainer.appendChild(label);
  sliderContainer.appendChild(slider);
  sliderContainer.appendChild(sliderVisual);
  
  const container = document.querySelector('[data-testid="composer-footer-actions"]')
  if (container) {
    container.appendChild(sliderContainer);
  }
}

function createPopupOverlay(id, content) {
  // Remove existing popup if it exists
  const existing = document.getElementById(id);
  if (existing) {
    existing.remove();
  }

  // Create overlay
  const overlay = document.createElement('div');
  overlay.id = id;
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
    font-family: ui-sans-serif, -apple-system, system-ui, Segoe UI, Helvetica, Apple Color Emoji, Arial, sans-serif, Segoe UI Emoji, Segoe UI Symbol;
  `;

  // Create popup container
  const popup = document.createElement('div');
  popup.style.cssText = `
    width: ${POPUP_WIDTH}px;
    height: ${POPUP_HEIGHT}px;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 20px;
    box-sizing: border-box;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  `;

  popup.innerHTML = content;
  overlay.appendChild(popup);
  document.body.appendChild(overlay);
  
  return overlay;
}

function showInitializingPopup() {
  const content = `
    <div style="display: flex; flex-direction: column; align-items: center; gap: 16px;">
      <div style="width: 96px; height: 96px; background-image: url('${chrome.runtime.getURL('logo-shield.png')}'); background-size: contain; background-repeat: no-repeat; animation: pulse 2s ease-in-out infinite;"></div>
      <div style="color: white; font-size: 16px; font-weight: 500; text-align: center; animation: pulse 2s ease-in-out infinite;">
        Initializing PocketShield
      </div>
    </div>
    <style>
      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(0.95); }
      }
    </style>
  `;
  
  createPopupOverlay('pocketshield-initializing-popup', content);
}

function hideInitializingPopup() {
  const popup = document.getElementById('pocketshield-initializing-popup');
  if (popup) {
    popup.remove();
  }
}

function showDownloadPocketshieldPopup() {
  const content = `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; gap: 24px;">
      <div style="color: white; font-size: 14px; line-height: 1.4; max-width: 260px;">
        It seems like the PocketShield app isn't running. If you haven't, 
        <a href="https://www.thirdai.com/pocketshield/" 
           style="color: #4A90E2; text-decoration: underline;" 
           target="_blank">download the app</a> and open it.
      </div>
      <div style="color: #AAA; font-size: 16px; animation: pulse 2s ease-in-out infinite;">
        waiting
      </div>
    </div>
    <style>
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
      }
    </style>
  `;
  
  createPopupOverlay('pocketshield-download-popup', content);
}

function hideDownloadPocketshieldPopup() {
  const popup = document.getElementById('pocketshield-download-popup');
  if (popup) {
    popup.remove();
  }
}

function showLoadSuccessPopup() {
  const content = `
    <div style="display: flex; flex-direction: column; align-items: center; gap: 16px;">
       <div style="width: 96px; height: 96px; background-image: url('${chrome.runtime.getURL('logo-shield.png')}'); background-size: contain; background-repeat: no-repeat;"></div>
       <div style="color: white; font-size: 16px; font-weight: 500; text-align: center; margin-bottom: 8px;">
         Your chat is secured.
       </div>
       <button id="success-popup-button" style="
         background-color: #4A90E2;
         color: white;
         border: none;
         padding: 8px 24px;
         border-radius: 20px;
         font-size: 14px;
         font-weight: bold;
         cursor: pointer;
         transition: background-color 0.3s;
       " onmouseover="this.style.backgroundColor='#357ABD'" onmouseout="this.style.backgroundColor='#4A90E2'">
         Let's go!
       </button>
     </div>
   `;
   
  const popup = createPopupOverlay('pocketshield-success-popup', content);
  
  // Add event listener programmatically
  const button = popup.querySelector('#success-popup-button');
  if (button) {
    button.addEventListener('click', hideLoadSuccessPopup);
  }
}

function hideLoadSuccessPopup() {
  const popup = document.getElementById('pocketshield-success-popup');
  if (popup) {
    popup.remove();
  }
}

DO_NOT_SHOW_FILE_NOT_SANITIZED_WARNING = false;

function showFileNotSanitizedWarningPopup() {
  if (DO_NOT_SHOW_FILE_NOT_SANITIZED_WARNING) {
    return;
  }

  const content = `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; gap: 20px;">
      <div style="color: white; font-size: 14px; line-height: 1.4; max-width: 260px;">
        PocketShield does not redact sensitive information from files. Please copy relevant text into the prompt area instead.
      </div>
      <div style="display: flex; flex-direction: column; align-items: center; gap: 16px;">
        <button id="file-warning-understand-button" style="
          background-color: #4A90E2;
          color: white;
          border: none;
          padding: 8px 24px;
          border-radius: 20px;
          font-size: 14px;
          font-weight: bold;
          cursor: pointer;
          transition: background-color 0.3s;
        " onmouseover="this.style.backgroundColor='#357ABD'" onmouseout="this.style.backgroundColor='#4A90E2'">
          I understand
        </button>
        <div style="display: flex; align-items: center; gap: 8px;">
          <input type="checkbox" id="dont-show-again-checkbox" style="display: none;">
          <label for="dont-show-again-checkbox" class="checkbox-label" style="
            display: flex;
            align-items: center;
            gap: 8px;
            color: #AAA;
            font-size: 12px;
            cursor: pointer;
            user-select: none;
          ">
            <div class="custom-checkbox" style="
              width: 16px;
              height: 16px;
              border: 2px solid #AAA;
              border-radius: 3px;
              display: flex;
              align-items: center;
              justify-content: center;
              background-color: transparent;
              transition: all 0.3s ease;
            ">
              <div class="checkbox-check" style="
                width: 8px;
                height: 8px;
                background-color: #4A90E2;
                border-radius: 1px;
                opacity: 0;
                transform: scale(0);
                transition: all 0.3s ease;
              "></div>
            </div>
            Don't show this again
          </label>
        </div>
        <style>
          #dont-show-again-checkbox:checked + .checkbox-label .custom-checkbox {
            border-color: #4A90E2 !important;
            background-color: rgba(74, 144, 226, 0.1) !important;
          }
          #dont-show-again-checkbox:checked + .checkbox-label .checkbox-check {
            opacity: 1 !important;
            transform: scale(1) !important;
          }
          .checkbox-label:hover .custom-checkbox {
            border-color: #4A90E2 !important;
          }
        </style>

      </div>
    </div>
  `;
  
  const popup = createPopupOverlay('pocketshield-file-warning-popup', content);
  
  // Add event listeners
  const button = popup.querySelector('#file-warning-understand-button');
  const checkbox = popup.querySelector('#dont-show-again-checkbox');
  const checkboxLabel = popup.querySelector('.checkbox-label');
  
  if (button) {
    button.addEventListener('click', () => {
      if (checkbox && checkbox.checked) {
        DO_NOT_SHOW_FILE_NOT_SANITIZED_WARNING = true;
      }
      hideFileNotSanitizedWarningPopup();
    });
  }
  
  // Make the custom checkbox clickable
  if (checkboxLabel && checkbox) {
    checkboxLabel.addEventListener('click', (e) => {
      e.preventDefault();
      checkbox.checked = !checkbox.checked;
    });
  }
}

function hideFileNotSanitizedWarningPopup() {
  const popup = document.getElementById('pocketshield-file-warning-popup');
  if (popup) {
    popup.remove();
  }
}

// Make hideLoadSuccessPopup available globally
window.hideLoadSuccessPopup = hideLoadSuccessPopup;

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
    let modified = messageNode.cloneNode(true);
    modified.style.display = '';
    modified.setAttribute('data-message-id', modifiedId);
    modified.setAttribute('data-timestamp', timestamp);
    await this.recursivelyProcessMessage(modified, timestamp, messageId);
  
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
    if (messageNode.style.display !== 'none' && !showRedaction) {
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

function cloneDropEvent(originalEvent) {
  const init = {
    bubbles: originalEvent.bubbles,
    cancelable: originalEvent.cancelable,
    composed: originalEvent.composed,
    clientX: originalEvent.clientX,
    clientY: originalEvent.clientY,
    screenX: originalEvent.screenX,
    screenY: originalEvent.screenY,
    ctrlKey: originalEvent.ctrlKey,
    shiftKey: originalEvent.shiftKey,
    altKey: originalEvent.altKey,
    metaKey: originalEvent.metaKey,
    button: originalEvent.button,
    buttons: originalEvent.buttons,
    relatedTarget: originalEvent.relatedTarget,
    dataTransfer: new DataTransfer(),
    // dataTransfer is intentionally excluded
  };

  return new DragEvent('drop', init);
}

function processUploadedFiles(files) {
  console.log("Uploaded files", files);
  Array.from(files).forEach(file => {
    file.text().then(text => {
      console.log("File text", text);
    });
  })
  // TODO: Process uploaded files.
}

class FileUploadInterceptor {
  static dropArea = null;

  setup() {
    this.setupManualUpload();
    this.setupDropArea();
  }

  setupManualUpload() {
    const file = document.querySelector('input[type="file"]');
    if (file) {
      file.removeEventListener('change', this.manualUploadEventListener, true);
      file.addEventListener('change', this.manualUploadEventListener, true);
    }
  }

  manualUploadEventListener(e) {
    e.stopPropagation();
    e.preventDefault();
    processUploadedFiles(e.target.files);
  }

  setupDropArea() {
    FileUploadInterceptor.dropArea = document.querySelector("div[role='presentation']");
    if (FileUploadInterceptor.dropArea) {
      FileUploadInterceptor.dropArea.removeEventListener('drop', this.dropEventListener, true);
      FileUploadInterceptor.dropArea.addEventListener('drop', this.dropEventListener, true);
    }
  }

  dropEventListener(e) {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      e.stopPropagation();
      e.preventDefault();
      processUploadedFiles(e.dataTransfer.files);
      if (FileUploadInterceptor.dropArea) {
        FileUploadInterceptor.dropArea.dispatchEvent(cloneDropEvent(e));
      }
    }
  }
}


async function setupPage(redact, restore) {  
  const promptInterceptor = new PromptInterceptor(redact);
  const messageModifier = new MessageModifier(restore);
  const fileUploadInterceptor = new FileUploadInterceptor();
  
  // Setup prompt interceptor and message modifier if they have been loaded at this time,
  // e.g. if the page was loaded before the extension.
  addToggleButton();
  promptInterceptor.setup();
  fileUploadInterceptor.setup();
  for (const message of document.querySelectorAll('[data-message-id]')) {
    if (!message.getAttribute('data-message-id').includes('-modified')) {
      await messageModifier.modifyMessage(message, Date.now());
    }
  }

  let dropArea = null;

  const dropEventListener = (e) => {
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      // TODO: Process the dropped files.
      if (dropArea) {
        dropArea.dispatchEvent(cloneDropEvent(e));
      }
      e.stopPropagation();
      e.preventDefault();
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
    fileUploadInterceptor.setup();
    addToggleButton();
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

async function backendIsHealthy() {
  return (await fetch('http://localhost:16549/api/v1/health').then(() => ({ok: true})).catch(() => ({ ok: false }))).ok;
}

async function healthCheck() {
  showInitializingPopup();
  let healthy = await backendIsHealthy();
  if (healthy) {
    hideInitializingPopup();
    showLoadSuccessPopup();
    return;
  }
  hideInitializingPopup();
  
  while (!healthy) {
    showDownloadPocketshieldPopup();
    await new Promise(resolve => setTimeout(resolve, 1000));
    healthy = await backendIsHealthy();
    console.log("health check done", healthy)
  }
  hideDownloadPocketshieldPopup();
  showLoadSuccessPopup();
  
}

async function constantHealthCheck() {
  let lastHealthy = true;
  while (true) {
    if (!(await backendIsHealthy())) {
      showDownloadPocketshieldPopup();
      lastHealthy = false;
    } else {
      if (!lastHealthy) {
        hideDownloadPocketshieldPopup();
        showLoadSuccessPopup();
        lastHealthy = true;
      }
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function initialize() {
  await healthCheck();
  constantHealthCheck();

  const locationTracker = new LocationTracker();
  let wasmRedactor = await newWasmRedactor('wasm/build/');
  const makeRedact = (sessionId) => (text) => wasmRedactor.redact(text, sessionId);
  // TODO: Can we reduce initial loading time by running restore asynchronously?
  const makeRestore = (sessionId) => async (text) => {
    const response = await fetch(`http://localhost:16549/api/v1/chat/sessions/${sessionId}/restore`, {
      method: 'POST',
      body: JSON.stringify({ Message: text }),
    }).then(response => response.json()).then(data => data.Message).catch(() => {
      console.error("Error restoring message", text);
      return text;
    });
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