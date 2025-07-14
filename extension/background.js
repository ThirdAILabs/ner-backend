console.log("Background script loaded");

function injectContentScriptOnMatchingTabs() {
  chrome.tabs.query({ url: "https://chatgpt.com/*" }, (tabs) => {
    for (const tab of tabs) {
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["wasm/wasm-loader.js", "wasm/markitdown-loader.js", "content.js"]
      });
    }
  });
}

injectContentScriptOnMatchingTabs();
chrome.runtime.onInstalled.addListener(injectContentScriptOnMatchingTabs);
chrome.runtime.onStartup.addListener(injectContentScriptOnMatchingTabs);