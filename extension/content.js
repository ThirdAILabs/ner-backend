// Content script for NER Chrome Extension
// Runs on all web pages to provide text extraction and highlighting capabilities

class NERContentScript {
    constructor() {
        this.isInitialized = false;
        this.highlightedElements = [];
        this.init();
    }

    async init() {
        if (this.isInitialized) return;
        
        console.log('NER Content Script loaded');
        this.setupMessageListener();
        
        // Check if auto-analysis is enabled
        const { autoAnalyze } = await chrome.storage.sync.get(['autoAnalyze']);
        if (autoAnalyze && this.isPageReadyForAnalysis()) {
            this.notifyBackgroundForAnalysis();
        }
        
        this.isInitialized = true;
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep the message channel open
        });
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'extractText':
                    const text = this.extractPageText();
                    sendResponse({ success: true, text: text });
                    break;
                    
                case 'highlightEntities':
                    this.highlightEntities(request.entities);
                    sendResponse({ success: true });
                    break;
                    
                case 'clearHighlights':
                    this.clearHighlights();
                    sendResponse({ success: true });
                    break;
                    
                case 'getPageInfo':
                    const info = this.getPageInfo();
                    sendResponse({ success: true, info: info });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Content script message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    isPageReadyForAnalysis() {
        // Skip non-content pages
        if (window.location.protocol === 'chrome-extension:' ||
            window.location.protocol === 'chrome:' ||
            window.location.protocol === 'moz-extension:') {
            return false;
        }
        
        // Check if page has meaningful content
        const textContent = document.body.innerText.trim();
        return textContent.length > 100;
    }

    notifyBackgroundForAnalysis() {
        // Send a message to background script for auto-analysis
        chrome.runtime.sendMessage({
            action: 'pageReady',
            url: window.location.href,
            title: document.title
        }).catch(error => {
            console.error('Failed to notify background:', error);
        });
    }

    extractPageText() {
        // Remove unwanted elements temporarily
        const unwantedElements = document.querySelectorAll(
            'script, style, nav, header, footer, .advertisement, .ad, .sidebar, .menu, .nav'
        );
        
        const hiddenElements = [];
        unwantedElements.forEach(el => {
            if (el.style.display !== 'none') {
                el.style.display = 'none';
                hiddenElements.push(el);
            }
        });

        // Try to find main content area
        const contentSelectors = [
            'main', 'article', '.content', '.post', '.entry-content',
            '[role="main"]', '.main-content', '.article-body', '.post-content'
        ];
        
        let mainContent = null;
        for (const selector of contentSelectors) {
            const element = document.querySelector(selector);
            if (element && element.innerText.trim().length > 50) {
                mainContent = element;
                break;
            }
        }
        
        // Extract text from main content or body
        const targetElement = mainContent || document.body;
        let text = targetElement.innerText;
        
        // Clean up the text
        text = text
            .replace(/\s+/g, ' ')  // Replace multiple whitespace with single space
            .replace(/\n\s*\n/g, '\n')  // Remove empty lines
            .trim();
        
        // Restore hidden elements
        hiddenElements.forEach(el => {
            el.style.display = '';
        });
        
        return text;
    }

    getPageInfo() {
        return {
            title: document.title,
            url: window.location.href,
            domain: window.location.hostname,
            textLength: document.body.innerText.length,
            hasMainContent: !!document.querySelector('main, article, .content'),
            language: document.documentElement.lang || 'unknown'
        };
    }

    highlightEntities(entities) {
        if (!entities || entities.length === 0) return;
        
        // Clear existing highlights first
        this.clearHighlights();
        
        // Create a tree walker to find text nodes
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: (node) => {
                    // Skip nodes in script, style, or already highlighted elements
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;
                    
                    const tagName = parent.tagName.toLowerCase();
                    if (['script', 'style', 'noscript'].includes(tagName)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    
                    if (parent.classList.contains('ner-highlight')) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );
        
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        // Sort entities by length (longest first) to avoid overlapping issues
        const sortedEntities = entities.sort((a, b) => b.text.length - a.text.length);
        
        sortedEntities.forEach(entity => {
            this.highlightEntityInNodes(textNodes, entity);
        });
    }

    highlightEntityInNodes(textNodes, entity) {
        const entityColors = {
            'PERSON': '#FFE4E1',
            'ORGANIZATION': '#E6F3FF',
            'LOCATION': '#E6FFE6',
            'DATE': '#FFF0E6',
            'MONEY': '#F0E6FF',
            'TIME': '#FFFFE6',
            'MISC': '#F5F5F5'
        };
        
        textNodes.forEach(textNode => {
            if (!textNode.parentNode) return; // Node might have been removed
            
            const text = textNode.textContent;
            const entityText = entity.text;
            const regex = new RegExp(this.escapeRegExp(entityText), 'gi');
            const matches = [...text.matchAll(regex)];
            
            if (matches.length === 0) return;
            
            // Process matches in reverse order to maintain indices
            matches.reverse().forEach(match => {
                const start = match.index;
                const end = start + match[0].length;
                
                // Split the text node
                const beforeText = text.substring(0, start);
                const matchText = text.substring(start, end);
                const afterText = text.substring(end);
                
                // Create highlight element
                const highlight = document.createElement('span');
                highlight.className = 'ner-highlight';
                highlight.style.cssText = `
                    background-color: ${entityColors[entity.label] || entityColors['MISC']};
                    padding: 1px 2px;
                    border-radius: 2px;
                    font-weight: 500;
                    border-bottom: 1px solid rgba(0,0,0,0.1);
                    position: relative;
                    cursor: help;
                `;
                highlight.textContent = matchText;
                highlight.title = `${entity.label}: ${entity.text}${entity.confidence ? ` (${Math.round(entity.confidence * 100)}%)` : ''}`;
                
                // Replace the text node with the new elements
                const parent = textNode.parentNode;
                
                if (beforeText) {
                    parent.insertBefore(document.createTextNode(beforeText), textNode);
                }
                
                parent.insertBefore(highlight, textNode);
                
                if (afterText) {
                    parent.insertBefore(document.createTextNode(afterText), textNode);
                    // Update textNode for next iteration
                    textNode.textContent = afterText;
                } else {
                    parent.removeChild(textNode);
                }
                
                this.highlightedElements.push(highlight);
            });
        });
    }

    clearHighlights() {
        // Remove all highlight elements and restore original text
        this.highlightedElements.forEach(highlight => {
            if (highlight.parentNode) {
                const textNode = document.createTextNode(highlight.textContent);
                highlight.parentNode.replaceChild(textNode, highlight);
            }
        });
        
        // Also clean up any remaining highlights (safety net)
        const remainingHighlights = document.querySelectorAll('.ner-highlight');
        remainingHighlights.forEach(highlight => {
            if (highlight.parentNode) {
                const textNode = document.createTextNode(highlight.textContent);
                highlight.parentNode.replaceChild(textNode, highlight);
            }
        });
        
        this.highlightedElements = [];
        
        // Normalize text nodes to merge adjacent text nodes
        this.normalizeTextNodes(document.body);
    }

    normalizeTextNodes(element) {
        let child = element.firstChild;
        while (child) {
            if (child.nodeType === Node.TEXT_NODE) {
                // Merge with next text node if it exists
                while (child.nextSibling && child.nextSibling.nodeType === Node.TEXT_NODE) {
                    child.textContent += child.nextSibling.textContent;
                    element.removeChild(child.nextSibling);
                }
            } else if (child.nodeType === Node.ELEMENT_NODE) {
                this.normalizeTextNodes(child);
            }
            child = child.nextSibling;
        }
    }

    escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}

// Initialize the content script
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new NERContentScript();
    });
} else {
    new NERContentScript();
} 