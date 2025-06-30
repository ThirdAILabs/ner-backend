// Background service worker for NER Chrome Extension

class NERExtensionBackground {
    constructor() {
        this.initializeExtension();
        this.setupEventListeners();
    }

    initializeExtension() {
        console.log('NER Extension background script loaded');
        
        // Set default settings on installation
        chrome.runtime.onInstalled.addListener((details) => {
            if (details.reason === 'install') {
                this.setDefaultSettings();
            }
        });
    }

    async setDefaultSettings() {
        const defaultSettings = {
            autoAnalyze: true,
            backendUrl: 'http://localhost:8080',
            highlightStyle: 'background',
            entityTypes: ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'MONEY', 'TIME']
        };
        
        try {
            await chrome.storage.sync.set(defaultSettings);
            console.log('Default settings initialized');
        } catch (error) {
            console.error('Failed to set default settings:', error);
        }
    }

    setupEventListeners() {
        // Handle messages from content scripts and popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep the message channel open for async responses
        });

        // Handle tab updates for auto-analysis
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url) {
                this.handleTabUpdate(tabId, tab);
            }
        });

        // Handle extension icon click
        chrome.action.onClicked.addListener((tab) => {
            this.handleIconClick(tab);
        });
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'analyzeText':
                    const result = await this.analyzeText(request.text, request.options);
                    sendResponse({ success: true, data: result });
                    break;
                    
                case 'getSettings':
                    const settings = await this.getSettings();
                    sendResponse({ success: true, data: settings });
                    break;
                    
                case 'saveSettings':
                    await this.saveSettings(request.settings);
                    sendResponse({ success: true });
                    break;
                    
                case 'extractPageText':
                    const text = await this.extractPageText(sender.tab.id);
                    sendResponse({ success: true, data: text });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async handleTabUpdate(tabId, tab) {
        try {
            // Check if auto-analysis is enabled
            const { autoAnalyze } = await chrome.storage.sync.get(['autoAnalyze']);
            
            if (!autoAnalyze) return;
            
            // Skip non-HTTP(S) pages
            if (!tab.url.startsWith('http://') && !tab.url.startsWith('https://')) {
                return;
            }
            
            // Wait a bit for the page to fully load
            setTimeout(async () => {
                try {
                    const text = await this.extractPageText(tabId);
                    if (text && text.length > 100) {
                        await this.analyzeText(text);
                        this.setBadgeText(tabId, '✓');
                    }
                } catch (error) {
                    console.error('Auto-analysis failed:', error);
                }
            }, 2000);
            
        } catch (error) {
            console.error('Tab update handling error:', error);
        }
    }

    async handleIconClick(tab) {
        // The popup will handle the interaction
        // This is just for additional logic if needed
        console.log('Extension icon clicked on tab:', tab.id);
    }

    async analyzeText(text, options = {}) {
        try {
            const { backendUrl } = await chrome.storage.sync.get(['backendUrl']);
            const url = backendUrl || 'http://localhost:8080';
            
            const response = await fetch(`${url}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    model: options.model || 'default',
                    confidence_threshold: options.confidence || 0.5
                })
            });
            
            if (!response.ok) {
                throw new Error(`Backend error: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Store the results for the popup
            await chrome.storage.local.set({
                lastAnalysis: {
                    timestamp: Date.now(),
                    text: text,
                    entities: result.entities || [],
                    metadata: result.metadata || {}
                }
            });
            
            return result;
            
        } catch (error) {
            console.error('Text analysis failed:', error);
            
            // Return mock data if backend is not available
            return {
                entities: this.generateMockEntities(text),
                metadata: {
                    source: 'mock',
                    timestamp: Date.now()
                }
            };
        }
    }

    generateMockEntities(text) {
        const mockPatterns = [
            { pattern: /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g, label: 'PERSON' },
            { pattern: /\b(?:New York|London|Paris|Tokyo|Berlin)\b/gi, label: 'LOCATION' },
            { pattern: /\b(?:Google|Microsoft|Apple|Amazon|Facebook)\b/gi, label: 'ORGANIZATION' },
            { pattern: /\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b/gi, label: 'DATE' }
        ];
        
        const entities = [];
        
        mockPatterns.forEach(({ pattern, label }) => {
            const matches = text.match(pattern) || [];
            matches.forEach(match => {
                entities.push({
                    text: match,
                    label: label,
                    confidence: 0.8 + Math.random() * 0.2,
                    start: text.indexOf(match),
                    end: text.indexOf(match) + match.length
                });
            });
        });
        
        return entities.slice(0, 20); // Limit to 20 entities
    }

    async extractPageText(tabId) {
        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: tabId },
                function: () => {
                    // Remove unwanted elements
                    const unwantedSelectors = [
                        'script', 'style', 'nav', 'header', 'footer',
                        '.advertisement', '.ad', '.sidebar', '.menu'
                    ];
                    
                    unwantedSelectors.forEach(selector => {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => el.remove());
                    });
                    
                    // Extract main content
                    const contentSelectors = [
                        'main', 'article', '.content', '.post', '.entry-content',
                        '[role="main"]', '.main-content'
                    ];
                    
                    let mainContent = null;
                    for (const selector of contentSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            mainContent = element;
                            break;
                        }
                    }
                    
                    const targetElement = mainContent || document.body;
                    return targetElement.innerText.trim();
                }
            });
            
            return results[0].result;
        } catch (error) {
            console.error('Failed to extract page text:', error);
            return null;
        }
    }

    async getSettings() {
        return await chrome.storage.sync.get([
            'autoAnalyze', 'backendUrl', 'highlightStyle', 'entityTypes'
        ]);
    }

    async saveSettings(settings) {
        await chrome.storage.sync.set(settings);
    }

    setBadgeText(tabId, text) {
        chrome.action.setBadgeText({
            tabId: tabId,
            text: text
        });
        
        chrome.action.setBadgeBackgroundColor({
            tabId: tabId,
            color: '#4CAF50'
        });
        
        // Clear badge after 3 seconds
        setTimeout(() => {
            chrome.action.setBadgeText({
                tabId: tabId,
                text: ''
            });
        }, 3000);
    }
}

// Initialize the background service
new NERExtensionBackground(); 