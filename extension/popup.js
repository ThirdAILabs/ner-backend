// Popup functionality for NER Chrome Extension

document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        extractText: document.getElementById('extractText'),
        highlightEntities: document.getElementById('highlightEntities'),
        clearHighlights: document.getElementById('clearHighlights'),
        stats: document.getElementById('stats'),
        entitiesContainer: document.getElementById('entitiesContainer'),
        entitiesList: document.getElementById('entitiesList'),
        textLength: document.getElementById('textLength'),
        wordCount: document.getElementById('wordCount'),
        entityCount: document.getElementById('entityCount'),
        autoAnalyze: document.getElementById('autoAnalyze'),
        backendUrl: document.getElementById('backendUrl'),
        status: document.getElementById('status')
    };

    let backendUrl = 'http://localhost:8080';

    // Load settings
    chrome.storage.sync.get(['autoAnalyze', 'backendUrl'], (result) => {
        if (result.autoAnalyze !== undefined) {
            elements.autoAnalyze.checked = result.autoAnalyze;
        }
        if (result.backendUrl) {
            backendUrl = result.backendUrl;
            elements.backendUrl.value = result.backendUrl;
        }
    });

    // Save settings
    const saveSettings = () => {
        const settings = {
            autoAnalyze: elements.autoAnalyze.checked,
            backendUrl: elements.backendUrl.value
        };
        chrome.storage.sync.set(settings);
        backendUrl = settings.backendUrl;
        updateStatus('Settings saved');
    };

    // Event listeners
    elements.extractText.addEventListener('click', extractText);
    elements.highlightEntities.addEventListener('click', highlightEntities);
    elements.clearHighlights.addEventListener('click', clearHighlights);
    elements.autoAnalyze.addEventListener('change', saveSettings);
    elements.backendUrl.addEventListener('change', saveSettings);

    // Extract text from page
    async function extractText() {
        try {
            updateStatus('Extracting text...', 'loading');
            
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: getPageText
            });
            
            const pageText = results[0].result;
            if (!pageText || pageText.trim().length === 0) {
                updateStatus('No text found on this page', 'error');
                return;
            }

            const wordCount = pageText.trim().split(/\s+/).length;
            elements.textLength.textContent = pageText.length;
            elements.wordCount.textContent = wordCount;
            elements.stats.style.display = 'block';
            
            await analyzeText(pageText);
        } catch (error) {
            console.error('Failed to extract text:', error);
            updateStatus('Failed to extract text', 'error');
        }
    }

    // Get page text content
    function getPageText() {
        const scripts = document.querySelectorAll('script, style, noscript');
        scripts.forEach(el => el.remove());
        
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            node => {
                const parent = node.parentElement;
                if (!parent) return NodeFilter.FILTER_REJECT;
                const style = window.getComputedStyle(parent);
                return (style.display === 'none' || style.visibility === 'hidden') 
                    ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
            }
        );
        
        let textContent = '';
        let node;
        while (node = walker.nextNode()) {
            const text = node.textContent.trim();
            if (text) textContent += text + ' ';
        }
        return textContent.trim();
    }

    // Analyze text with backend
    async function analyzeText(text) {
        try {
            updateStatus('Analyzing text...', 'loading');
            
            const response = await fetch(`${backendUrl}/api/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, model: 'default' })
            });
            
            if (!response.ok) throw new Error(`Backend error: ${response.status}`);
            
            const result = await response.json();
            displayEntities(result.entities || []);
            updateStatus('Analysis complete', 'success');
        } catch (error) {
            console.error('Failed to analyze text:', error);
            updateStatus('Backend not available', 'error');
            displayMockEntities();
        }
    }

    // Display entities
    function displayEntities(entities) {
        elements.entityCount.textContent = entities.length;
        
        if (entities.length === 0) {
            elements.entitiesContainer.style.display = 'none';
            return;
        }
        
        elements.entitiesList.innerHTML = '';
        entities.forEach(entity => {
            const div = document.createElement('div');
            div.className = 'entity-item';
            div.innerHTML = `
                <span class="entity-text" title="${entity.text}">${entity.text}</span>
                <span class="entity-type">${entity.label}</span>
            `;
            elements.entitiesList.appendChild(div);
        });
        
        elements.entitiesContainer.style.display = 'block';
        chrome.storage.local.set({ currentEntities: entities });
    }

    // Display mock entities for demo
    function displayMockEntities() {
        const mockEntities = [
            { text: 'John Doe', label: 'PERSON' },
            { text: 'New York', label: 'LOCATION' },
            { text: 'Google', label: 'ORGANIZATION' },
            { text: 'January 2024', label: 'DATE' }
        ];
        displayEntities(mockEntities);
    }

    // Highlight entities on page
    async function highlightEntities() {
        try {
            updateStatus('Highlighting entities...', 'loading');
            
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            const { currentEntities } = await chrome.storage.local.get(['currentEntities']);
            
            if (!currentEntities || currentEntities.length === 0) {
                updateStatus('No entities to highlight', 'error');
                return;
            }
            
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: highlightText,
                args: [currentEntities]
            });
            
            updateStatus('Entities highlighted', 'success');
        } catch (error) {
            console.error('Failed to highlight:', error);
            updateStatus('Failed to highlight', 'error');
        }
    }

    // Highlight text function
    function highlightText(entities) {
        const colors = {
            'PERSON': '#FFE4E1', 'ORGANIZATION': '#E6F3FF', 'LOCATION': '#E6FFE6',
            'DATE': '#FFF0E6', 'MONEY': '#F0E6FF', 'TIME': '#FFFFE6'
        };
        
        // Clear existing highlights
        document.querySelectorAll('.ner-highlight').forEach(el => {
            el.parentNode.replaceChild(document.createTextNode(el.textContent), el);
        });
        
        // Add highlights
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        const textNodes = [];
        let node;
        while (node = walker.nextNode()) textNodes.push(node);
        
        entities.forEach(entity => {
            textNodes.forEach(textNode => {
                const text = textNode.textContent;
                const index = text.toLowerCase().indexOf(entity.text.toLowerCase());
                
                if (index !== -1) {
                    const span = document.createElement('span');
                    span.className = 'ner-highlight';
                    span.style.backgroundColor = colors[entity.label] || '#F5F5F5';
                    span.style.padding = '2px 4px';
                    span.style.borderRadius = '3px';
                    span.style.fontWeight = 'bold';
                    span.textContent = text.substring(index, index + entity.text.length);
                    span.title = `${entity.label}: ${entity.text}`;
                    
                    const parent = textNode.parentNode;
                    parent.insertBefore(document.createTextNode(text.substring(0, index)), textNode);
                    parent.insertBefore(span, textNode);
                    parent.insertBefore(document.createTextNode(text.substring(index + entity.text.length)), textNode);
                    parent.removeChild(textNode);
                }
            });
        });
    }

    // Clear highlights
    async function clearHighlights() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    document.querySelectorAll('.ner-highlight').forEach(el => {
                        el.parentNode.replaceChild(document.createTextNode(el.textContent), el);
                    });
                }
            });
            updateStatus('Highlights cleared', 'success');
        } catch (error) {
            updateStatus('Failed to clear highlights', 'error');
        }
    }

    // Update status
    function updateStatus(message, type = 'default') {
        elements.status.textContent = message;
        elements.status.className = `status ${type}`;
        
        if (type !== 'loading') {
            setTimeout(() => {
                elements.status.textContent = 'Ready';
                elements.status.className = 'status';
            }, 3000);
        }
    }
}); 