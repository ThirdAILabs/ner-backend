# NER Text Analyzer Chrome Extension

A Chrome extension for extracting and analyzing text from web pages using Named Entity Recognition (NER). This extension works with your NER backend to identify and highlight entities like people, organizations, locations, dates, and more.

## Features

- **Text Extraction**: Extract meaningful text content from any web page
- **Entity Analysis**: Send text to your NER backend for entity recognition
- **Visual Highlighting**: Highlight detected entities directly on web pages
- **Auto-Analysis**: Automatically analyze pages when they load (optional)
- **Multiple Entity Types**: Support for PERSON, ORGANIZATION, LOCATION, DATE, MONEY, TIME, and more
- **Offline Fallback**: Mock entity detection when backend is unavailable
- **Customizable Settings**: Configure backend URL and analysis preferences

## Installation

1. Clone or download this extension to your local machine
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right corner
4. Click "Load unpacked" and select the extension directory
5. The NER Text Analyzer extension should now appear in your extensions list

## Setup

1. **Backend Configuration**: 
   - Make sure your NER backend is running (default: `http://localhost:8080`)
   - Update the backend URL in the extension settings if needed

2. **Permissions**: 
   - The extension will request permission to access web pages for text extraction
   - Grant the necessary permissions when prompted

## Usage

### Basic Usage

1. **Open the Extension**: Click the NER Text Analyzer icon in your browser toolbar
2. **Extract Text**: Click "Extract Page Text" to get text from the current page
3. **View Results**: See analysis results including text statistics and detected entities
4. **Highlight Entities**: Click "Highlight Entities" to visually mark entities on the page
5. **Clear Highlights**: Click "Clear Highlights" to remove visual markers

### Settings

- **Auto-analyze on page load**: Automatically analyze pages when they finish loading
- **Backend URL**: Configure the URL of your NER backend service

### Entity Types

The extension recognizes and highlights various entity types:

- **PERSON** (Pink highlight): Names of people
- **ORGANIZATION** (Blue highlight): Company and organization names  
- **LOCATION** (Green highlight): Geographic locations
- **DATE** (Orange highlight): Dates and time periods
- **MONEY** (Purple highlight): Monetary amounts
- **TIME** (Yellow highlight): Time expressions

## Backend Integration

The extension expects your NER backend to provide an endpoint at `/api/analyze` that accepts POST requests with the following format:

```json
{
  "text": "Text to analyze",
  "model": "default",
  "confidence_threshold": 0.5
}
```

Expected response format:

```json
{
  "entities": [
    {
      "text": "Entity text",
      "label": "ENTITY_TYPE",
      "confidence": 0.95,
      "start": 0,
      "end": 11
    }
  ],
  "metadata": {
    "model": "model_name",
    "timestamp": 1234567890
  }
}
```

## Files Structure

- `manifest.json` - Extension configuration and permissions
- `popup.html` - Extension popup interface
- `popup.css` - Styling for the popup
- `popup.js` - Popup functionality and user interactions
- `background.js` - Background service worker for extension logic
- `content.js` - Content script for page interaction and highlighting
- `content.css` - Styles for entity highlighting on web pages

## Development

### Adding New Entity Types

1. Update the entity colors in `popup.js` and `content.js`
2. Add corresponding CSS classes in `content.css`
3. Update the default settings in `background.js`

### Customizing Highlights

Modify the highlighting styles in `content.css` to change:
- Colors for different entity types
- Hover effects and tooltips
- Animation timing
- Print styles

### Backend Integration

To integrate with a different backend:

1. Update the API endpoint URL in the settings
2. Modify the request/response format in `popup.js` and `background.js`
3. Adjust entity processing logic as needed

## Troubleshooting

### Common Issues

1. **No entities detected**: 
   - Check that your backend is running and accessible
   - Verify the backend URL in settings
   - Try extracting text first to ensure the page has content

2. **Highlights not appearing**:
   - Make sure you've extracted text and found entities first
   - Check browser console for JavaScript errors
   - Try clearing and re-applying highlights

3. **Backend connection failed**:
   - Verify your backend is running on the configured port
   - Check for CORS issues in your backend configuration
   - The extension will show mock data when backend is unavailable

### Debug Mode

To debug the extension:

1. Open Chrome DevTools on the extension popup
2. Check the browser console for error messages
3. Use `chrome://extensions/` to view extension logs
4. Enable Developer mode for additional debugging options

## Privacy

This extension:
- Only processes text from pages you actively analyze
- Sends text to your configured backend (localhost by default)
- Stores settings and recent analysis locally in Chrome storage
- Does not collect or transmit personal data to external services

## License

This extension is provided as-is for use with your NER backend system. 