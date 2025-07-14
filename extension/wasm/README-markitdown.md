# MarkItDown WASM Module

This directory contains the WebAssembly (WASM) compiled version of the MarkItDown Python library, designed to run in browser environments.

## Files

- `markitdown_wasm.py` - Original Python source code that was compiled to WASM
- `markitdown_wasm.wasm` - Compiled WebAssembly module (68MB)
- `markitdown-loader.js` - JavaScript loader and interface for the WASM module
- `test-markitdown.html` - Test page to verify the WASM module works correctly

## How It Works

The MarkItDown Python library has been compiled to WebAssembly using `py2wasm`, which creates a standalone executable that can run in browsers. The module:

1. **Compiles Python to WASM**: Uses `py2wasm` with Python 3.11 and Nuitka to create a WASM executable
2. **Provides WASI Interface**: The WASM module uses the WebAssembly System Interface (WASI) 
3. **Runs as Executable**: Unlike traditional WASM modules, py2wasm creates standalone executables
4. **Handles File Processing**: Can convert various file types to markdown using the MarkItDown library

## Usage

### Basic Usage

```javascript
import MarkItDownWasm from './markitdown-loader.js';

const wasmModule = new MarkItDownWasm();

// Initialize the module
const success = await wasmModule.initialize('./markitdown_wasm.wasm');

if (success) {
    // Parse a file
    const result = await wasmModule.parse('document.txt', 'Hello, World!');
    console.log(result);
}
```

### Browser Extension Integration

```javascript
// In a browser extension context
const wasmModule = new MarkItDownWasm();
const wasmPath = 'wasm/markitdown_wasm.wasm';

try {
    await wasmModule.initialize(wasmPath);
    
    // Parse file content
    const markdown = await wasmModule.parse(fileName, fileContents);
    
    // Use the converted markdown
    console.log('Converted to markdown:', markdown);
} catch (error) {
    console.error('WASM parsing failed:', error);
    
    // Fallback to basic parsing
    const fallbackResult = wasmModule.fallbackParse(fileName, fileContents);
    console.log('Fallback result:', fallbackResult);
}
```

### Error Handling

The WASM module includes comprehensive error handling:

```javascript
try {
    const result = await wasmModule.parse('document.docx', wordContent);
    return result;
} catch (error) {
    console.error('WASM error:', error.message);
    
    // Check stderr for detailed error information
    const stderr = wasmModule.getStderr();
    if (stderr) {
        console.error('WASM stderr:', stderr);
    }
    
    // Use fallback parsing
    return wasmModule.fallbackParse('document.docx', wordContent);
}
```

## Compilation Process

The WASM module was compiled using the following steps:

1. **Set up Python 3.11 environment**:
   ```bash
   conda create -n py2wasm python=3.11 -y
   conda activate py2wasm
   ```

2. **Install dependencies**:
   ```bash
   pip install py2wasm markitdown
   ```

3. **Compile to WASM**:
   ```bash
   py2wasm markitdown_wasm.py -o markitdown_wasm.wasm
   ```

## Technical Details

### WASI Implementation

The JavaScript loader implements the following WASI functions:

- **Process Management**: `proc_exit`, `args_get`, `args_sizes_get`
- **I/O Operations**: `fd_write`, `fd_read`, `fd_close`
- **Environment**: `environ_get`, `environ_sizes_get`
- **System Calls**: `clock_time_get`, `random_get`
- **File System**: Basic stubs for file operations

### Memory Management

The WASM module manages its own memory:

- **Automatic Memory**: Memory is managed by the WASM runtime
- **String Handling**: Text encoding/decoding between JavaScript and WASM
- **Buffer Management**: Efficient handling of file content buffers

### Performance Considerations

- **Module Size**: 68MB (includes Python runtime and dependencies)
- **Initialization Time**: ~2-3 seconds for first load
- **Parsing Speed**: Depends on file size and complexity
- **Memory Usage**: Scales with file size being processed

## Supported File Types

The MarkItDown library supports various file formats:

- **Text Files**: `.txt`, `.md`, `.rst`
- **Office Documents**: `.docx`, `.xlsx`, `.pptx`
- **PDFs**: `.pdf`
- **HTML**: `.html`, `.htm`
- **Images**: `.png`, `.jpg`, `.jpeg` (with OCR)
- **Audio**: `.mp3`, `.wav` (with transcription)
- **Archives**: `.zip`, `.tar`

## Browser Compatibility

The WASM module requires:

- **WebAssembly Support**: All modern browsers (Chrome 57+, Firefox 52+, Safari 11+)
- **WASI Support**: Implemented by the JavaScript loader
- **ES6 Modules**: For the loader interface
- **TextEncoder/TextDecoder**: For string handling
- **Crypto API**: For random number generation

## Testing

Use the included test page to verify functionality:

1. Open `test-markitdown.html` in a browser
2. Click "Initialize WASM" to load the module
3. Test parsing with different file types
4. Run performance tests to measure speed

## Limitations

- **File Size**: Large files may cause memory issues
- **Async Only**: All operations are asynchronous
- **No File System**: Cannot read from local file system directly
- **Limited Error Handling**: Some Python errors may not be caught properly

## Fallback Strategy

The loader includes a fallback parser for cases where WASM fails:

```javascript
// Automatic fallback on error
const result = wasmModule.fallbackParse(fileName, fileContents);
```

The fallback provides basic text-to-markdown conversion for reliability.

## Integration with Extension

To integrate with the browser extension:

1. **Add to Manifest**: Include the WASM file in web_accessible_resources
2. **Load Module**: Initialize the module when the extension starts
3. **Process Files**: Use the parse function to convert documents
4. **Handle Errors**: Implement fallback strategies for reliability

## Future Improvements

- **Size Optimization**: Reduce WASM module size through tree shaking
- **Streaming**: Support for streaming large files
- **Caching**: Cache compiled module for faster initialization
- **Worker Integration**: Run WASM in Web Workers for better performance
- **Progressive Loading**: Load module components on demand 