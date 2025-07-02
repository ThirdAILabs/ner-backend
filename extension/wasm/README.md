# WASM Redaction Module

This directory contains the WebAssembly (WASM) redaction module for the browser extension. The module provides privacy-focused text redaction and restoration capabilities using a pure C implementation compiled to WASM.

## Features

- **Redact sensitive information**: Email addresses, phone numbers, SSNs, credit card numbers, names
- **Restore redacted content**: Maintains mappings to restore original text
- **Extensible patterns**: Easy to add custom redaction patterns
- **Reliable deployment**: Emscripten WASM with JavaScript fallback
- **Memory efficient**: Proper cleanup and resource management

## Files Overview

- `redactor.c` - Main C implementation with redaction logic
- `Makefile` - Build configuration for compilation
- `wasm-loader.js` - JavaScript interface for loading and using the WASM module
- `test.html` - Test page for development and debugging
- `README.md` - This documentation

## Prerequisites

### Emscripten Build

1. **Install Emscripten**:
   ```bash
   # macOS
   brew install emscripten
   
   # Ubuntu/Debian
   sudo apt update
   sudo apt install emscripten
   
   # Manual installation
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. **Install Build Dependencies**:
   ```bash
   # Using the Makefile
   make install-deps-macos    # for macOS
   make install-deps-ubuntu   # for Ubuntu/Debian
   ```

## Building the Module

### Emscripten Build

1. **Navigate to the wasm directory**:
   ```bash
   cd extension/wasm
   ```

2. **Build the WASM module**:
   ```bash
   make all
   ```

3. **Verify the build**:
   ```bash
   ls -la build/
   # Should contain:
   # - redactor.js
   # - redactor.wasm
   # - redactor.data (if using embedded files)
   ```

## Testing

### Test C Module Locally

```bash
# Test the C redaction logic (compiles native version)
make test

# Expected output:
# Testing C redactor module...
# Original: Contact John Doe at john.doe@email.com or call 555-123-4567
# Redacted: Contact [NAME_1] at [EMAIL_1] or call [PHONE_1]
# Restored: Contact John Doe at john.doe@email.com or call 555-123-4567
# ✓ Test passed: Redaction and restoration work correctly
```

### Test WASM Module

1. **Start a development server**:
   ```bash
   make serve
   ```

2. **Open a browser and navigate to** `http://localhost:8000`

3. **Open browser console and test**:
   ```javascript
   // Initialize the redactor
   const redactor = new WasmRedactor();
   await redactor.initialize();
   
   // Test redaction
   const text = "Contact John Doe at john.doe@email.com";
   const redacted = redactor.redact(text);
   console.log("Redacted:", redacted);
   
   const restored = redactor.restore(redacted);
   console.log("Restored:", restored);
   ```

## Integration with Extension

The WASM module is automatically loaded by the browser extension through the following integration points:

1. **Manifest Declaration**: `wasm-loader.js` is included in content scripts
2. **Web Accessible Resources**: WASM files are accessible to the extension
3. **Content Script Integration**: `content.js` uses the WASM redactor
4. **Fallback Mechanism**: Graceful degradation to JavaScript if WASM fails

## API Reference

### WasmRedactor Class

#### Methods

- `initialize(wasmPath)` - Initialize the WASM module
- `redact(text)` - Redact sensitive information from text
- `restore(text)` - Restore redacted information
- `clearMappings()` - Clear all redaction mappings
- `getStats()` - Get redaction statistics
- `cleanup()` - Clean up resources

#### Example Usage

```javascript
const redactor = new WasmRedactor();
await redactor.initialize();

const text = "Call me at 555-123-4567";
const redacted = redactor.redact(text);      // "Call me at [PHONE_1]"
const restored = redactor.restore(redacted); // "Call me at 555-123-4567"

console.log(redactor.getStats());
// { total_redactions: 1, redaction_categories: ["PHONE"] }
```



## Customization

### Adding New Redaction Patterns

Edit `redactor.c` and add patterns to the `patterns` array:

```c
static RedactionPattern patterns[] = {
    {"EMAIL", "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", {0}, 0},
    {"PHONE", "\\b[0-9]{3}[-.]?[0-9]{3}[-.]?[0-9]{4}\\b", {0}, 0},
    {"CUSTOM", "your_regex_pattern_here", {0}, 0},
    // Add more patterns...
    {NULL, NULL, {0}, 0} // Keep this sentinel at the end
};
```

### Modifying Redaction Tokens

Customize the `generate_token` function:

```c
static void generate_token(const char* category, char* token_buf, size_t buf_size) {
    snprintf(token_buf, buf_size, "[%s_%d]", category, ++token_counter);
    // Customize the format here: change "[%s_%d]" to your preferred format
}
```

## Troubleshooting

### Common Issues

1. **Emscripten command not found**: Source the Emscripten environment
   ```bash
   source path/to/emsdk/emsdk_env.sh
   ```

2. **Build tools missing**: Install build essentials
   ```bash
   # macOS
   xcode-select --install
   
   # Ubuntu/Debian
   sudo apt install build-essential
   ```

3. **WASM module fails to load**: Check browser console for errors and verify web accessible resources in manifest.json

4. **Memory allocation errors**: Increase INITIAL_MEMORY in Makefile:
   ```makefile
   -s INITIAL_MEMORY=33554432  # 32MB instead of 16MB
   ```

### Debug Build

For debugging, use the debug flags in Makefile:

```makefile
# Add these flags for debugging
-g4 \
-s ASSERTIONS=2 \
-s SAFE_HEAP=1 \
--source-map-base http://localhost:8000/
```

### Performance Optimization

1. **Optimize for size**:
   ```makefile
   -Os \  # Optimize for size
   -s MODULARIZE=1 \
   -s SINGLE_FILE=1  # Single file output
   ```

2. **Optimize for speed**:
   ```makefile
   -O3 \  # Maximum optimization
   -s FAST_MATH=1
   ```

## Directory Structure After Build

```
extension/wasm/
├── redactor.c           # C source implementation
├── Makefile             # Build configuration
├── wasm-loader.js       # JavaScript interface
├── test.html            # Test page
├── README.md            # This file
└── build/              # Generated build files
    ├── redactor.js     # Emscripten JS glue
    └── redactor.wasm   # Compiled WASM binary
```

## Security Considerations

- The redaction mappings are stored in memory only
- Sensitive data is not persisted to disk
- Consider sandboxing for production deployments

## Performance Notes

- **C-to-WASM build**: ~100KB, fast execution, full regex support
- **JavaScript fallback**: <1KB, basic functionality only

## Contributing

1. Test your changes with `make test`
2. Ensure WASM build works
3. Update documentation for any API changes
4. Add test cases for new redaction patterns

## License

Same as the parent extension project. 