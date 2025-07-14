/**
 * MarkItDown WASM Loader
 * Loads and provides interface to the Python-compiled MarkItDown WASM module
 * Note: py2wasm modules are standalone executables, not function libraries
 */

class MarkItDownWasm {
  constructor() {
    this.wasmModule = null;
    this.initialized = false;
    this.loading = false;
    this.memory = null;
    this.wasmImports = null;
    this.wasmExports = null;
    this.env = {};
    this.args = [];
    this.stdout = '';
    this.stderr = '';
  }

  /**
   * Initialize the WASM module
   * @param {string} wasmPath - Path to the WASM module file
   * @returns {Promise<boolean>} - Success status
   */
  async initialize(wasmPath) {
    if (this.initialized) {
      return true;
    }

    if (this.loading) {
      // Wait for current loading to complete
      while (this.loading) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return this.initialized;
    }

    this.loading = true;

    try {
      // Load the WASM file
      const wasmUrl = typeof chrome !== 'undefined' && chrome.runtime ? 
        chrome.runtime.getURL(wasmPath) : wasmPath;
      const wasmArrayBuffer = await fetch(wasmUrl).then(r => r.arrayBuffer());
      
      // Create WASI imports for py2wasm modules
      this.wasmImports = {
        wasi_snapshot_preview1: this.createWasiImports()
      };

      // Compile and instantiate the WASM module
      this.wasmModule = await WebAssembly.instantiate(wasmArrayBuffer, this.wasmImports);
      this.wasmExports = this.wasmModule.instance.exports;
      this.memory = this.wasmExports.memory;

      this.initialized = true;
      this.loading = false;
      console.log('MarkItDown WASM module initialized successfully');
      return true;

    } catch (error) {
      console.error('Failed to initialize MarkItDown WASM module:', error);
      this.loading = false;
      return false;
    }
  }

  /**
   * Create WASI imports for py2wasm modules
   * @returns {Object} - WASI import functions
   */
  createWasiImports() {
    return {
      proc_exit: (code) => {
        console.log(`WASM process exited with code: ${code}`);
        if (code !== 0) {
          throw new Error(`WASM process exited with error code: ${code}`);
        }
      },
      
      fd_write: (fd, iovs, iovs_len, nwritten) => {
        try {
          const memoryView = new DataView(this.memory.buffer);
          let written = 0;
          
          for (let i = 0; i < iovs_len; i++) {
            const iov = iovs + i * 8;
            const ptr = memoryView.getUint32(iov, true);
            const len = memoryView.getUint32(iov + 4, true);
            
            const buffer = new Uint8Array(this.memory.buffer, ptr, len);
            const text = new TextDecoder().decode(buffer);
            
            if (fd === 1) { // stdout
              this.stdout += text;
            } else if (fd === 2) { // stderr
              this.stderr += text;
            }
            
            written += len;
          }
          
          // Write the number of bytes written
          if (nwritten !== 0) {
            memoryView.setUint32(nwritten, written, true);
          }
          
          return 0; // Success
        } catch (error) {
          console.error('fd_write error:', error);
          return 28; // EINVAL
        }
      },
      
      fd_read: () => 28, // EINVAL - no stdin support
      fd_close: () => 0,
      fd_seek: () => 28, // EINVAL
      fd_fdstat_get: () => 28, // EINVAL
      
      environ_get: (environ, environ_buf) => {
        // Simple environment setup
        return 0;
      },
      
      environ_sizes_get: (environ_count, environ_buf_size) => {
        const memoryView = new DataView(this.memory.buffer);
        memoryView.setUint32(environ_count, 0, true);
        memoryView.setUint32(environ_buf_size, 0, true);
        return 0;
      },
      
      args_get: (argv, argv_buf) => {
        // Set up command line arguments
        const memoryView = new DataView(this.memory.buffer);
        let bufPtr = argv_buf;
        
        for (let i = 0; i < this.args.length; i++) {
          const arg = this.args[i];
          const argBytes = new TextEncoder().encode(arg + '\0');
          
          // Write argument pointer
          memoryView.setUint32(argv + i * 4, bufPtr, true);
          
          // Write argument data
          new Uint8Array(this.memory.buffer, bufPtr, argBytes.length).set(argBytes);
          bufPtr += argBytes.length;
        }
        
        return 0;
      },
      
      args_sizes_get: (argc, argv_buf_size) => {
        const memoryView = new DataView(this.memory.buffer);
        memoryView.setUint32(argc, this.args.length, true);
        
        const totalSize = this.args.reduce((sum, arg) => sum + arg.length + 1, 0);
        memoryView.setUint32(argv_buf_size, totalSize, true);
        
        return 0;
      },
      
      random_get: (buf, buf_len) => {
        const bytes = new Uint8Array(this.memory.buffer, buf, buf_len);
        if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
          crypto.getRandomValues(bytes);
        } else {
          // Fallback for environments without crypto
          for (let i = 0; i < buf_len; i++) {
            bytes[i] = Math.floor(Math.random() * 256);
          }
        }
        return 0;
      },
      
      clock_time_get: (clock_id, precision, time) => {
        const now = BigInt(Date.now()) * BigInt(1000000);
        const memoryView = new DataView(this.memory.buffer);
        memoryView.setBigUint64(time, now, true);
        return 0;
      },
      
      clock_res_get: (clock_id, resolution) => {
        const memoryView = new DataView(this.memory.buffer);
        memoryView.setBigUint64(resolution, BigInt(1000000), true);
        return 0;
      },
      
      // File system operations (minimal stubs)
      path_open: () => 28,
      path_create_directory: () => 28,
      path_remove_directory: () => 28,
      path_unlink_file: () => 28,
      fd_prestat_get: () => 28,
      fd_prestat_dir_name: () => 28,
      path_filestat_get: () => 28,
      fd_filestat_get: () => 28,
      fd_readdir: () => 28,
    };
  }

  /**
   * Parse file contents using the MarkItDown WASM module
   * Note: This is a simplified approach - py2wasm modules are designed as executables
   * @param {string} fileName - Name of the file (for type detection)
   * @param {string} fileContents - Contents of the file to parse
   * @returns {Promise<string>} - Parsed markdown content
   */
  async parse(fileName, fileContents) {
    if (!this.initialized) {
      throw new Error('MarkItDown WASM module not initialized');
    }

    try {
      // Reset output buffers
      this.stdout = '';
      this.stderr = '';
      
      // Set up arguments as if calling from command line
      // For py2wasm modules, we need to simulate command line execution
      this.args = ['markitdown_wasm.py', fileName, fileContents];
      
      // Call the main function (_start)
      if (this.wasmExports._start) {
        try {
          this.wasmExports._start();
        } catch (error) {
          // _start might throw on exit, which is normal
          if (error.message && error.message.includes('exit')) {
            // This is expected for py2wasm modules
          } else {
            throw error;
          }
        }
      }
      
      // Return the stdout as the result
      return this.stdout || 'No output generated';
      
    } catch (error) {
      console.error('Failed to parse file with MarkItDown WASM:', error);
      
      // If there's stderr output, include it in the error
      if (this.stderr) {
        throw new Error(`MarkItDown parsing failed: ${this.stderr}`);
      }
      
      throw new Error(`MarkItDown parsing failed: ${error.message}`);
    }
  }

  /**
   * Simple fallback parsing using a basic implementation
   * This is a backup in case WASM fails
   * @param {string} fileName - Name of the file
   * @param {string} fileContents - Contents of the file
   * @returns {string} - Basic markdown conversion
   */
  fallbackParse(fileName, fileContents) {
    // Basic text-to-markdown conversion
    const lines = fileContents.split('\n');
    let markdown = '';
    
    for (const line of lines) {
      if (line.trim() === '') {
        markdown += '\n';
        continue;
      }
      
      // Basic markdown conversion rules
      if (line.startsWith('#')) {
        markdown += line + '\n'; // Already markdown header
      } else if (line.startsWith('*') || line.startsWith('-')) {
        markdown += line + '\n'; // Already markdown list
      } else if (line.startsWith('```')) {
        markdown += line + '\n'; // Already markdown code block
      } else if (line.match(/^\d+\./)) {
        markdown += line + '\n'; // Numbered list
      } else {
        // Regular text - check if it looks like a heading
        if (line.length < 60 && !line.includes('.') && line.trim() === line.trim().toUpperCase()) {
          markdown += `# ${line}\n`;
        } else {
          markdown += line + '\n';
        }
      }
    }
    
    return markdown;
  }

  /**
   * Check if the module is initialized
   * @returns {boolean} - Initialization status
   */
  isInitialized() {
    return this.initialized;
  }

  /**
   * Get the last stdout output
   * @returns {string} - Last stdout output
   */
  getStdout() {
    return this.stdout;
  }

  /**
   * Get the last stderr output
   * @returns {string} - Last stderr output
   */
  getStderr() {
    return this.stderr;
  }

  /**
   * Cleanup resources
   */
  cleanup() {
    this.wasmModule = null;
    this.initialized = false;
    this.loading = false;
    this.memory = null;
    this.wasmImports = null;
    this.wasmExports = null;
    this.env = {};
    this.args = [];
    this.stdout = '';
    this.stderr = '';
  }
}

/**
 * Factory function to create and initialize a MarkItDown WASM module
 * @param {string} wasmPath - Path to the WASM module file (e.g., 'wasm/markitdown_wasm.wasm')
 * @returns {Promise<MarkItDownWasm>} - Initialized MarkItDown WASM module
 */
async function newMarkItDownWasm(wasmPath) {
  const markitdownWasm = new MarkItDownWasm();
  const success = await markitdownWasm.initialize(wasmPath);
  
  if (!success) {
    throw new Error('Failed to initialize MarkItDown WASM module');
  }
  
  return markitdownWasm;
}

// Export for use in browser extension
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MarkItDownWasm, newMarkItDownWasm };
} else if (typeof window !== 'undefined') {
  window.MarkItDownWasm = MarkItDownWasm;
  window.newMarkItDownWasm = newMarkItDownWasm;
}

// Default export for ES6 modules
export default MarkItDownWasm;
export { newMarkItDownWasm }; 