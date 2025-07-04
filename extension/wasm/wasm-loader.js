/**
 * WASM Redactor Loader
 * Loads and provides interface to the Python-compiled WASM redaction module
 */

class WasmRedactor {
  constructor() {
    this.module = null;
    this.initialized = false;
    this.loading = false;
    
    // Function wrappers
    this.updateExtensionIdFunc = null;
    this.redactFunc = null;
    this.restoreFunc = null;
    this.clearFunc = null;
    this.statsFunc = null;
    this.freeFunc = null;
    this.initFunc = null;
    this.cleanupFunc = null;
  }

  /**
   * Initialize the WASM module
   * @param {string} wasmPath - Path to the WASM module files
   * @returns {Promise<boolean>} - Success status
   */
  async initialize(wasmPath = './build/') {
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
      const wasmDir = chrome.runtime.getURL('wasm/build/');
      const res = await fetch(wasmDir + 'redactor.js', { method: 'HEAD' });
      const importResult = (await import(`${wasmDir}redactor.js`));
      const createRedactorModule = importResult.default;
      this.module = await createRedactorModule({
        locateFile: (path) => wasmDir + path,
      });

      // Check if module is properly initialized
      if (!this.module.cwrap || typeof this.module.cwrap !== 'function') {
        throw new Error('Module missing cwrap function. Available functions: ' + Object.keys(this.module).join(', '));
      }

      // Wrap the exported functions
      this.updateExtensionIdFunc = this.module.cwrap('update_extension_id', null, ['string', 'string']);
      this.redactFunc = this.module.cwrap('redact', 'string', ['string', 'string']); // session_id, text
      this.restoreFunc = this.module.cwrap('restore', 'string', ['string', 'string']);
      this.clearFunc = this.module.cwrap('clear_redaction_mappings', null, []);
      this.statsFunc = this.module.cwrap('get_redaction_stats', 'string', []);
      this.freeFunc = this.module.cwrap('free_string', null, ['number']);
      this.initFunc = this.module.cwrap('init_redactor', 'number', []);
      this.cleanupFunc = this.module.cwrap('cleanup_redactor', null, []);

      // Initialize the Python redactor
      const initResult = this.initFunc();
      if (!initResult) {
        throw new Error('Failed to initialize Python redactor');
      }

      this.initialized = true;
      console.log('WASM redactor module initialized successfully');
      return true;

    } catch (error) {
      console.error('Failed to initialize WASM redactor:', error);
      this.initialized = false;
      return false;
    } finally {
      this.loading = false;
    }
  }

  updateExtensionId(oldSessionId, newSessionId) {
    if (!this.initialized || !this.updateExtensionIdFunc) {
      console.warn('WASM redactor not initialized, skipping update');
      return;
    }

    try {
      this.updateExtensionIdFunc(oldSessionId, newSessionId);
    } catch (error) {
      console.error('Error in WASM updateExtensionId:', error);
    }
  }
  
  /**
   * Redact sensitive information from text
   * @param {string} text - Text to redact
   * @param {string} sessionId - Session ID for the redaction
   * @returns {string} - Redacted text
   */
  redact(text, sessionId = 'default-session') {
    if (!this.initialized || !this.redactFunc) {
      console.warn('WASM redactor not initialized, returning original text');
      return text;
    }

    try {
      return this.redactFunc(sessionId, text);
    } catch (error) {
      return text;
    }
  }

  /**
   * Restore redacted information in text
   * @param {string} text - Text to restore
   * @returns {string} - Restored text
   */
  restore(text, sessionId = 'default-session') {
    if (!this.initialized || !this.restoreFunc) {
      console.warn('WASM redactor not initialized, returning original text');
      return text;
    }

    try {
      return this.restoreFunc(sessionId, text);
    } catch (error) {
      console.error('Error in WASM restore:', error);
      return text;
    }
  }

  /**
   * Clear all redaction mappings
   */
  clearMappings() {
    if (!this.initialized || !this.clearFunc) {
      console.warn('WASM redactor not initialized');
      return;
    }

    try {
      this.clearFunc();
    } catch (error) {
      console.error('Error clearing redaction mappings:', error);
    }
  }

  /**
   * Get redaction statistics
   * @returns {Object} - Statistics object
   */
  getStats() {
    if (!this.initialized || !this.statsFunc) {
      console.warn('WASM redactor not initialized');
      return {};
    }

    try {
      const statsJson = this.statsFunc();
      return JSON.parse(statsJson);
    } catch (error) {
      console.error('Error getting redaction stats:', error);
      return {};
    }
  }

  /**
   * Cleanup and free resources
   */
  cleanup() {
    if (this.initialized && this.cleanupFunc) {
      try {
        this.cleanupFunc();
      } catch (error) {
        console.error('Error during cleanup:', error);
      }
    }
    
    this.module = null;
    this.initialized = false;
    this.redactFunc = null;
    this.restoreFunc = null;
    this.clearFunc = null;
    this.statsFunc = null;
    this.freeFunc = null;
    this.initFunc = null;
    this.cleanupFunc = null;
  }
}



// Export classes for use in extension
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WasmRedactor };
} else {
  window.WasmRedactor = WasmRedactor;
} 