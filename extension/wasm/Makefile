# Makefile for building Python redaction module to WebAssembly
# Requires Emscripten and Python development headers

# Configuration
PYTHON_VERSION ?= 3.11
EMSCRIPTEN ?= emcc
OUTPUT_DIR = build
WASM_OUTPUT = $(OUTPUT_DIR)/redactor.js
WASM_FILE = $(OUTPUT_DIR)/redactor.wasm

# Emscripten flags (no Python needed for C-only implementation)

# Emscripten compilation flags
EMFLAGS = -s WASM=1 \
          -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap", "allocate", "intArrayFromString", "ALLOC_NORMAL", "UTF8ToString", "stringToUTF8", "lengthBytesUTF8"]' \
          -s EXPORTED_FUNCTIONS='["_redact", "_restore", "_clear_redaction_mappings", "_get_redaction_stats", "_free_string", "_init_redactor", "_cleanup_redactor", "_malloc", "_free"]' \
          -s MODULARIZE=1 \
		  -s EXPORT_ES6=1 \
          -s EXPORT_NAME='RedactorModule' \
          -s ENVIRONMENT=web \
          -s SINGLE_FILE=0 \
          -s USE_PTHREADS=0 \
          -s ALLOW_MEMORY_GROWTH=1 \
          -s INITIAL_MEMORY=16777216 \
          -s MAXIMUM_MEMORY=67108864 \
          -s STACK_SIZE=1048576 \
          -s FORCE_FILESYSTEM=1 \
          -s ASYNCIFY=1 \
          -O3

# Source files
SOURCES = redactor.c

.PHONY: all clean setup test install-deps

all: setup $(WASM_OUTPUT)

# Create output directory
setup:
	@mkdir -p $(OUTPUT_DIR)

# Build the WebAssembly module
$(WASM_OUTPUT): $(SOURCES)
	@echo "Building WebAssembly module..."
	$(EMSCRIPTEN) $(SOURCES) \
		$(EMFLAGS) \
		-o $(WASM_OUTPUT)
	@echo "WebAssembly module built successfully!"
	@echo "Output files:"
	@echo "  - $(WASM_OUTPUT)"
	@echo "  - $(WASM_FILE)"



# Test the C module locally (compile and run native version)
test:
	@echo "Testing C redactor module..."
	@gcc -std=c99 -Wall test_redactor.c -o test_redactor
	@echo "Running tests..."
	@./test_redactor
	@echo "Cleaning up test executable..."
	@rm -f test_redactor

# Test compilation only (useful for CI/CD)
test-compile:
	@echo "Testing compilation only..."
	@gcc -std=c99 -Wall test_redactor.c -o test_redactor
	@echo "Compilation successful!"
	@rm -f test_redactor

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OUTPUT_DIR)
	@rm -f *.o *.so *.pyc
	@rm -rf __pycache__

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "Installing dependencies for Ubuntu/Debian..."
	sudo apt update
	sudo apt install -y build-essential nodejs npm
	npm install -g emscripten

# Install dependencies (macOS)
install-deps-macos:
	@echo "Installing dependencies for macOS..."
	brew install node emscripten

# Install dependencies (general)
install-deps:
	@echo "Installing general dependencies..."
	@echo "Please install Emscripten for your platform"

# Development server for testing
serve:
	@echo "Starting development server..."
	@cd $(OUTPUT_DIR) && python3 -m http.server 8000

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build the WebAssembly module"
	@echo "  test             - Test the C module locally (compile and run)"
	@echo "  test-compile     - Test compilation only (no execution)"
	@echo "  clean            - Clean build artifacts"
	@echo "  install-deps     - Install general dependencies"
	@echo "  install-deps-ubuntu - Install dependencies on Ubuntu/Debian"
	@echo "  install-deps-macos  - Install dependencies on macOS"
	@echo "  serve            - Start development server"
	@echo "  help             - Show this help message" 