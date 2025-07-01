# Build Dependencies

## Tokenizers Library

This project requires the `libtokenizers.a` static library from the Hugging Face tokenizers library.

### Building the Library

1. **Clone the tokenizers repository** (same directory as ner-backend):
   ```bash
   cd ..
   git clone https://github.com/daulet/tokenizers.git
   cd tokenizers
   ```

2. **Checkout the correct version** (must match go.mod dependency):
   ```bash
   git checkout v1.20.2
   ```

3. **Build the static library**:
   ```bash
   make build
   ```

4. **Copy to ner-backend**:
   ```bash
   cp libtokenizers.a ../ner-backend/
   ```

### For GitHub Actions

Add this step to your workflow:

```yaml
- name: Build tokenizers library
  run: |
    cd ..
    git clone https://github.com/daulet/tokenizers.git
    cd tokenizers
    git checkout v1.20.2
    make build
    cp libtokenizers.a ../ner-backend/
    cd ../ner-backend
```

### Requirements

- Rust toolchain (for building tokenizers)
- Make
- C++ compiler

The built library (`libtokenizers.a`) should be approximately 28MB and is required for CGO linking in the core module. 