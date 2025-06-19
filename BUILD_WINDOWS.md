# Building NER Backend for Windows

This document explains the Windows-specific build process and the platform differences.

## Platform Differences

### 1. BoltUdt Model Support
- **macOS/Linux**: Full support for BoltUdt model
- **Windows**: BoltUdt is disabled due to C library dependencies that are difficult to cross-compile
- The code automatically detects the platform and excludes BoltUdt on Windows

### 2. Build Requirements

#### Windows-Specific Requirements:
- MSYS2 with MinGW-w64 GCC
- Windows system libraries (ws2_32, bcrypt, userenv, ntdll)
- ONNX Runtime DLL for Windows
- MinGW runtime DLLs

#### Required DLLs for Windows:
- `libgcc_s_seh-1.dll`
- `libstdc++-6.dll`
- `libwinpthread-1.dll`
- `onnxruntime.dll`

### 3. Build Process

#### Automated Build (All Platforms):
```bash
./build.sh
```

#### Manual Windows Build:
```bash
# Set environment variables
export GOOS=windows
export GOARCH=amd64
export CGO_LDFLAGS="-L. -lws2_32 -lbcrypt -luserenv -lntdll"

# Create dummy libdl.a (Windows doesn't have dlopen)
touch empty.c
gcc -c empty.c -o empty.o
ar rcs libdl.a empty.o

# Build
go build -o main.exe ./cmd/local/main.go
```

## Why These Changes?

1. **BoltUdt Disabled on Windows**: The Bolt library uses C bindings that require platform-specific compilation. Cross-compiling these for Windows is complex and error-prone.

2. **Dummy libdl.a**: Windows doesn't have the POSIX `dlopen` functionality. The dummy library satisfies the linker without providing the functionality (which isn't needed on Windows).

3. **System Libraries**: Windows requires linking against specific system libraries for network operations, cryptography, and user environment access.

## Supported Models by Platform

| Model Type | macOS | Linux | Windows |
|------------|-------|-------|---------|
| ONNX CNN   | ✅    | ✅    | ✅      |
| Python CNN | ✅    | ✅    | ✅      |
| Python Transformer | ✅ | ✅ | ✅   |
| Presidio   | ✅    | ✅    | ✅      |
| Bolt UDT   | ✅    | ✅    | ❌      |

## Troubleshooting

### Missing DLLs Error
If you get errors about missing DLLs when running the executable:
1. Copy the required DLLs from your MinGW installation (usually in `/ucrt64/bin/`)
2. Place them in the same directory as `main.exe`

### Build Errors
If the build fails with linker errors:
1. Ensure you have MSYS2/MinGW-w64 installed
2. Run the build from the UCRT64 terminal
3. Check that all required DLLs are present

### Model Not Found
If BoltUdt model is not available on Windows:
- Use ONNX CNN model instead (`MODEL_TYPE=onnx_cnn`)
- This provides similar functionality with cross-platform support