# PocketShield

This is an Electron application for Named Entity Recognition that migrates the functionality from the Next.js frontend app.

## Setup and Migration

### Install Dependencies

First, install the necessary dependencies:

```bash
npm install
```

### Run Migration Script

Run the migration script to copy components and assets from the Next.js app:

```bash
./migrate.sh
```

### Development Mode

To run the app in development mode:

```bash
npm run dev
```

This will start both the Vite development server and the Electron app that connects to it.

### Building for Production

To build the app for production:

```bash
MODEL_DIR=/path/to/models MODEL_TYPE=python_cnn npm run build
```

Ensure that you have a clean python 3.11 environment activated when running the build command. 

/path/to/models should have a directory called python_cnn with cnn_model.pth and qwen_tokenizer inside, or it should have a directory called bolt_udt with model.bin inside.

To download the qwen_tokenizer folder, run the following:
```bash
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); tokenizer.save_pretrained('qwen_tokenizer')"
```

### Windows Build
1. Install Git Bash (https://git-scm.com/downloads). Use Git Bash Terminal until step 5
2. Clone this repo (ner-backend) in Git Bash in your Documents folder (or any folder you want)
3. Install NVM (https://github.com/coreybutler/nvm-windows) in Git Bash
4. Install Node v22 using NVM in Git Bash
5. Run npm i and npm run build in the frontend/ dir
6. Install https://www.msys2.org/. This will install a UCRT64 terminal environment. Use UCRT64 Terminal until step 16
7. Open the UCRT64 terminal and install mingw-w64 GCC (https://www.msys2.org/)
8. Clone the tokenizers repo (https://github.com/daulet/tokenizers) in the same directory as you cloned ner-backend. To navigate there, you will run `cd /c/Users/<username>/Documents`
9. cd into the tokenizers repo and checkout branch v1.20.2 (same version as in our go dependencies) and run `make build` (you might run into some errors here, I forgot what I did to fix them but u got this)
10. A file called libtokenizers.a will be created, copy this to the ner-backend directory
11. Download the onnxruntime dll (https://github.com/microsoft/onnxruntime/releases) for windows x64. Open the zip file, go to lib/ and copy the onnxruntime.dll to ner-backend
12. cd into /ucrt64/bin and copy the following three files into ner-backend: libstdc++-6.dll, libwinpthread-1.dll, libgcc_s_seh-1.dll
13. In ner-backend, run `touch empty.c && gcc -c empty.c` and `ar rcs libdl.a empty.o`
14. Go to https://go.dev/dl/ and download the windows msi for go 1.23.10. Run through the installation.
15. Run `export PATH="/c/Program Files/Go/bin:$PATH"` in the terminal
16. Run `GOOS=windows GOARCH=amd64 CGO_LDFLAGS="-L. -lws2_32 -lbcrypt -luserenv -lntdll" go build -o main ./cmd/local/main.go` in ner-backend
17. Open the Git Bash terminal, navigate to ner-backend and download the onnx model checkpoint to models/
18. Run MODEL_DIR=/c/Users/Administrator/Documents/ner-backend/models MODEL_TYPE=onnx_cnn npm run build

in theory, these steps should build a .exe! probably won't though, good luck and have fun. enjoy the journey.


## Migration Notes

The migration process involves:

1. Setting up a Vite-based React development environment within the Electron app
2. Copying UI components, styles, and assets from the Next.js app
3. Adapting the routing system to work with Electron
4. Ensuring API calls work correctly from the Electron environment

### API Endpoints

API endpoints should be configured to point to the same backend as the Next.js app was using. Update the API URLs in the appropriate configuration files if needed.

### CSS and Styling

The migration preserves the Tailwind CSS styling from the original Next.js app.

### Component Adaptation

Some Next.js specific components may need adjustment to work in a standard React + Vite environment:

- Next.js `Link` components should be replaced with standard navigation
- Next.js specific routing needs to be adapted
- Server components must be converted to client components

## Troubleshooting

If you encounter issues:

1. Check the developer console for errors (View > Toggle Developer Tools)
2. Ensure all dependencies are installed
3. Verify that API endpoints are correctly configured 

## Running with Integrated Backend

This Electron app now has the Go backend integrated. Here's how to run it:

### Development Mode

1. Make sure you have the Go backend built:
   ```
   cd cmd/local/
   go clean -cache
   go build -o main
   cp main ../../
   cd ../../electron-ner-app
   ```

2. Start the integrated app:
   ```
   MODEL_DIR=/share/pratik/ MODEL_TYPE=onnx_cnn npm run dev
   ```
   This will:
   - Copy the Go backend to the `bin` directory
   - Start the Vite development server
   - Start Electron
   - Start the Go backend

### Production Build (macOS)

To build a macOS DMG with integrated backend:

```
MODEL_DIR=/share/pratik/ MODEL_TYPE=onnx_cnn npm run build
```

This will:
1. Build the Go backend for macOS
2. Copy it to the Electron app
3. Build the React frontend
4. Package everything into a DMG file with the backend included

The resulting DMG file will be in the `dist` directory. 

When installed, the app will automatically find and use the backend without any additional steps required.

### Troubleshooting

If you encounter issues with the backend not starting:

1. Check the app logs via the developer tools console (View > Toggle Developer Tools)
2. Ensure there are no firewall issues blocking port 8000
3. Make sure no other processes are using port 8000 