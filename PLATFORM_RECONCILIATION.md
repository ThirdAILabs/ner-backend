# Platform Reconciliation Changes

This document summarizes the changes made to reconcile Windows and macOS versions into a single mergeable codebase.

## Key Changes Made

### 1. Platform Utilities
Created `electron-ner-app/scripts/platform-utils.js` with utility functions:
- `getPlatform()` - Returns the current platform
- `isWindows()`, `isMacOS()`, `isLinux()` - Platform detection helpers
- `getExecutableName()` - Returns platform-specific executable name (adds .exe for Windows)
- `getWindowsDependencies()` - Returns list of Windows-specific DLLs
- `getOnnxRuntimePath()` - Returns platform-specific ONNX runtime library path

Created `frontend/lib/platform.ts` for frontend platform detection:
- Platform detection utilities for the frontend
- `shouldShowWindowControls()` - Determines if Windows controls should be shown

### 2. Frontend Changes

#### Window Controls
- `WindowControls.tsx` now conditionally renders only on Windows
- Uses platform detection to determine visibility
- No changes needed for macOS users

#### File Selection Dialog
- **macOS**: Single "Local Files or Folders" button that allows selecting both files and folders in one dialog
- **Windows/Linux**: Separate "Local Files" and "Local Directory" buttons due to OS limitations
- Uses `navigator.platform` to detect macOS and render appropriate UI
- Combined mode parameter added to file chooser API

### 3. Electron Main Process

#### Scripts Updated
- `copy-backend.js` - Now uses platform utilities for executable names and Windows DLL copying
- `start-backend.js` - Uses platform utilities for executable paths and ONNX runtime library
- `after-pack.js` - Handles platform-specific packaging requirements

#### Build Scripts
- `build-dmg.js` - Fixed to use correct executable name for macOS
- `build-win.js` - Properly handles Windows dependencies
- `build-linux.js` - Updated for consistency

### 4. Package.json Configuration
- Moved platform-specific resources (dylib files) into mac-specific configuration
- Base extraResources now only includes cross-platform items
- Platform-specific resources are defined in their respective sections

## Remaining Windows-Specific Values

### Backend Configuration
The following still has a Windows default that may need adjustment:
- `cmd/local/main.go` - `EnablePython` defaults to `true` (was `false` in main branch)

## How to Merge

1. Ensure all platform utilities are imported correctly
2. Test on both Windows and macOS to verify:
   - Window controls appear only on Windows
   - Backend starts correctly on both platforms
   - Build scripts work for each platform
   - Package builds successfully include all required dependencies

## Usage Examples

### Building for Different Platforms

```bash
# On macOS, build for macOS
npm run build-dmg

# On Windows, build for Windows
npm run build-win

# Cross-platform builds still work
GOOS=windows GOARCH=amd64 go build -o main.exe  # Build Windows exe on Mac
GOOS=darwin GOARCH=amd64 go build -o main       # Build Mac binary on Windows
```

### Runtime Platform Detection

```javascript
// In Electron scripts
const { isWindows, getExecutableName } = require('./platform-utils.js');

if (isWindows()) {
  // Windows-specific code
}

const executable = getExecutableName('main'); // Returns 'main.exe' on Windows, 'main' elsewhere
```

```typescript
// In React components
import { shouldShowWindowControls } from '@/lib/platform';

if (shouldShowWindowControls()) {
  // Show Windows controls
}
```

## Testing Checklist

- [ ] Window controls appear on Windows but not macOS
- [ ] Backend starts successfully on both platforms
- [ ] File dialog shows single button on macOS, two buttons on Windows
- [ ] macOS file dialog allows selecting both files and folders
- [ ] Windows file dialog separates file and folder selection
- [ ] Build scripts complete without errors
- [ ] Packaged apps include all necessary dependencies
- [ ] ONNX runtime loads correctly on both platforms