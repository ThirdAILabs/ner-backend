// Platform detection utilities for frontend

export function getPlatform(): 'windows' | 'macos' | 'linux' | 'unknown' {
  if (typeof window === 'undefined') {
    return 'unknown';
  }

  const userAgent = window.navigator.userAgent.toLowerCase();
  const platform = window.navigator.platform?.toLowerCase() || '';

  if (platform.includes('win') || userAgent.includes('windows')) {
    return 'windows';
  } else if (platform.includes('mac') || userAgent.includes('mac')) {
    return 'macos';
  } else if (platform.includes('linux') || userAgent.includes('linux')) {
    return 'linux';
  }

  return 'unknown';
}

export function isWindows(): boolean {
  return getPlatform() === 'windows';
}

export function isMacOS(): boolean {
  return getPlatform() === 'macos';
}

export function isLinux(): boolean {
  return getPlatform() === 'linux';
}

export function isElectron(): boolean {
  return typeof window !== 'undefined' && !!(window as any).electron;
}

export function shouldShowWindowControls(): boolean {
  // Show window controls only on Windows in Electron
  return isElectron() && isWindows();
}
