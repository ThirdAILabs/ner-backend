import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const NO_GROUP = 'Ungrouped';

export const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(2)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`;
  }
  return num.toString();
};

export const formatFileSize = (bytes: number, space: boolean = false): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = [' Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + (space ? ' ' : '') + sizes[i];
};

export const getFilesFromElectron = async (
  supportedTypes: string[],
  isDirectoryMode: boolean = false
): Promise<{ allFilesMeta: any[]; totalSize: number; error?: string }> => {
  // @ts-ignore
  const results = await window.electron.openFileChooser(
    // Electron API does not expect '.' in the file extension
    supportedTypes.map((t) => t.replace('.', '')),
    isDirectoryMode
  );
  if (results.error) {
    return { allFilesMeta: [], totalSize: 0, error: results.error };
  }
  return { allFilesMeta: results.allFilesMeta, totalSize: results.totalSize };
};

export const uniqueFileNames = (fileNames: string[]): string[] => {
  const existingFileNameCount: Record<string, number> = {};
  const newFileNames = fileNames.map((fileName) => {
    let newFileName = fileName;
    if (existingFileNameCount[fileName] === undefined) {
      existingFileNameCount[fileName] = 0;
    } else {
      const ext_idx = fileName.lastIndexOf('.');
      newFileName =
        fileName.substring(0, ext_idx) +
        ` (${existingFileNameCount[fileName]})` +
        fileName.substring(ext_idx);
      // This is to handle an edge case like [a/file.txt, b/file.txt, b/file (1).txt]
      // a/file.txt -> file.txt | b/file.txt -> file (1).txt | b/file (1).txt -> b/file (1) (1).txt
      existingFileNameCount[newFileName] = 1;
    }
    existingFileNameCount[newFileName]++;
    return newFileName;
  });
  return newFileNames;
};
