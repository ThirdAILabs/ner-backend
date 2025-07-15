package storage

import "path/filepath"

func localStorageFullpath(baseDir, key string) string {
	return filepath.Join(baseDir, key)
}
