package storage

import "path/filepath"

func localStorageFullpath(baseDir, bucket, key string) string {
	return filepath.Join(baseDir, bucket, key)
}