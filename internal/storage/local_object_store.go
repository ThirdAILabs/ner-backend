package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type LocalObjectStore struct {
	baseDir string
}

var _ ObjectStore = (*LocalObjectStore)(nil)

func NewLocalObjectStore(dir string) (*LocalObjectStore, error) {
	baseDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path for %s: %w", dir, err)
	}

	return &LocalObjectStore{baseDir: baseDir}, nil
}

func (s *LocalObjectStore) PutObject(ctx context.Context, key string, data io.Reader) error {
	path := localStorageFullpath(s.baseDir, key)
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory for %s/%s: %w", s.baseDir, key, err)
	}

	dst, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file %s/%s: %w", s.baseDir, key, err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, data); err != nil {
		return fmt.Errorf("failed to write file %s/%s: %w", s.baseDir, key, err)
	}

	return nil
}

func (s *LocalObjectStore) DeleteObjects(ctx context.Context, dir string) error {
	fullPath := localStorageFullpath(s.baseDir, dir)
	if err := os.RemoveAll(fullPath); err != nil {
		return fmt.Errorf("failed to delete objects in %s/%s: %w", s.baseDir, dir, err)
	}
	return nil
}

func (s *LocalObjectStore) DownloadDir(ctx context.Context, src, dest string, overwrite bool) error {
	sourcePath := localStorageFullpath(s.baseDir, src)

	if _, err := os.Stat(dest); err == nil {
		if !overwrite {
			return fmt.Errorf("destination %s already exists and overwrite is false", dest)
		}
		if err := os.RemoveAll(dest); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	if err := os.MkdirAll(filepath.Dir(dest), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create parent directory for destination: %w", err)
	}

	if err := os.Symlink(sourcePath, dest); err != nil {
		return fmt.Errorf("failed to create symlink from %s/%s to %s: %w", s.baseDir, src, dest, err)
	}
	return nil
}

func (s *LocalObjectStore) UploadDir(ctx context.Context, src, dest string) error {
	destPath := localStorageFullpath(s.baseDir, dest)

	if _, err := os.Stat(destPath); err == nil {
		if err := os.RemoveAll(destPath); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	if err := os.MkdirAll(filepath.Dir(destPath), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create parent directory for %s/%s: %w", s.baseDir, dest, err)
	}

	if err := os.Symlink(src, destPath); err != nil {
		return fmt.Errorf("failed to create symlink from %s to %s/%s: %w", src, s.baseDir, dest, err)
	}
	return nil
}

func (s *LocalObjectStore) GetUploadConnector(ctx context.Context, uploadDir string, uploadParams UploadParams) (Connector, error) {
	return NewLocalConnector(
		LocalConnectorParams{
			BaseDir: s.baseDir,
			SubDir:  filepath.Join(uploadDir, uploadParams.UploadId.String()),
		},
	), nil
}
