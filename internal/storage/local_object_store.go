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

var _ ObjectStore = &LocalObjectStore{}

func NewLocalObjectStore(dir string) (*LocalObjectStore, error) {
	baseDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path for %s: %w", dir, err)
	}
	
	return &LocalObjectStore{baseDir: baseDir}, nil
}

func (s *LocalObjectStore) fullpath(bucket, key string) string {
	return filepath.Join(s.baseDir, bucket, key)
}

func (s *LocalObjectStore) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func (s *LocalObjectStore) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	path := s.fullpath(bucket, key)
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory for %s/%s: %w", bucket, key, err)
	}

	dst, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file %s/%s: %w", bucket, key, err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, data); err != nil {
		return fmt.Errorf("failed to write file %s/%s: %w", bucket, key, err)
	}

	return nil
}

func (s *LocalObjectStore) DeleteObjects(ctx context.Context, bucket string, dir string) error {
	fullPath := s.fullpath(bucket, dir)
	if err := os.RemoveAll(fullPath); err != nil {
		return fmt.Errorf("failed to delete objects in %s/%s: %w", bucket, dir, err)
	}
	return nil
}

func (s *LocalObjectStore) DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error {
	sourcePath := s.fullpath(bucket, prefix)

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
		return fmt.Errorf("failed to create symlink from %s/%s to %s: %w", bucket, prefix, dest, err)
	}
	return nil
}

func (s *LocalObjectStore) UploadDir(ctx context.Context, bucket, prefix, src string) error {
	destPath := s.fullpath(bucket, prefix)

	if _, err := os.Stat(destPath); err == nil {
		if err := os.RemoveAll(destPath); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	if err := os.MkdirAll(filepath.Dir(destPath), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create parent directory for %s/%s: %w", bucket, prefix, err)
	}

	if err := os.Symlink(src, destPath); err != nil {
		return fmt.Errorf("failed to create symlink from %s to %s/%s: %w", src, bucket, prefix, err)
	}
	return nil
}

func (s *LocalObjectStore) GetConnector(bucket, prefix string) (Connector, error) {
	return NewLocalConnector(LocalConnectorParams{
		BaseDir: s.baseDir,
		Bucket: bucket,
		Prefix: prefix,
	}), nil
}