package storage

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

type LocalProvider struct {
	baseDir string
}

func (p *LocalProvider) fullpath(bucket, key string) string {
	return filepath.Join(p.baseDir, bucket, key)
}

var _ Provider = &LocalProvider{}

func NewLocalProvider(dir string) (*LocalProvider, error) {
	baseDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path for %s: %w", dir, err)
	}

	return &LocalProvider{baseDir: baseDir}, nil
}

func (p *LocalProvider) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func (p *LocalProvider) GetObject(ctx context.Context, bucket, key string) ([]byte, error) {
	data, err := os.ReadFile(p.fullpath(bucket, key))
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s/%s: %w", bucket, key, err)
	}
	return data, nil
}

func (p *LocalProvider) GetObjectStream(bucket, key string) (io.Reader, error) {
	data, err := p.GetObject(context.Background(), bucket, key)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}

func (p *LocalProvider) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	path := p.fullpath(bucket, key)
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

func (p *LocalProvider) DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error {
	sourcePath := p.fullpath(bucket, prefix)

	if _, err := os.Stat(dest); err == nil {
		if !overwrite {
			return fmt.Errorf("destination %s already exists and overwrite is false", dest)
		}
		if err := os.RemoveAll(dest); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	return os.CopyFS(dest, os.DirFS(sourcePath))
}

func (p *LocalProvider) UploadDir(ctx context.Context, bucket, prefix, src string) error {
	destPath := p.fullpath(bucket, prefix)

	if _, err := os.Stat(destPath); err == nil {
		if err := os.RemoveAll(destPath); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	if err := os.CopyFS(destPath, os.DirFS(src)); err != nil {
		return fmt.Errorf("failed to copy directory from %s to %s/%s: %w", src, bucket, prefix, err)
	}
	return nil
}

func (p *LocalProvider) ListObjects(ctx context.Context, bucket, dir string) ([]Object, error) {
	files, err := os.ReadDir(p.fullpath(bucket, dir))
	if err != nil {
		return nil, fmt.Errorf("failed to list files in %s/%s: %w", bucket, dir, err)
	}

	var objects []Object
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		info, err := file.Info()
		if err != nil {
			return nil, fmt.Errorf("failed to get file info for %s/%s/%s: %w", bucket, dir, file.Name(), err)
		}

		objects = append(objects, Object{Name: filepath.Join(dir, file.Name()), Size: info.Size()})
	}

	return objects, nil
}

func (p *LocalProvider) IterObjects(ctx context.Context, bucket, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		err := filepath.WalkDir(p.fullpath(bucket, dir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}

			info, err := d.Info()
			if err != nil {
				return err
			}

			obj := Object{Name: filepath.Join(dir, d.Name()), Size: info.Size()}
			if !yield(obj, nil) {
				return io.EOF
			}
			return nil
		})

		if err != nil && !errors.Is(err, io.EOF) {
			yield(Object{}, err)
		}
	}
}

func (p *LocalProvider) ValidateAccess(ctx context.Context, bucket, prefix string) error {
	return nil
}
