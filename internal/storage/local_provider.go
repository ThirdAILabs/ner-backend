package storage

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// This is not intended to be used in production, it is just a simple implementation for testing.
type LocalProvider struct {
	dir string
}

func NewLocalProvider(dir string) *LocalProvider {
	return &LocalProvider{dir: dir}
}

func (p *LocalProvider) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func (p *LocalProvider) GetObject(ctx context.Context, bucket, key string) ([]byte, error) {
	return os.ReadFile(filepath.Join(p.dir, bucket, key))
}

func (p *LocalProvider) DownloadObject(ctx context.Context, bucket, key, filename string) error {
	path := filepath.Join(p.dir, bucket, key)
	// copy path to filename
	src, err := os.Open(path)
	if err != nil {
		return err
	}
	defer src.Close()

	if err := os.MkdirAll(filepath.Dir(filename), os.ModePerm); err != nil {
		return err
	}

	dst, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		return err
	}

	return nil
}

func (p *LocalProvider) GetObjectStream(bucket, key string) (io.Reader, error) {
	data, err := p.GetObject(context.Background(), bucket, key)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}

func (p *LocalProvider) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	path := filepath.Join(p.dir, bucket, key)
	if err := os.MkdirAll(filepath.Dir(path), os.ModePerm); err != nil {
		return err
	}

	dst, err := os.Create(path)
	if err != nil {
		return err
	}
	defer dst.Close()

	if _, err := io.Copy(dst, data); err != nil {
		return err
	}

	return nil
}

func (p *LocalProvider) ListObjects(ctx context.Context, bucket, prefix string) ([]Object, error) {

	files, err := os.ReadDir(filepath.Join(p.dir, bucket))
	if err != nil {
		return nil, err
	}

	var objects []Object
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		if prefix != "" && !strings.HasPrefix(file.Name(), prefix) {
			continue
		}

		info, err := file.Info()
		if err != nil {
			return nil, err
		}

		objects = append(objects, Object{Name: file.Name(), Size: info.Size()})
	}

	return objects, nil
}

func (p *LocalProvider) IterObjects(ctx context.Context, bucket, prefix string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		files, err := os.ReadDir(filepath.Join(p.dir, bucket))
		if err != nil {
			yield(Object{}, err)
			return
		}

		for _, file := range files {
			if file.IsDir() {
				continue
			}
			if prefix != "" && !strings.HasPrefix(file.Name(), prefix) {
				continue
			}

			info, err := file.Info()
			if err != nil {
				yield(Object{}, err)
				return
			}

			if !yield(Object{Name: file.Name(), Size: info.Size()}, nil) {
				return
			}
		}
	}
}
