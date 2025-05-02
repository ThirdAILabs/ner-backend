package storage

import (
	"context"
	"io"
)

type Object struct {
	Name string
	Size int64
}

type ObjectIterator func(yield func(obj Object, err error) bool)

type Provider interface {
	CreateBucket(ctx context.Context, bucket string) error

	GetObject(ctx context.Context, bucket, key string) ([]byte, error)

	DownloadObject(ctx context.Context, bucket, key, filename string) error

	GetObjectStream(bucket, key string) (io.Reader, error)

	PutObject(ctx context.Context, bucket, key string, data io.Reader) error

	ListObjects(ctx context.Context, bucket, prefix string) ([]Object, error)

	IterObjects(ctx context.Context, bucket, prefix string) ObjectIterator
}
