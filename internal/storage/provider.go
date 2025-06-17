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

	GetObjectStream(bucket, key string) (io.Reader, error)

	PutObject(ctx context.Context, bucket, key string, data io.Reader) error

	DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error

	UploadDir(ctx context.Context, bucket, prefix, src string) error

	ListObjects(ctx context.Context, bucket, prefix string) ([]Object, error)

	IterObjects(ctx context.Context, bucket, prefix string) ObjectIterator

	DeleteObjects(ctx context.Context, bucket string, prefix string) error

	ValidateAccess(ctx context.Context, bucket, prefix string) error
}
