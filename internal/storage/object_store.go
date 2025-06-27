package storage

import (
	"context"
	"io"

	"github.com/google/uuid"
)

type UploadParams struct {
	UploadId uuid.UUID
}

type ObjectStore interface {
	CreateBucket(ctx context.Context, bucket string) error

	PutObject(ctx context.Context, bucket, key string, data io.Reader) error

	DeleteObjects(ctx context.Context, bucket string, prefix string) error

	DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error

	UploadDir(ctx context.Context, bucket, prefix, src string) error

	GetUploadConnector(ctx context.Context, bucket string, uploadParams UploadParams) (Connector, error)
}
