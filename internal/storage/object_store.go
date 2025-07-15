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
	PutObject(ctx context.Context, key string, data io.Reader) error

	DeleteObjects(ctx context.Context, dir string) error

	DownloadDir(ctx context.Context, src, dest string, overwrite bool) error

	UploadDir(ctx context.Context, src, dest string) error

	GetUploadConnector(ctx context.Context, dir string, uploadParams UploadParams) (Connector, error)
}
