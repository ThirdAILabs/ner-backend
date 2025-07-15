package storage

import (
	"context"
	"encoding/json"
	"fmt"
)

type Object struct {
	Name string
	Size int64
}

type InferenceTask struct {
	Params    []byte
	TotalSize int64
}

type Chunk struct {
	Text    string
	Offset  int
	RawSize int64
	Error   error
}

type ObjectChunkStream struct {
	Name   string
	Chunks <-chan Chunk
	Error  error
}

type Connector interface {
	CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error)

	IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error)
}

type storageType string

const (
	UploadType storageType = "upload"
	LocalType  storageType = "local"
	S3Type     storageType = "s3"
)

func ToStorageType(typeString string) (storageType, error) {
	switch typeString {
	case string(LocalType):
		return LocalType, nil
	case string(S3Type):
		return S3Type, nil
	}
	return "", fmt.Errorf("unknown connector type: %s", typeString)
}

func NewConnector(ctx context.Context, storageType storageType, params []byte) (Connector, error) {
	switch storageType {
	// Local connector is not included because it is only used for uploads and therefore can only be
	// instantiated via an object store.
	case S3Type:
		var s3ConnectorParams S3ConnectorParams
		if err := json.Unmarshal(params, &s3ConnectorParams); err != nil {
			return nil, err
		}
		return NewS3Connector(ctx, s3ConnectorParams)

	default:
		return nil, fmt.Errorf("unknown connector type: %s", storageType)
	}
}
