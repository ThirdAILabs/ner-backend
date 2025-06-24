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

type ObjectIterator func(yield func(obj Object, err error) bool)

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
	Type() string

	GetParams() ([]byte, error)

	ValidateParams(ctx context.Context) error

	CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error)

	IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error)
}

const LocalConnectorType = "local"
const S3ConnectorType = "s3"

func NewConnector(name string, params []byte) (Connector, error) {
	switch name {
	case LocalConnectorType:
		var localConnectorParams LocalConnectorParams
		if err := json.Unmarshal(params, &localConnectorParams); err != nil {
			return nil, err
		}
		return NewLocalConnector(localConnectorParams), nil
		
	case S3ConnectorType:
		var s3ConnectorParams S3ConnectorParams
		if err := json.Unmarshal(params, &s3ConnectorParams); err != nil {
			return nil, err
		}
		return NewS3Connector(s3ConnectorParams)
		
	default:
		return nil, fmt.Errorf("unknown connector type: %s", name)
	}
}