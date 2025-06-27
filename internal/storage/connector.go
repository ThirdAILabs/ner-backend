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

type connectorType string
const (
	LocalConnectorType connectorType = "local"
	S3ConnectorType connectorType = "s3"
)

func ToConnectorType(typeString string) (connectorType, error) {
	switch typeString {
	case string(LocalConnectorType):
		return LocalConnectorType, nil
	case string(S3ConnectorType):
		return S3ConnectorType, nil
	}
	return "", fmt.Errorf("unknown connector type: %s", typeString)
}

func NewConnector(ctx context.Context, connectorType connectorType, params []byte) (Connector, error) {
	switch connectorType {
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
		return NewS3Connector(ctx, s3ConnectorParams)
		
	default:
		return nil, fmt.Errorf("unknown connector type: %s", connectorType)
	}
}