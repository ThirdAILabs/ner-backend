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

type ConnectorConfigs struct {
	Local LocalConnectorConfig
	S3    S3ConnectorConfig
}

func NewConnector(ctx context.Context, connectorType connectorType, defaultConfigs ConnectorConfigs, params []byte) (Connector, error) {
	// Config is for app-wide configurations such as root directories or credentials while params are for task specific identifiers.
	// Unlike config, params is stored in the reports table of the database.
	// 
	switch connectorType {
	case LocalConnectorType:
		var localConnectorParams LocalConnectorParams
		if err := json.Unmarshal(params, &localConnectorParams); err != nil {
			return nil, err
		}
		return NewLocalConnector(defaultConfigs.Local, localConnectorParams), nil
		
	case S3ConnectorType:
		var s3ConnectorParams S3ConnectorParams
		if err := json.Unmarshal(params, &s3ConnectorParams); err != nil {
			return nil, err
		}
		return NewS3Connector(ctx, defaultConfigs.S3, s3ConnectorParams)
		
	default:
		return nil, fmt.Errorf("unknown connector type: %s", connectorType)
	}
}