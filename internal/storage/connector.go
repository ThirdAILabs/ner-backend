package storage

import "context"

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
	ValidateParams(ctx context.Context, params []byte) error

	CreateInferenceTasks(ctx context.Context, params []byte) ([]InferenceTask, error)

	IterTaskChunks(ctx context.Context, params []byte) <-chan ObjectChunkStream
}

func NewConnector(name string, params []byte) *Connector { return nil}

const LocalConnectorType = "local"
const S3ConnectorType = "s3"

type LocalConnectorParams struct {
	Bucket string
	UploadId string
}

type S3ConnectorParams struct {
	Endpoint string
	Region string
	Bucket string
	Prefix string
}

type LocalConnectorTaskParams struct {
	ChunkKeys []string
}

type S3ConnectorTaskParams struct {
	ChunkKeys []string
}