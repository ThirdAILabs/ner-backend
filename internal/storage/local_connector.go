package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

type LocalConnectorParams struct {
	BaseDir string
	Bucket string
	Prefix string
}

type LocalConnectorTaskParams struct {
	ChunkKeys []string
}

type LocalConnector struct {
	params LocalConnectorParams
}

var _ Connector = (*LocalConnector)(nil)

func NewLocalConnector(params LocalConnectorParams) *LocalConnector {
	return &LocalConnector{params: params}
}

func (c *LocalConnector) Type() string {
	return LocalConnectorType
}

func (c *LocalConnector) GetParams() ([]byte, error) {
	cfgJson, err := json.Marshal(c.params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal local connector params: %w", err)
	}

	return cfgJson, nil
}

func (c *LocalConnector) CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error) {
	return createInferenceTasks(c.iterObjects(c.params.Bucket, c.params.Prefix), targetBytes)
}

func (c *LocalConnector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams LocalConnectorTaskParams
	if err := json.Unmarshal(params, &parsedParams); err != nil {
		return nil, fmt.Errorf("error unmarshalling params: %w", err)
	}

	return iterTaskChunks(c.params.Bucket, parsedParams.ChunkKeys, c.getObjectStream)
}

func (c *LocalConnector) iterObjects(bucket, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		err := filepath.WalkDir(localStorageFullpath(c.params.BaseDir, bucket, dir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}

			info, err := d.Info()
			if err != nil {
				return err
			}

			obj := Object{Name: filepath.Join(dir, d.Name()), Size: info.Size()}
			if !yield(obj, nil) {
				return io.EOF
			}
			return nil
		})

		if err != nil && !errors.Is(err, io.EOF) {
			yield(Object{}, err)
		}
	}
}


func (c *LocalConnector) getObject(bucket, key string) ([]byte, error) {
	data, err := os.ReadFile(localStorageFullpath(c.params.BaseDir, bucket, key))
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s/%s: %w", bucket, key, err)
	}
	return data, nil
}

func (c *LocalConnector) getObjectStream(bucket, key string) (io.Reader, error) {
	data, err := c.getObject(bucket, key)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}