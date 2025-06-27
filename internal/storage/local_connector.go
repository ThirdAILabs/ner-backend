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

func (c *LocalConnector) CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error) {
	return createInferenceTasks(c.iterObjects(c.params.Bucket, c.params.Prefix), targetBytes)
}

func (c *LocalConnector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams LocalConnectorTaskParams
	if err := json.Unmarshal(params, &parsedParams); err != nil {
		return nil, fmt.Errorf("error unmarshalling params: %w", err)
	}

	return iterTaskChunks(ctx, c.params.Bucket, parsedParams.ChunkKeys, c)
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
	// TODO: This method loads entire files into memory which can cause issues with large files.
	// Change this method to return io.ReadCloser instead of []byte to stream data
	data, err := os.ReadFile(localStorageFullpath(c.params.BaseDir, bucket, key))
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s/%s: %w", bucket, key, err)
	}
	return data, nil
}

func (c *LocalConnector) GetObjectStream(ctx context.Context, bucket, key string) (io.Reader, error) {
	data, err := c.getObject(bucket, key)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}