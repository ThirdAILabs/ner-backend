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
	SubDir  string
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
	return createInferenceTasks(c.iterObjects(), targetBytes)
}

func (c *LocalConnector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams LocalConnectorTaskParams
	if err := json.Unmarshal(params, &parsedParams); err != nil {
		return nil, fmt.Errorf("error unmarshalling params: %w", err)
	}

	return iterTaskChunks(ctx, parsedParams.ChunkKeys, c)
}

func (c *LocalConnector) iterObjects() ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		err := filepath.WalkDir(localStorageFullpath(c.params.BaseDir, c.params.SubDir), func(path string, d fs.DirEntry, err error) error {
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

			obj := Object{Name: filepath.Join(c.params.SubDir, d.Name()), Size: info.Size()}
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

func (c *LocalConnector) getObject(key string) ([]byte, error) {
	// TODO: This method loads entire files into memory which can cause issues with large files.
	// Change this method to return io.ReadCloser instead of []byte to stream data
	data, err := os.ReadFile(localStorageFullpath(c.params.BaseDir, key))
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s/%s/%s: %w", c.params.BaseDir, c.params.SubDir, key, err)
	}
	return data, nil
}

func (c *LocalConnector) GetObjectStream(ctx context.Context, key string) (io.Reader, error) {
	data, err := c.getObject(key)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(data), nil
}
