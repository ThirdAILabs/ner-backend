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

var _ Connector = &LocalConnector{}

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

func (c *LocalConnector) ValidateParams(ctx context.Context) error {
	return nil
}

func (c *LocalConnector) CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error) {
	var tasks []InferenceTask

	var currentChunkKeys []string
	var currentChunkSize int64 = 0
	var totalObjects int = 0

	addTask := func(chunkKeys []string, chunkSize int64) error {
		taskParams := LocalConnectorTaskParams{
			ChunkKeys: chunkKeys,
		}
		
		taskParamsBytes, err := json.Marshal(taskParams)
		if err != nil {
			return fmt.Errorf("error marshalling task params: %w", err)
		}

		tasks = append(tasks, InferenceTask{
			Params: taskParamsBytes,
			TotalSize: chunkSize,
		})

		return nil
	}

	for obj, err := range c.iterObjects(c.params.Bucket, c.params.Prefix) {
		if err != nil {
			return nil, 0, fmt.Errorf("error iterating over local objects: %w", err)
		}

		if currentChunkSize+obj.Size > targetBytes && len(currentChunkKeys) > 0 {
			if err := addTask(currentChunkKeys, currentChunkSize); err != nil {
				return nil, 0, err
			}
			
			currentChunkKeys = []string{}
			currentChunkSize = 0
		}
		
		currentChunkKeys = append(currentChunkKeys, obj.Name)
		currentChunkSize += obj.Size
		totalObjects++
	}

	if len(currentChunkKeys) > 0 {
		if err := addTask(currentChunkKeys, currentChunkSize); err != nil {
			return nil, 0, err
		}
	}

	return tasks, int64(totalObjects), nil
}

func (c *LocalConnector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams LocalConnectorTaskParams
	if err := json.Unmarshal(params, &parsedParams); err != nil {
		return nil, fmt.Errorf("error unmarshalling params: %w", err)
	}

	parser := NewDefaultParser()

	chunkStreams := make(chan ObjectChunkStream)

	go func() {
		defer close(chunkStreams)
		
		
		for _, objectKey := range parsedParams.ChunkKeys {
			objectStream, err := c.getObjectStream(c.params.Bucket, objectKey)
			if err != nil {
				chunkStreams <- ObjectChunkStream{Name: objectKey, Chunks: nil, Error: err}
				continue
			}

			parsedChunks := parser.Parse(objectKey, objectStream)
			
			chunkStreams <- ObjectChunkStream{
				Name: objectKey,
				Chunks: parsedChunks,
				Error: nil,
			}
		}
	}()

	return chunkStreams, nil
}

func (c *LocalConnector) fullpath(bucket, key string) string {
	return filepath.Join(c.params.BaseDir, bucket, key)
}

func (c *LocalConnector) iterObjects(bucket, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		err := filepath.WalkDir(c.fullpath(bucket, dir), func(path string, d fs.DirEntry, err error) error {
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
	data, err := os.ReadFile(c.fullpath(bucket, key))
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