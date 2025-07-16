package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

type ObjectIterator func(yield func(obj Object, err error) bool)

func createInferenceTasks(iterObjects ObjectIterator, targetBytes int64) ([]InferenceTask, int64, error) {
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
			Params:    taskParamsBytes,
			TotalSize: chunkSize,
		})

		return nil
	}

	for obj, err := range iterObjects {
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

type FsConnector interface {
	GetObjectStream(ctx context.Context, key string) (io.Reader, error)
}

func iterTaskChunks(ctx context.Context, keys []string, connector FsConnector) (<-chan ObjectChunkStream, error) {
	parser := NewDefaultParser()

	// Allocate a small buffer so we can preprocess multiple chunks in the background while the inference is running.
	chunkStreams := make(chan ObjectChunkStream, 10)

	go func() {
		defer close(chunkStreams)

		for _, objectKey := range keys {
			objectStream, err := connector.GetObjectStream(ctx, objectKey)
			if err != nil {
				chunkStreams <- ObjectChunkStream{Name: objectKey, Chunks: nil, Error: err}
				continue
			}

			parsedChunks, err := parser.Parse(objectKey, objectStream)
			if err != nil {
				chunkStreams <- ObjectChunkStream{Name: objectKey, Chunks: nil, Error: err}
				continue
			}

			chunkStreams <- ObjectChunkStream{
				Name:   objectKey,
				Chunks: parsedChunks,
				Error:  nil,
			}
		}
	}()

	return chunkStreams, nil
}
