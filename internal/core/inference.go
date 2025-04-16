package core

import (
	"context"
	"log/slog"
	"ner-backend/internal/s3"
)

const (
	maxInflightChunks = 20
)

func RunModelInference(parser Parser, model Model, s3client s3.Client, modelArtifactPath, bucket string, objects []string) (map[string][]Entity, []error) {
	chunks := make(chan ParsedChunk, maxInflightChunks)

	go func() {
		for _, object := range objects {
			stream := s3client.DownloadFileStream(bucket, object)
			parser.Parse(object, stream, chunks)
		}
		close(chunks)
	}()

	obj2Entity := make(map[string][]Entity)
	var errors []error

	for chunk := range chunks {
		if chunk.Error != nil {
			slog.Error("error parsing document", "error", chunk.Error)
			errors = append(errors, chunk.Error)
			continue
		}

		chunkEntities, err := model.Predict(context.TODO(), chunk.Text)
		if err != nil {
			slog.Error("error running model inference", "error", err)
			errors = append(errors, err)
			continue
		}

		objEntities := obj2Entity[chunk.Object]

		for _, entity := range chunkEntities {
			entity.Start += chunk.Offset
			entity.End += chunk.Offset
			objEntities = append(objEntities, entity)
		}

		obj2Entity[chunk.Object] = objEntities
	}

	return obj2Entity, errors
}
