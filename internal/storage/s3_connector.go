package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3ConnectorParams struct {
	Endpoint string
	Region string
	Bucket string
	Prefix string
	AccessKeyID string
	SecretAccessKey string
}

type S3ConnectorTaskParams struct {
	ChunkKeys []string
}

type S3Connector struct {
	client     *s3.Client
	params S3ConnectorParams
}

func NewS3Connector(ctx context.Context, params S3ConnectorParams) (*S3Connector, error) {
	client, err := initializeS3Client(S3ClientConfig{
		Endpoint: params.Endpoint,
		Region: params.Region,
		AccessKeyID: params.AccessKeyID,
		SecretAccessKey: params.SecretAccessKey,
	})
	slog.Info("Initialized S3 Connector", "params", params)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize s3 client: %w", err)
	}

	if err := validateParams(ctx, client, params.Bucket, params.Prefix); err != nil {
		return nil, fmt.Errorf("failed to validate s3 connector params: %w", err)
	}

	return &S3Connector{
		client:     client,
		params:     params,
	}, nil
}

var _ Connector = (*S3Connector)(nil)

func (c *S3Connector) CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error) {
	return createInferenceTasks(c.iterObjects(ctx, c.params.Bucket, c.params.Prefix), targetBytes)
}

func (c *S3Connector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams S3ConnectorTaskParams
	if err := json.Unmarshal(params, &parsedParams); err != nil {
		return nil, fmt.Errorf("error unmarshalling params: %w", err)
	}

	return iterTaskChunks(ctx, c.params.Bucket, parsedParams.ChunkKeys, c)
}

func (c *S3Connector) iterObjects(ctx context.Context, bucket, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		paginator := s3.NewListObjectsV2Paginator(c.client, &s3.ListObjectsV2Input{
			Bucket: aws.String(bucket),
			Prefix: aws.String(c.params.Prefix),
		})

		for paginator.HasMorePages() {
			page, err := paginator.NextPage(ctx)
			if err != nil {
				if !yield(Object{}, err) {
					return
				}
			}

			for _, obj := range page.Contents {
				if !yield(Object{Name: *obj.Key, Size: *obj.Size}, nil) {
					return
				}
			}
		}
	}
}

type s3ConnectorObjectStream struct {
	client *s3.Client
	bucket string
	key    string
	offset int
}

func (s *s3ConnectorObjectStream) Read(p []byte) (int, error) {
	rng := fmt.Sprintf("bytes=%d-%d", s.offset, s.offset+len(p)-1)

	resp, err := s.client.GetObject(context.TODO(), &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(s.key),
		Range:  aws.String(rng),
	})
	if err != nil {
		return 0, fmt.Errorf("failed to read object %s from s3://%s/%s: %w", rng, s.bucket, s.key, err)
	}
	defer resp.Body.Close()

	n, err := io.ReadFull(resp.Body, p)
	s.offset += n
	if err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return n, io.EOF
		}
		return n, fmt.Errorf("error reading resp body %s from s3://%s/%s: %w", rng, s.bucket, s.key, err)
	}
	return n, nil
}

func (c *S3Connector) GetObjectStream(ctx context.Context, bucket, key string) (io.Reader, error) {
	return &s3ConnectorObjectStream{client: c.client, bucket: bucket, key: key, offset: 0}, nil
}