package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
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
	downloader *manager.Downloader
	uploader   *manager.Uploader
	params S3ConnectorParams
}

func createS3ConnectorConfig(s3Endpoint, s3Region string, creds aws.CredentialsProvider) (aws.Config, error) {
	opts := []func(*aws_config.LoadOptions) error{}

	if s3Endpoint != "" {
		resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) { // nolint:staticcheck
			return aws.Endpoint{ // nolint:staticcheck
				PartitionID:       "aws",
				URL:               s3Endpoint,
				SigningRegion:     s3Region,
				HostnameImmutable: true, // Important for MinIO
			}, nil
		})

		opts = append(opts, aws_config.WithEndpointResolverWithOptions(resolver)) // nolint:staticcheck
	}

	if s3Region != "" {
		opts = append(opts, aws_config.WithRegion(s3Region))
	}

	if creds != nil {
		opts = append(opts, aws_config.WithCredentialsProvider(creds))
	}

	return aws_config.LoadDefaultConfig(context.Background(), opts...)
}


func NewS3Connector(params S3ConnectorParams) (*S3Connector, error) {
	var creds aws.CredentialsProvider = nil
	if params.AccessKeyID != "" && params.SecretAccessKey != "" {
		creds = credentials.NewStaticCredentialsProvider(params.AccessKeyID, params.SecretAccessKey, "")
	}

	awsCfg, err := createS3ConnectorConfig(params.Endpoint, params.Region, creds)
	if err != nil {
		return nil, fmt.Errorf("failed to create aws config: %w", err)
	}

	// This checks if credentials can be loaded from the environment, for example from
	// env variables or ~/.aws/credentials. If no credentials are found, then we fallback
	// to anonymous credentials, this is needed to be able to access public s3 buckets.
	if _, err := awsCfg.Credentials.Retrieve(context.Background()); err != nil {
		awsCfg, err = createS3ConnectorConfig(params.Endpoint, params.Region, aws.AnonymousCredentials{})
		if err != nil {
			return nil, fmt.Errorf("failed to create aws config with anonymous credentials: %w", err)
		}
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return &S3Connector{
		client:     client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
		params:     params,
	}, nil
}

var _ Connector = &S3Connector{}

func (c *S3Connector) Type() string {
	return S3ConnectorType
}

func (c *S3Connector) GetParams() ([]byte, error) {
	cfgJson, err := json.Marshal(c.params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal s3 connector params: %w", err)
	}

	return cfgJson, nil
}

func (c *S3Connector) ValidateParams(ctx context.Context) error {
	if c.params.Bucket == "" {
		return fmt.Errorf("bucket is required")
	}
	
	if _, err := c.client.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(c.params.Bucket),
	}); err != nil {
		return fmt.Errorf("failed to verify access to s3://%s: %w", c.params.Bucket, err)
	}

	if _, err := c.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket: aws.String(c.params.Bucket),
		Prefix: aws.String(c.params.Prefix),
	}); err != nil {
		return fmt.Errorf("failed to verify objects in s3://%s/%s: %w", c.params.Bucket, c.params.Prefix, err)
	}

	return nil
}

func (c *S3Connector) CreateInferenceTasks(ctx context.Context, targetBytes int64) ([]InferenceTask, int64, error) {
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

	for obj, err := range c.iterObjects(ctx, c.params.Bucket, c.params.Prefix) {
		if err != nil {
			return nil, 0, fmt.Errorf("error iterating over local objects: %w", err)
		}

		totalObjects++

		if currentChunkSize+obj.Size > targetBytes && len(currentChunkKeys) > 0 {
			if err := addTask(currentChunkKeys, currentChunkSize); err != nil {
				return nil, 0, err
			}

			currentChunkKeys = []string{}
			currentChunkSize = 0
		}

		currentChunkKeys = append(currentChunkKeys, obj.Name)
		currentChunkSize += obj.Size
	}

	if len(currentChunkKeys) > 0 {
		if err := addTask(currentChunkKeys, currentChunkSize); err != nil {
			return nil, 0, err
		}
	}
	return tasks, int64(totalObjects), nil
}

func (c *S3Connector) IterTaskChunks(ctx context.Context, params []byte) (<-chan ObjectChunkStream, error) {
	var parsedParams S3ConnectorTaskParams
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

func (c *S3Connector) getObjectStream(bucket, key string) (io.Reader, error) {
	return &s3ConnectorObjectStream{client: c.client, bucket: bucket, key: key, offset: 0}, nil
}