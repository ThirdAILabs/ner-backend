package storage

import (
	"context"
	"fmt"
	"io"
	"log/slog"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3Provider struct {
	client     *s3.Client
	downloader *manager.Downloader
	uploader   *manager.Uploader
	cfg        S3ProviderConfig
}

type S3ProviderConfig struct {
	S3EndpointURL     string
	S3AccessKeyID     string
	S3SecretAccessKey string
	S3Region          string
}

func createS3Config(s3Endpoint, s3Region string, creds aws.CredentialsProvider) (aws.Config, error) {
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

func NewS3Provider(cfg S3ProviderConfig) (*S3Provider, error) {
	var creds aws.CredentialsProvider = nil
	if cfg.S3AccessKeyID != "" && cfg.S3SecretAccessKey != "" {
		creds = credentials.NewStaticCredentialsProvider(cfg.S3AccessKeyID, cfg.S3SecretAccessKey, "")
	}

	awsCfg, err := createS3Config(cfg.S3EndpointURL, cfg.S3Region, creds)
	if err != nil {
		return nil, fmt.Errorf("failed to create aws config: %w", err)
	}

	// This checks if credentials can be loaded from the environment, for example from
	// env variables or ~/.aws/credentials. If no credentials are found, then we fallback
	// to anonymous credentials, this is needed to be able to access public s3 buckets.
	if _, err := awsCfg.Credentials.Retrieve(context.Background()); err != nil {
		awsCfg, err = createS3Config(cfg.S3EndpointURL, cfg.S3Region, aws.AnonymousCredentials{})
		if err != nil {
			return nil, fmt.Errorf("failed to create aws config with anonymous credentials: %w", err)
		}
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return &S3Provider{
		client:     client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
		cfg:        cfg,
	}, nil
}

func (s *S3Provider) GetObject(ctx context.Context, bucket, key string) ([]byte, error) {
	// get object size using client
	headObj, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get object size: %w", err)
	}

	buffer := manager.NewWriteAtBuffer(make([]byte, *headObj.ContentLength))

	_, err = s.downloader.Download(ctx, buffer, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to download object: %w", err)
	}
	slog.Info("Object downloaded successfully", "bucket", bucket, "key", key)

	return buffer.Bytes(), nil
}

type s3ObjectStream struct {
	client *s3.Client
	bucket string
	key    string
	offset int
}

func (s *s3ObjectStream) Read(p []byte) (int, error) {
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

func (s *S3Provider) GetObjectStream(bucket, key string) (io.Reader, error) {
	return &s3ObjectStream{client: s.client, bucket: bucket, key: key, offset: 0}, nil
}

func (s *S3Provider) ListObjects(ctx context.Context, bucket, prefix string) ([]Object, error) {
	var objects []Object

	paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list objects in bucket %s with prefix %s: %w", bucket, prefix, err)
		}

		for _, obj := range page.Contents {
			objects = append(objects, Object{
				Name: *obj.Key,
				Size: *obj.Size,
			})
		}
	}

	return objects, nil
}

func (s *S3Provider) IterObjects(ctx context.Context, bucket, prefix string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
			Bucket: aws.String(bucket),
			Prefix: aws.String(prefix),
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

func (s *S3Provider) DeleteObjects(ctx context.Context, bucket string, prefix string) error {
	for obj, err := range s.IterObjects(ctx, bucket, prefix) {
		if err != nil {
			return fmt.Errorf("failed to iterate objects in bucket %s with prefix %s: %w", bucket, prefix, err)
		}

		_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
			Bucket: aws.String(bucket),
			Key:    aws.String(obj.Name),
		})
		if err != nil {
			return fmt.Errorf("failed to delete objects in bucket %s with prefix %s: %w", bucket, prefix, err)
		}
	}

	slog.Info("Objects deleted successfully", "bucket", bucket, "prefix", prefix)

	return nil
}

func (s *S3Provider) ValidateAccess(ctx context.Context, bucket, prefix string) error {
	if _, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(bucket),
	}); err != nil {
		return fmt.Errorf("failed to verify access to s3://%s: %w", bucket, err)
	}

	if _, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	}); err != nil {
		return fmt.Errorf("failed to verify objects in s3://%s/%s: %w", bucket, prefix, err)
	}

	return nil
}
