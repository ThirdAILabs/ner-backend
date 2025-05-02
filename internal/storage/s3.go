package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

type S3Provider struct {
	client     *s3.Client
	downloader *manager.Downloader
	uploader   *manager.Uploader
}

type S3ProviderConfig struct {
	S3EndpointURL     string
	S3AccessKeyID     string
	S3SecretAccessKey string
	S3Region          string
}

func NewS3Provider(cfg *S3ProviderConfig) (*S3Provider, error) {
	resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) { // nolint:staticcheck
		if cfg.S3EndpointURL != "" {
			return aws.Endpoint{ // nolint:staticcheck
				PartitionID:       "aws",
				URL:               cfg.S3EndpointURL,
				SigningRegion:     cfg.S3Region,
				HostnameImmutable: true, // Important for MinIO
			}, nil
		}
		// fallback to default AWS endpoint resolution
		return aws.Endpoint{}, &aws.EndpointNotFoundError{} // nolint:staticcheck
	})

	awsCfg, err := aws_config.LoadDefaultConfig(context.TODO(),
		aws_config.WithRegion(cfg.S3Region),
		aws_config.WithEndpointResolverWithOptions(resolver), // nolint:staticcheck
		aws_config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(cfg.S3AccessKeyID, cfg.S3SecretAccessKey, "")),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return &S3Provider{
		client:     client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
	}, nil
}

func (s *S3Provider) CreateBucket(ctx context.Context, bucket string) error {
	_, err := s.client.CreateBucket(ctx, &s3.CreateBucketInput{
		Bucket: aws.String(bucket),
	})
	if err != nil {
		var existErr *types.BucketAlreadyExists
		var ownedErr *types.BucketAlreadyOwnedByYou
		if errors.As(err, &existErr) || errors.As(err, &ownedErr) {
			slog.Info("Bucket already exists", "bucket", bucket)
			return nil
		}

		return fmt.Errorf("failed to create bucket %s: %w", bucket, err)
	}

	slog.Info("Bucket created successfully", "bucket", bucket)

	return nil
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

func (s *S3Provider) DownloadObject(ctx context.Context, bucket, key, filename string) error {
	if err := os.MkdirAll(filepath.Dir(filename), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory for download %s: %w", filepath.Dir(filename), err)
	}
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()

	_, err = s.downloader.Download(ctx, file, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("failed to download object %s from s3://%s/%s: %w", filename, bucket, key, err)
	}
	slog.Info("Object downloaded successfully", "bucket", bucket, "key", key)

	return nil
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

func (s *S3Provider) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	_, err := s.uploader.Upload(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   data,
	})
	if err != nil {
		return fmt.Errorf("failed to upload object %s to s3://%s/%s: %w", key, bucket, key, err)
	}
	slog.Info("Object uploaded successfully", "bucket", bucket, "key", key)

	return nil
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
