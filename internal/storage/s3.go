package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

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

func (s *S3Provider) DownloadDir(ctx context.Context, bucket, prefix, dest string) error {
	if err := os.MkdirAll(dest, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dest, err)
	}

	if !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	objects, err := s.ListObjects(ctx, bucket, prefix)
	if err != nil {
		return fmt.Errorf("error downloading directory %s/%s to %s: %w", bucket, prefix, dest, err)
	}

	for _, obj := range objects {
		localPath := filepath.Join(dest, strings.TrimPrefix(obj.Name, prefix))

		if err := s.DownloadObject(ctx, bucket, obj.Name, localPath); err != nil {
			return fmt.Errorf("error downloading directory %s/%s to %s: %w", bucket, prefix, dest, err)
		}
	}

	return nil
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

func (s *S3Provider) UploadDir(ctx context.Context, bucket, prefix, src string) error {
	if !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	err := filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("failed to walk directory %s: %w", src, err)
		}

		if info.IsDir() {
			return nil
		}

		key := filepath.Join(prefix, strings.TrimPrefix(path, src))

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		if err := s.PutObject(ctx, bucket, key, file); err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("error uploading directory %s to %s/%s: %w", src, bucket, prefix, err)
	}

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

func (s *S3Provider) ValidateAccess(ctx context.Context, bucket string) error {
	_, err := s.client.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(bucket),
	})

	if err != nil {
		return fmt.Errorf("failed to verify access to s3://%s: %w", bucket, err)
	}
	return nil
}
