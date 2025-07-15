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
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

type S3ObjectStore struct {
	bucket     string
	client     *s3.Client
	downloader *manager.Downloader
	uploader   *manager.Uploader
	cfg        S3ClientConfig
}

var _ ObjectStore = (*S3ObjectStore)(nil)

func NewS3ObjectStore(bucket string, cfg S3ClientConfig) (*S3ObjectStore, error) {
	client, err := initializeS3Client(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize s3 client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	if _, err := client.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(bucket),
	}); err != nil {
		var notFoundErr *types.NotFound
		if !errors.As(err, &notFoundErr) {
			slog.Error("Failed to verify bucket access", "bucket", bucket, "error", err)
			return nil, fmt.Errorf("failed to verify bucket access %s: %w", bucket, err)
		}

		slog.Info("Bucket does not exist, creating", "bucket", bucket)
		if _, err := client.CreateBucket(ctx, &s3.CreateBucketInput{
			Bucket: aws.String(bucket),
		}); err != nil {
			var alreadyExists *types.BucketAlreadyExists
			var alreadyOwned *types.BucketAlreadyOwnedByYou
			if errors.As(err, &alreadyExists) || errors.As(err, &alreadyOwned) {
				slog.Info("Bucket already exists or is owned by you", "bucket", bucket)
			} else {
				slog.Error("Failed to create bucket", "bucket", bucket, "error", err)
				return nil, fmt.Errorf("failed to create bucket %s: %w", bucket, err)
			}
		}

		slog.Info("Bucket created successfully", "bucket", bucket)
	}

	return &S3ObjectStore{
		bucket:     bucket,
		client:     client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
		cfg:        cfg,
	}, nil
}

func (s *S3ObjectStore) ListObjects(ctx context.Context, dir string) ([]Object, error) {
	var objects []Object

	paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
		Bucket: aws.String(s.bucket),
		Prefix: aws.String(dir),
	})

	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list objects in bucket %s with prefix %s: %w", s.bucket, dir, err)
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

func (s *S3ObjectStore) IterObjects(ctx context.Context, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		paginator := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{
			Bucket: aws.String(s.bucket),
			Prefix: aws.String(dir),
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

func (s *S3ObjectStore) downloadObject(ctx context.Context, key, filename string) error {
	if err := os.MkdirAll(filepath.Dir(filename), os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory for download %s: %w", filepath.Dir(filename), err)
	}
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()

	_, err = s.downloader.Download(ctx, file, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("failed to download object %s from s3://%s/%s: %w", filename, s.bucket, key, err)
	}
	slog.Info("Object downloaded successfully", "bucket", s.bucket, "key", key)

	return nil
}

func (s *S3ObjectStore) GetObject(ctx context.Context, key string) (io.ReadCloser, error) {
	resp, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get object %s from s3://%s/%s: %w", key, s.bucket, key, err)
	}

	return resp.Body, nil
}

func (s *S3ObjectStore) PutObject(ctx context.Context, key string, data io.Reader) error {
	_, err := s.uploader.Upload(ctx, &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Body:   data,
	})
	if err != nil {
		return fmt.Errorf("failed to upload object %s to s3://%s/%s: %w", key, s.bucket, key, err)
	}
	slog.Info("Object uploaded successfully", "bucket", s.bucket, "key", key)

	return nil
}

func (s *S3ObjectStore) DeleteObjects(ctx context.Context, dir string) error {
	for obj, err := range s.IterObjects(ctx, dir) {
		if err != nil {
			return fmt.Errorf("failed to iterate objects in bucket %s with dir %s: %w", s.bucket, dir, err)
		}

		_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
			Bucket: aws.String(s.bucket),
			Key:    aws.String(obj.Name),
		})
		if err != nil {
			return fmt.Errorf("failed to delete object in bucket %s with key %s: %w", s.bucket, obj.Name, err)
		}
	}

	slog.Info("Objects deleted successfully", "bucket", s.bucket, "dir", dir)

	return nil
}

func (s *S3ObjectStore) DownloadDir(ctx context.Context, src, dest string, overwrite bool) error {
	if _, err := os.Stat(dest); err == nil {
		if !overwrite {
			return fmt.Errorf("destination %s already exists and overwrite is false", dest)
		}
		if err := os.RemoveAll(dest); err != nil {
			return fmt.Errorf("failed to remove existing destination: %w", err)
		}
	}

	if err := os.MkdirAll(dest, os.ModePerm); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dest, err)
	}

	if !strings.HasSuffix(src, "/") {
		src += "/"
	}

	objects, err := s.ListObjects(ctx, src)
	if err != nil {
		return fmt.Errorf("error downloading directory %s/%s to %s: %w", s.bucket, src, dest, err)
	}

	for _, obj := range objects {
		localPath := filepath.Join(dest, strings.TrimPrefix(obj.Name, src))

		if err := s.downloadObject(ctx, obj.Name, localPath); err != nil {
			return fmt.Errorf("error downloading directory %s/%s to %s: %w", s.bucket, src, dest, err)
		}
	}

	return nil
}

func (s *S3ObjectStore) UploadDir(ctx context.Context, src, dest string) error {
	if !strings.HasSuffix(dest, "/") {
		dest += "/"
	}

	err := filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("failed to walk directory %s: %w", src, err)
		}

		if info.IsDir() {
			return nil
		}

		key := filepath.Join(dest, strings.TrimPrefix(path, src))

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		if err := s.PutObject(ctx, key, file); err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		return fmt.Errorf("error uploading directory %s to %s/%s: %w", src, s.bucket, dest, err)
	}

	return nil
}

func (s *S3ObjectStore) GetUploadConnector(ctx context.Context, uploadDir string, uploadParams UploadParams) (Connector, error) {
	return NewS3ConnectorWithAccessKey(
		ctx,
		S3ConnectorParams{
			Endpoint: s.cfg.Endpoint,
			Region:   s.cfg.Region,
			Bucket:   s.bucket,
			Prefix:   filepath.Join(uploadDir, uploadParams.UploadId.String()),
		},
		s.cfg.AccessKeyID,
		s.cfg.SecretAccessKey,
	)
}
