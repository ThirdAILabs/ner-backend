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
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

type S3ObjectStore struct {
	client     *s3.Client
	downloader *manager.Downloader
	uploader   *manager.Uploader
	cfg        S3ClientConfig
}

var _ ObjectStore = (*S3ObjectStore)(nil)


func NewS3ObjectStore(cfg S3ClientConfig) (*S3ObjectStore, error) {
	client, err := initializeS3Client(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize s3 client: %w", err)
	}

	return &S3ObjectStore{
		client:     client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
		cfg:        cfg,
	}, nil
}

func (s *S3ObjectStore) listObjects(ctx context.Context, bucket, prefix string) ([]Object, error) {
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

func (s *S3ObjectStore) iterObjects(ctx context.Context, bucket, prefix string) ObjectIterator {
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

func (s *S3ObjectStore) DownloadObject(ctx context.Context, bucket, key, filename string) error {
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

func (s *S3ObjectStore) CreateBucket(ctx context.Context, bucket string) error {
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

func (s *S3ObjectStore) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
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

func (s *S3ObjectStore) DeleteObjects(ctx context.Context, bucket string, prefix string) error {
	for obj, err := range s.iterObjects(ctx, bucket, prefix) {
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

func (s *S3ObjectStore) DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error {
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

	if !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	objects, err := s.listObjects(ctx, bucket, prefix)
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

func (s *S3ObjectStore) UploadDir(ctx context.Context, bucket, prefix, src string) error {
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

func (s *S3ObjectStore) GetUploadConnector(ctx context.Context, bucket string, uploadParams UploadParams) (Connector, error) {
	return NewS3Connector(
		ctx,
		S3ConnectorParams{
			Endpoint: s.cfg.Endpoint,
			Region: s.cfg.Region,
			Bucket: bucket,
			Prefix: uploadParams.UploadId.String(),
			AccessKeyID: s.cfg.AccessKeyID,
			SecretAccessKey: s.cfg.SecretAccessKey,
		},
	)
}