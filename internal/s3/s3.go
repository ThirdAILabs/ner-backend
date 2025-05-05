package s3

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/google/uuid"
)

type S3Api interface {
	manager.DownloadAPIClient
	manager.UploadAPIClient
	manager.ListObjectsV2APIClient

	CreateBucket(ctx context.Context, params *s3.CreateBucketInput, optFns ...func(*s3.Options)) (*s3.CreateBucketOutput, error)
}

type Client struct {
	s3Client   S3Api
	downloader *manager.Downloader
	uploader   *manager.Uploader
	bucketName string // Default model bucket
}

type Config struct {
	S3EndpointURL     string
	S3AccessKeyID     string
	S3SecretAccessKey string
	S3Region          string
	ModelBucketName   string
}

func NewS3Client(cfg *Config) (*Client, error) {
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

	s3Client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return NewFromClient(s3Client, cfg.ModelBucketName), nil
}

func NewFromClient(client S3Api, bucketName string) *Client {
	return &Client{
		s3Client:   client,
		downloader: manager.NewDownloader(client),
		uploader:   manager.NewUploader(client),
		bucketName: bucketName,
	}
}

func (c *Client) UploadDirectory(ctx context.Context, localDirPath, bucket, prefix string) (string, error) {
	walkErr := filepath.Walk(localDirPath, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("error accessing path %q: %w", filePath, err)
		}

		if info.IsDir() {
			return nil
		}

		relPath, err := filepath.Rel(localDirPath, filePath)
		if err != nil {
			return fmt.Errorf("failed to get relative path for %s: %w", filePath, err)
		}

		s3Key := path.Join(prefix, filepath.ToSlash(relPath))

		_, uploadErr := c.UploadFile(ctx, filePath, bucket, s3Key)
		if uploadErr != nil {

			return fmt.Errorf("upload failed for %s: %w", filePath, uploadErr)
		}
		return nil
	})

	if walkErr != nil {
		return "", walkErr
	}

	s3PrefixPath := fmt.Sprintf("s3://%s/%s", bucket, prefix)
	log.Printf("Successfully uploaded directory %s to %s", localDirPath, s3PrefixPath)
	return s3PrefixPath, nil
}

func (c *Client) UploadFile(ctx context.Context, localPath, bucket, key string) (string, error) {
	file, err := os.Open(localPath)
	if err != nil {
		return "", fmt.Errorf("failed to open file %s: %w", localPath, err)
	}
	defer file.Close()

	return c.UploadObject(ctx, bucket, key, file)
}

func (c *Client) UploadObject(ctx context.Context, bucket, key string, data io.Reader) (string, error) {
	log.Printf("Uploading object to s3://%s/%s", bucket, key)
	_, err := c.uploader.Upload(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   data,
	})
	if err != nil {
		return "", fmt.Errorf("failed to upload file to s3://%s/%s: %w", bucket, key, err)
	}
	s3Path := fmt.Sprintf("s3://%s/%s", bucket, key)
	log.Printf("Successfully uploaded to %s", s3Path)
	return s3Path, nil
}

func (c *Client) DownloadFile(ctx context.Context, bucket, key, localPath string) error {
	// Ensure directory exists
	dir := filepath.Dir(localPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	file, err := os.Create(localPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", localPath, err)
	}
	defer file.Close()

	log.Printf("Downloading s3://%s/%s to %s", bucket, key, localPath)
	_, err = c.downloader.Download(ctx, file, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		// Clean up empty file on failure
		file.Close()
		os.Remove(localPath)
		return fmt.Errorf("failed to download file s3://%s/%s: %w", bucket, key, err)
	}
	log.Printf("Successfully downloaded to %s", localPath)
	return nil
}

func (c *Client) DownloadDirectory(ctx context.Context, bucket, prefix, localDir string) error {
	logArgs := []any{
		slog.String("bucket", bucket),
		slog.String("s3_prefix", prefix),
		slog.String("local_dir", localDir),
	}
	slog.Info("Starting exact download of S3 directory", logArgs...)

	// 1. Ensure the base local directory exists
	if err := os.MkdirAll(localDir, 0755); err != nil {
		slog.Error("Failed to create base local directory", append(logArgs, slog.Any("error", err))...)
		return fmt.Errorf("failed to create base local directory %s: %w", localDir, err)
	}

	// 2. Ensure prefix format is consistent
	originalPrefix := prefix
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
		slog.Debug("Adjusted S3 prefix to end with '/'",
			slog.String("original_prefix", originalPrefix),
			slog.String("adjusted_prefix", prefix),
		)
		logArgs = []any{
			slog.String("bucket", bucket),
			slog.String("s3_prefix", prefix), // Use updated prefix
			slog.String("local_dir", localDir),
		}
	}

	// 3. List objects using a paginator
	paginator := s3.NewListObjectsV2Paginator(c.s3Client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})

	pageCount := 0
	objectCount := 0
	dirCount := 0
	// 4. Iterate through pages of results
	for paginator.HasMorePages() {
		pageCount++
		slog.Debug("Fetching S3 object list page", append(logArgs, slog.Int("page_number", pageCount))...)
		page, err := paginator.NextPage(ctx)
		if err != nil {
			slog.Error("Failed to list objects page from S3", append(logArgs, slog.Int("page_number", pageCount), slog.Any("error", err))...)
			return fmt.Errorf("failed to list objects page %d in s3://%s/%s: %w", pageCount, bucket, prefix, err)
		}

		// 5. Process each object in the current page
		for _, obj := range page.Contents {
			objectKey := aws.ToString(obj.Key)
			objLogArgs := append(logArgs, slog.String("s3_key", objectKey), slog.Int64("size_bytes", *obj.Size))

			// 6. Calculate the relative path and full local path
			relativePath := filepath.FromSlash(strings.TrimPrefix(objectKey, prefix))
			localPath := filepath.Join(localDir, relativePath)
			objLogArgs = append(objLogArgs, slog.String("local_path", localPath))

			// 7. Check if the S3 object represents a directory
			if strings.HasSuffix(objectKey, "/") {
				// It's an S3 directory object. Ensure the corresponding local directory exists.
				if objectKey != prefix && *obj.Size == 0 { // Check size is 0 for directory markers
					slog.Debug("Creating local directory for S3 directory object", objLogArgs...)
					if err := os.MkdirAll(localPath, 0755); err != nil {
						slog.Error("Failed to create local directory for S3 directory object", append(objLogArgs, slog.Any("error", err))...)
						return fmt.Errorf("failed to create local directory %s for S3 directory %s: %w", localPath, objectKey, err)
					}
					dirCount++
				} else if objectKey == prefix {
					slog.Debug("Skipping explicit directory creation for base prefix", objLogArgs...)
				} else {
					slog.Warn("Skipping object ending in '/' but not a typical directory marker", objLogArgs...)
				}
			} else {
				// It's an S3 file object. Download it.
				objectCount++
				err := c.DownloadFile(ctx, bucket, objectKey, localPath)
				if err != nil {
					return fmt.Errorf("failed during directory download (object: %s): %w", objectKey, err)
				}
			}
		}
	}

	slog.Info("Finished S3 directory processing",
		append(logArgs,
			slog.Int("files_downloaded", objectCount),
			slog.Int("directories_created", dirCount),
			slog.Int("pages_processed", pageCount),
		)...)
	return nil
}

type s3ObjectStream struct {
	client S3Api
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

func (c *Client) DownloadFileStream(bucket, key string) io.Reader {
	return &s3ObjectStream{client: c.s3Client, bucket: bucket, key: key, offset: 0}
}

func (c *Client) ListAndChunkS3Objects(
	ctx context.Context,
	bucket string,
	prefix string,
	targetChunkBytes int64,
	callerID uuid.UUID,
	processChunk func(ctx context.Context, chunkKeys []string, chunkSize int64) error,
) (int, error) {
	logger := slog.Default().With("component", "S3Chunker", "caller", callerID, "bucket", bucket, "prefix", prefix)

	// Use the client from the receiver (c.s3Client) instead of a parameter
	paginator := s3.NewListObjectsV2Paginator(c.s3Client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})

	var currentChunkKeys []string
	var currentChunkSizeBytes int64 = 0
	var totalChunksProcessed int = 0
	pageCount := 0

	logger.Info("Starting S3 listing", "targetBytes", targetChunkBytes)

	for paginator.HasMorePages() {
		pageCount++
		pageLogger := logger.With("page", pageCount)
		pageLogger.Debug("Fetching S3 page")
		page, err := paginator.NextPage(ctx)
		if err != nil {
			pageLogger.Error("Failed to list objects page", "error", err)
			return totalChunksProcessed, fmt.Errorf("failed to list objects page %d for caller %s: %w", pageCount, callerID, err)
		}

		for _, obj := range page.Contents {
			if obj.Key == nil || strings.HasSuffix(*obj.Key, "/") || obj.Size == nil {
				continue
			}
			objectKey := *obj.Key
			objectSize := *obj.Size
			if objectSize == 0 {
				logger.Debug("Skipping zero-byte object", "key", objectKey)
				continue
			}

			// Finalize previous chunk if needed
			if len(currentChunkKeys) > 0 && (currentChunkSizeBytes+objectSize) > targetChunkBytes {
				chunkNum := totalChunksProcessed + 1
				chunkLogger := logger.With("chunkNum", chunkNum, "keyCount", len(currentChunkKeys), "sizeBytes", currentChunkSizeBytes)
				chunkLogger.Info("Finalizing chunk", "reason", "next_item_exceeds_size", "nextKey", objectKey, "nextSize", objectSize)

				keysToProcess := make([]string, len(currentChunkKeys))
				copy(keysToProcess, currentChunkKeys)
				if err := processChunk(ctx, keysToProcess, currentChunkSizeBytes); err != nil {
					chunkLogger.Error("Processing chunk failed", "error", err)
					return totalChunksProcessed, fmt.Errorf("processing chunk %d failed for caller %s: %w", chunkNum, callerID, err)
				}
				totalChunksProcessed++
				currentChunkKeys = []string{}
				currentChunkSizeBytes = 0
			}

			// Add current object to chunk
			currentChunkKeys = append(currentChunkKeys, objectKey)
			currentChunkSizeBytes += objectSize

			// Finalize current chunk if target met/exceeded
			if currentChunkSizeBytes >= targetChunkBytes {
				chunkNum := totalChunksProcessed + 1
				chunkLogger := logger.With("chunkNum", chunkNum, "keyCount", len(currentChunkKeys), "sizeBytes", currentChunkSizeBytes)
				chunkLogger.Info("Finalizing chunk", "reason", "target_met_or_exceeded")

				keysToProcess := make([]string, len(currentChunkKeys))
				copy(keysToProcess, currentChunkKeys)
				if err := processChunk(ctx, keysToProcess, currentChunkSizeBytes); err != nil {
					chunkLogger.Error("Processing chunk failed", "error", err)
					return totalChunksProcessed, fmt.Errorf("processing chunk %d failed for caller %s: %w", chunkNum, callerID, err)
				}
				totalChunksProcessed++
				currentChunkKeys = []string{}
				currentChunkSizeBytes = 0
			}
		}
	}

	// Process final chunk
	if len(currentChunkKeys) > 0 {
		chunkNum := totalChunksProcessed + 1
		chunkLogger := logger.With("chunkNum", chunkNum, "keyCount", len(currentChunkKeys), "sizeBytes", currentChunkSizeBytes)
		chunkLogger.Info("Finalizing last chunk")

		if err := processChunk(ctx, currentChunkKeys, currentChunkSizeBytes); err != nil {
			chunkLogger.Error("Processing final chunk failed", "error", err)
			return totalChunksProcessed, fmt.Errorf("processing final chunk %d failed for caller %s: %w", chunkNum, callerID, err)
		}
		totalChunksProcessed++
	}

	logger.Info("Finished S3 listing", slog.Int("processedChunks", totalChunksProcessed))
	return totalChunksProcessed, nil // Success
}

func (c *Client) CreateBucket(ctx context.Context, bucketName string) error {
	_, err := c.s3Client.CreateBucket(ctx, &s3.CreateBucketInput{
		Bucket: aws.String(bucketName),
	})
	if err != nil {
		var existErr *types.BucketAlreadyExists
		var ownedErr *types.BucketAlreadyOwnedByYou
		if errors.As(err, &existErr) || errors.As(err, &ownedErr) {
			slog.Info("Bucket already exists", "bucketName", bucketName)
			return nil
		}

		return fmt.Errorf("failed to create bucket %s: %w", bucketName, err)
	}
	log.Printf("Successfully created bucket %s", bucketName)
	return nil
}
