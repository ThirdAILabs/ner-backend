package s3

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"ner-backend/internal/config"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/google/uuid"
)

type Client struct {
	s3Client   *s3.Client
	downloader *manager.Downloader
	uploader   *manager.Uploader
	bucketName string // Default model bucket
}

func NewS3Client(cfg *config.Config) (*Client, error) {
	resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) {
		if cfg.S3EndpointURL != "" {
			return aws.Endpoint{
				PartitionID:       "aws",
				URL:               cfg.S3EndpointURL,
				SigningRegion:     cfg.S3Region,
				HostnameImmutable: true, // Important for MinIO
			}, nil
		}
		// fallback to default AWS endpoint resolution
		return aws.Endpoint{}, &aws.EndpointNotFoundError{}
	})

	awsCfg, err := aws_config.LoadDefaultConfig(context.TODO(),
		aws_config.WithRegion(cfg.S3Region),
		aws_config.WithEndpointResolverWithOptions(resolver),
		aws_config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(cfg.S3AccessKeyID, cfg.S3SecretAccessKey, "")),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	s3Client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return &Client{
		s3Client:   s3Client,
		downloader: manager.NewDownloader(s3Client),
		uploader:   manager.NewUploader(s3Client),
		bucketName: cfg.ModelBucketName,
	}, nil
}

func (c *Client) UploadFile(ctx context.Context, localPath, bucket, key string) (string, error) {
	file, err := os.Open(localPath)
	if err != nil {
		return "", fmt.Errorf("failed to open file %s: %w", localPath, err)
	}
	defer file.Close()

	log.Printf("Uploading %s to s3://%s/%s", localPath, bucket, key)
	_, err = c.uploader.Upload(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   file,
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

func (c *Client) ListFiles(ctx context.Context, bucket, prefix string) ([]string, error) {
	var keys []string
	paginator := s3.NewListObjectsV2Paginator(c.s3Client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})

	log.Printf("Listing files in s3://%s/%s", bucket, prefix)
	pageCount := 0
	for paginator.HasMorePages() {
		pageCount++
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to list objects (page %d) in s3://%s/%s: %w", pageCount, bucket, prefix, err)
		}
		for _, obj := range page.Contents {
			if obj.Key != nil && !strings.HasSuffix(*obj.Key, "/") { // Exclude "directories"
				keys = append(keys, *obj.Key)
			}
		}
	}
	log.Printf("Found %d files in s3://%s/%s", len(keys), bucket, prefix)
	return keys, nil
}

func ParseS3Path(s3Path string) (bucket, key string, err error) {
	parsed, err := url.Parse(s3Path)
	if err != nil {
		return "", "", fmt.Errorf("invalid S3 path '%s': %w", s3Path, err)
	}
	if parsed.Scheme != "s3" {
		return "", "", fmt.Errorf("invalid scheme in S3 path '%s', expected 's3'", s3Path)
	}
	bucket = parsed.Host
	key = strings.TrimPrefix(parsed.Path, "/")
	return bucket, key, nil
}

// --- Model Specific Wrappers ---

func (c *Client) UploadModelArtifact(ctx context.Context, localPath string, modelId uuid.UUID, filename string) (string, error) {
	key := fmt.Sprintf("%s/%s", modelId.String(), filename)
	return c.UploadFile(ctx, localPath, c.bucketName, key)
}

func (c *Client) DownloadModelArtifact(ctx context.Context, modelId uuid.UUID, downloadDir, filename string) (string, error) {
	key := fmt.Sprintf("%s/%s", modelId.String(), filename)
	localPath := filepath.Join(downloadDir, filename)
	err := c.DownloadFile(ctx, c.bucketName, key, localPath)
	if err != nil {
		return "", err // Error includes details already
	}
	return localPath, nil
}

func (c *Client) DownloadTrainingData(ctx context.Context, sourceS3Path, downloadDir string) ([]string, error) {
	bucket, prefix, err := ParseS3Path(sourceS3Path)
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(downloadDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory %s: %w", downloadDir, err)
	}

	keys, err := c.ListFiles(ctx, bucket, prefix)
	if err != nil {
		return nil, err // Error includes details
	}

	var downloadedFiles []string
	for _, key := range keys {
		localFilename := filepath.Join(downloadDir, filepath.Base(key))
		err := c.DownloadFile(ctx, bucket, key, localFilename)
		if err != nil {
			log.Printf("Failed to download training file %s: %v", key, err)
		} else {
			downloadedFiles = append(downloadedFiles, localFilename)
		}
	}

	if len(downloadedFiles) == 0 && len(keys) > 0 {
		return nil, fmt.Errorf("failed to download any training files from %s", sourceS3Path)
	}
	return downloadedFiles, nil
}

func (c *Client) ListAndChunkS3Objects( // Changed to a method with receiver (c *Client)
	ctx context.Context,
	bucket string,
	prefix string,
	targetChunkBytes int64,
	callerID uuid.UUID, // Generic ID for logging context (e.g., Job ID, Request ID)
	processChunk func(ctx context.Context, chunkKeys []string, chunkSize int64) error,
) (int, error) {
	// Using slog.Default() for logging. Could use c.logger if it were initialized.
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
