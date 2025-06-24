package storage

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/feature/s3/manager"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

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

func initializeS3Client(cfg S3ObjectStoreConfig) (*s3.Client, *manager.Downloader, *manager.Uploader, error) {
	var creds aws.CredentialsProvider = nil
	if cfg.AccessKeyID != "" && cfg.SecretAccessKey != "" {
		creds = credentials.NewStaticCredentialsProvider(cfg.AccessKeyID, cfg.SecretAccessKey, "")
	}

	awsCfg, err := createS3Config(cfg.Endpoint, cfg.Region, creds)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create aws config: %w", err)
	}

	// This checks if credentials can be loaded from the environment, for example from
	// env variables or ~/.aws/credentials. If no credentials are found, then we fallback
	// to anonymous credentials, this is needed to be able to access public s3 buckets.
	if _, err := awsCfg.Credentials.Retrieve(context.Background()); err != nil {
		awsCfg, err = createS3Config(cfg.Endpoint, cfg.Region, aws.AnonymousCredentials{})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to create aws config with anonymous credentials: %w", err)
		}
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return client, manager.NewDownloader(client), manager.NewUploader(client), nil
}