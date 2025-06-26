package storage

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	aws_config "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3ClientConfig struct {
	Endpoint string
	Region string
	AccessKeyID string
	SecretAccessKey string
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

func initializeS3Client(cfg S3ClientConfig) (*s3.Client, error) {
	var creds aws.CredentialsProvider = nil
	if cfg.AccessKeyID != "" && cfg.SecretAccessKey != "" {
		creds = credentials.NewStaticCredentialsProvider(cfg.AccessKeyID, cfg.SecretAccessKey, "")
	}

	awsCfg, err := createS3Config(cfg.Endpoint, cfg.Region, creds)
	if err != nil {
		return nil, fmt.Errorf("failed to create aws config: %w", err)
	}

	// This checks if credentials can be loaded from the environment, for example from
	// env variables or ~/.aws/credentials. If no credentials are found, then we fallback
	// to anonymous credentials, this is needed to be able to access public s3 buckets.
	if _, err := awsCfg.Credentials.Retrieve(context.Background()); err != nil {
		awsCfg, err = createS3Config(cfg.Endpoint, cfg.Region, aws.AnonymousCredentials{})
		if err != nil {
			return nil, fmt.Errorf("failed to create aws config with anonymous credentials: %w", err)
		}
	}

	client := s3.NewFromConfig(awsCfg, func(o *s3.Options) {
		// Needed for MinIO which doesn't enforce bucket naming rules always
		o.UsePathStyle = true // Use path-style addressing (needed for MinIO) - Assuming true based on original, not cfg.S3UsePathStyle
	})

	return client, nil
}

func validateParams(ctx context.Context, client *s3.Client, bucket, prefix string) error {
	if bucket == "" {
		return fmt.Errorf("bucket is required")
	}
	
	if _, err := client.HeadBucket(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(bucket),
	}); err != nil {
		return fmt.Errorf("failed to verify access to s3://%s: %w", bucket, err)
	}

	if _, err := client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	}); err != nil {
		return fmt.Errorf("failed to verify objects in s3://%s/%s: %w", bucket, prefix, err)
	}

	return nil
}