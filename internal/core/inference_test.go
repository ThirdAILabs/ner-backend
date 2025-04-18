package core

import (
	"context"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/s3"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	aws_s3 "github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/google/uuid"
)

type regexModel struct {
	patterns map[string]regexp.Regexp
}

func (m *regexModel) Predict(text string) ([]types.Entity, error) {
	var entities []types.Entity
	for label, pattern := range m.patterns {
		matches := pattern.FindAllStringSubmatchIndex(text, -1)
		for _, match := range matches {
			if len(match) > 0 {
				entities = append(entities, types.Entity{
					Label: label,
					Text:  text[match[0]:match[1]],
					Start: match[0],
					End:   match[1],
				})
			}
		}
	}
	return entities, nil
}

func (m *regexModel) Release() {}

type mockS3Client struct{}

type stringReadCloser struct{ *strings.Reader }

func (r *stringReadCloser) Close() error { return nil }

const testDoc = "This is a test doc. It contains a phone number: 012-345-6789, and a email: test@email.com."

func (m *mockS3Client) GetObject(context.Context, *aws_s3.GetObjectInput, ...func(*aws_s3.Options)) (*aws_s3.GetObjectOutput, error) {
	return &aws_s3.GetObjectOutput{
		Body: &stringReadCloser{
			Reader: strings.NewReader(testDoc),
		},
	}, nil
}

// Unused for these tests
func (m *mockS3Client) PutObject(context.Context, *aws_s3.PutObjectInput, ...func(*aws_s3.Options)) (*aws_s3.PutObjectOutput, error) {
	return nil, nil
}
func (m *mockS3Client) UploadPart(context.Context, *aws_s3.UploadPartInput, ...func(*aws_s3.Options)) (*aws_s3.UploadPartOutput, error) {
	return nil, nil
}
func (m *mockS3Client) CreateMultipartUpload(context.Context, *aws_s3.CreateMultipartUploadInput, ...func(*aws_s3.Options)) (*aws_s3.CreateMultipartUploadOutput, error) {
	return nil, nil
}
func (m *mockS3Client) CompleteMultipartUpload(context.Context, *aws_s3.CompleteMultipartUploadInput, ...func(*aws_s3.Options)) (*aws_s3.CompleteMultipartUploadOutput, error) {
	return nil, nil
}
func (m *mockS3Client) AbortMultipartUpload(context.Context, *aws_s3.AbortMultipartUploadInput, ...func(*aws_s3.Options)) (*aws_s3.AbortMultipartUploadOutput, error) {
	return nil, nil
}
func (m *mockS3Client) ListObjectsV2(context.Context, *aws_s3.ListObjectsV2Input, ...func(*aws_s3.Options)) (*aws_s3.ListObjectsV2Output, error) {
	return nil, nil
}

func TestInference(t *testing.T) {
	model := &regexModel{
		patterns: map[string]regexp.Regexp{
			"phone": *regexp.MustCompile(`\d{3}-\d{3}-\d{4}`),
			"email": *regexp.MustCompile(`\w+@email\.com`),
		},
	}

	jobId := uuid.New()

	inferenceJobProcessor := InferenceJobProcessor{
		s3client: s3.NewFromClient(&mockS3Client{}, "test-bucket"),
	}

	groupId1, groupId2 := uuid.New(), uuid.New()
	group1, err := ParseQuery(`COUNT(phone) > 0 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	group2, err := ParseQuery(`COUNT(phone) > 1 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	object := "test.txt"

	allEntities, groups, err := inferenceJobProcessor.processObject(
		jobId,
		NewDefaultParser(),
		model,
		map[uuid.UUID]Filter{groupId1: group1, groupId2: group2},
		"",
		object,
	)
	assert.NoError(t, err)

	phone, email := "012-345-6789", "test@email.com"
	phoneStart, emailStart := strings.Index(testDoc, phone), strings.Index(testDoc, email)

	assert.ElementsMatch(t, allEntities, []database.ObjectEntity{
		{InferenceJobId: jobId, Object: object, Label: "phone", Text: phone, Start: phoneStart, End: phoneStart + len(phone)},
		{InferenceJobId: jobId, Object: object, Label: "email", Text: email, Start: emailStart, End: emailStart + len(email)},
	})

	assert.ElementsMatch(t, groups, []database.ObjectGroup{
		{InferenceJobId: jobId, GroupId: groupId1, Object: object},
	})
}
