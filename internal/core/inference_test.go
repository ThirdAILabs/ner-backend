package core

import (
	"context"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/s3"
	"ner-backend/pkg/api"
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
					Label:    label,
					Text:     text[match[0]:match[1]],
					Start:    match[0],
					End:      match[1],
					LContext: text[max(0, match[0]-20):match[0]],
					RContext: text[match[1]:min(len(text), match[1]+20)],
				})
			}
		}
	}
	return entities, nil
}

func (m *regexModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return nil
}

func (m *regexModel) Save(path string) error {
	return nil
}

func (m *regexModel) Release() {}

type mockS3Client struct {
	s3.S3Api
}

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

func TestInference(t *testing.T) {
	model := &regexModel{
		patterns: map[string]regexp.Regexp{
			"phone": *regexp.MustCompile(`\d{3}-\d{3}-\d{4}`),
			"email": *regexp.MustCompile(`\w+@email\.com`),
		},
	}

	reportId := uuid.New()

	inferenceJobProcessor := TaskProcessor{
		s3Client: s3.NewFromClient(&mockS3Client{}, "test-bucket"),
	}

	groupId1, groupId2 := uuid.New(), uuid.New()
	group1, err := ParseQuery(`COUNT(phone) > 0 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	group2, err := ParseQuery(`COUNT(phone) > 1 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	object := "test.txt"

	allEntities, groups, tagCount, err := inferenceJobProcessor.runInferenceOnObject(
		reportId,
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
		{ReportId: reportId, Object: object, Label: "phone", Text: phone, Start: phoneStart, End: phoneStart + len(phone), LContext: testDoc[phoneStart-20 : phoneStart], RContext: testDoc[phoneStart+len(phone) : phoneStart+len(phone)+20]},
		{ReportId: reportId, Object: object, Label: "email", Text: email, Start: emailStart, End: emailStart + len(email), LContext: testDoc[emailStart-20 : emailStart], RContext: testDoc[emailStart+len(email):]},
	})

	assert.Equal(t, tagCount, api.TagCount{
		"phone": 1,
		"email": 1,
	})

	assert.ElementsMatch(t, groups, []database.ObjectGroup{
		{ReportId: reportId, GroupId: groupId1, Object: object},
	})
}
