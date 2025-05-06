package core

import (
	"io"
	"ner-backend/internal/core/types"
	"ner-backend/internal/database"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

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

type mockStorage struct {
	storage.Provider
}

const testDoc = "This is a test doc. It contains a phone number: 012-345-6789, an email: test@email.com, and a special token a1b2c3."

func (m *mockStorage) GetObjectStream(bucket, key string) (io.Reader, error) {
	return strings.NewReader(testDoc), nil
}

func TestInference(t *testing.T) {
	model := &regexModel{
		patterns: map[string]regexp.Regexp{
			"phone": *regexp.MustCompile(`\d{3}-\d{3}-\d{4}`),
			"email": *regexp.MustCompile(`\w+@email\.com`),
			"test":  *regexp.MustCompile(`test`),
		},
	}

	reportId := uuid.New()

	inferenceJobProcessor := TaskProcessor{}

	groupId1, groupId2 := uuid.New(), uuid.New()
	group1, err := ParseQuery(`COUNT(phone) > 0 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	group2, err := ParseQuery(`COUNT(phone) > 1 AND email CONTAINS "test"`)
	assert.NoError(t, err)

	object := "test.txt"

	allEntities, groups, err := inferenceJobProcessor.runInferenceOnObject(
		reportId,
		&mockStorage{},
		NewDefaultParser(),
		model,
		map[string]struct{}{"phone": {}, "email": {}},
		map[string]*regexp.Regexp{"special_token": regexp.MustCompile(`(\w\d){3}`)},
		map[uuid.UUID]Filter{groupId1: group1, groupId2: group2},
		"",
		object,
	)
	assert.NoError(t, err)

	phone, email, special := "012-345-6789", "test@email.com", "a1b2c3"
	phoneStart, emailStart, specialStart := strings.Index(testDoc, phone), strings.Index(testDoc, email), strings.Index(testDoc, special)

	assert.ElementsMatch(t, allEntities, []database.ObjectEntity{
		{ReportId: reportId, Object: object, Label: "phone", Text: phone, Start: phoneStart, End: phoneStart + len(phone), LContext: testDoc[phoneStart-20 : phoneStart], RContext: testDoc[phoneStart+len(phone) : phoneStart+len(phone)+20]},
		{ReportId: reportId, Object: object, Label: "email", Text: email, Start: emailStart, End: emailStart + len(email), LContext: testDoc[emailStart-20 : emailStart], RContext: testDoc[emailStart+len(email) : emailStart+len(email)+20]},
		{ReportId: reportId, Object: object, Label: "special_token", Text: special, Start: specialStart, End: specialStart + len(special), LContext: testDoc[specialStart-20 : specialStart], RContext: testDoc[specialStart+len(special):]},
	})

	assert.ElementsMatch(t, groups, []database.ObjectGroup{
		{ReportId: reportId, GroupId: groupId1, Object: object},
	})
}
