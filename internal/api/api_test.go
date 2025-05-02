package api_test

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	backend "ner-backend/internal/api"
	"ner-backend/internal/database"
	"ner-backend/internal/messaging"
	"ner-backend/internal/storage"
	"ner-backend/pkg/api"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func createDB(t *testing.T, create ...any) *gorm.DB {
	db, err := gorm.Open(sqlite.Open("file::memory:"), &gorm.Config{})
	require.NoError(t, err)

	require.NoError(t, database.GetMigrator(db).Migrate())

	for _, c := range create {
		require.NoError(t, db.Create(c).Error)
	}

	return db
}

type mockStorage struct {
	storage.Provider
}

func (m *mockStorage) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func TestListModels(t *testing.T) {
	id1, id2 := uuid.New(), uuid.New()
	db := createDB(t,
		&database.Model{Id: id1, Name: "Model1", Type: "regex", Status: database.ModelTrained, CreationTime: time.Now()},
		&database.Model{Id: id2, Name: "Model2", Type: "bolt", Status: database.ModelTraining, CreationTime: time.Now()},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/models", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	var response []api.Model
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.ElementsMatch(t, []api.Model{
		{Id: id1, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		{Id: id2, Name: "Model2", Type: "bolt", Status: database.ModelTraining},
	}, response)
}

func TestGetModel(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: uuid.New(), Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Model{Id: modelId, Name: "Model2", Type: "bolt", Status: database.ModelTraining},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/models/"+modelId.String(), nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	var response api.Model
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, api.Model{Id: modelId, Name: "Model2", Type: "bolt", Status: database.ModelTraining, Tags: []string{"name", "phone"}}, response)
}

func TestFinetuneModel(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	var response api.FinetuneResponse
	t.Run("Finetuning", func(t *testing.T) {
		payload := api.FinetuneRequest{
			Name:       "FinetunedModel",
			TaskPrompt: "Finetuning test",
			Tags:       []api.TagInfo{{Name: "tag1"}},
		}
		body, err := json.Marshal(payload)
		assert.NoError(t, err)

		req := httptest.NewRequest(http.MethodPost, "/models/"+modelId.String()+"/finetune", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code, "recieved response: "+rec.Body.String())
		err = json.Unmarshal(rec.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.NotEqual(t, uuid.Nil, response.ModelId)
	})

	t.Run("GetFinetunedModel", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/models/"+response.ModelId.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var model api.Model
		err := json.Unmarshal(rec.Body.Bytes(), &model)
		assert.NoError(t, err)

		assert.Equal(t, response.ModelId, model.Id)
		assert.Equal(t, "FinetunedModel", model.Name)
		assert.Equal(t, "regex", model.Type)
		assert.Equal(t, database.ModelQueued, model.Status)
		assert.Equal(t, modelId, *model.BaseModelId)
	})
}

func TestCreateReport(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "email"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	payload := api.CreateReportRequest{
		ModelId:        modelId,
		SourceS3Bucket: "test-bucket",
		SourceS3Prefix: "test-prefix",
		Tags:           []string{"name", "phone"},
		CustomTags:     map[string]string{"tag1": "pattern1", "tag2": "pattern2"},
		Groups: map[string]string{
			"group1": `label1 CONTAINS "xyz"`,
			"group2": `COUNT(label2) > 8`,
		},
	}
	body, err := json.Marshal(payload)
	assert.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/reports", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code, "recieved response: "+rec.Body.String())
	var response api.CreateReportResponse
	err = json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.NotEqual(t, uuid.Nil, response.ReportId)

	req = httptest.NewRequest(http.MethodGet, "/reports/"+response.ReportId.String(), nil)
	rec = httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	var report api.Report
	err = json.Unmarshal(rec.Body.Bytes(), &report)
	assert.NoError(t, err)

	assert.Equal(t, report.Id, response.ReportId)
	assert.Equal(t, api.Model{
		Id:     modelId,
		Name:   "Model1",
		Type:   "regex",
		Status: database.ModelTrained,
	}, report.Model)
	assert.Equal(t, "test-bucket", report.SourceS3Bucket)
	assert.Equal(t, "test-prefix", report.SourceS3Prefix)
	assert.ElementsMatch(t, []string{"name", "phone"}, report.Tags)
	assert.Equal(t, map[string]string{"tag1": "pattern1", "tag2": "pattern2"}, report.CustomTags)
	assert.Equal(t, 2, len(report.Groups))
	assert.Equal(t, database.JobQueued, report.ShardDataTaskStatus)
}

func TestGetReport(t *testing.T) {
	modelId, reportId, group1, group2 := uuid.New(), uuid.New(), uuid.New(), uuid.New()

	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "email"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
		&database.Report{
			Id:             reportId,
			ModelId:        modelId,
			SourceS3Bucket: "test-bucket",
			SourceS3Prefix: sql.NullString{String: "test-prefix", Valid: true},
			Groups: []database.Group{
				{Id: group1, Name: "group_a", ReportId: reportId, Query: `label1 CONTAINS "xyz"`},
				{Id: group2, Name: "group_b", ReportId: reportId, Query: `label1 = "xyz"`},
			},
		},
		&database.ReportTag{ReportId: reportId, Tag: "name"},
		&database.ReportTag{ReportId: reportId, Tag: "phone"},
		&database.CustomTag{ReportId: reportId, Tag: "tag1", Pattern: "pattern1"},
		&database.ShardDataTask{ReportId: reportId, Status: database.JobCompleted},
		&database.InferenceTask{ReportId: reportId, TaskId: 1, Status: database.JobCompleted},
		&database.InferenceTask{ReportId: reportId, TaskId: 2, Status: database.JobRunning},
		&database.InferenceTask{ReportId: reportId, TaskId: 3, Status: database.JobRunning},
		&database.ObjectGroup{ReportId: reportId, GroupId: group1, Object: "object1"},
		&database.ObjectGroup{ReportId: reportId, GroupId: group2, Object: "object2"},
		&database.ObjectGroup{ReportId: reportId, GroupId: group1, Object: "object3"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
		&database.ObjectEntity{ReportId: reportId, Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	t.Run("GetReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports/"+reportId.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var report api.Report
		err := json.Unmarshal(rec.Body.Bytes(), &report)
		assert.NoError(t, err)

		assert.Equal(t, report.Id, reportId)
		assert.Equal(t, api.Model{
			Id:     modelId,
			Name:   "Model1",
			Type:   "regex",
			Status: database.ModelTrained,
		}, report.Model)
		assert.Equal(t, "test-bucket", report.SourceS3Bucket)
		assert.Equal(t, "test-prefix", report.SourceS3Prefix)
		assert.ElementsMatch(t, []string{"name", "phone"}, report.Tags)
		assert.Equal(t, map[string]string{"tag1": "pattern1"}, report.CustomTags)
		assert.Equal(t, 2, len(report.Groups))
		assert.Equal(t, database.JobCompleted, report.ShardDataTaskStatus)
	})

	t.Run("GetReportGroups", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports/"+reportId.String()+"/groups/"+group1.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var group api.Group
		err := json.Unmarshal(rec.Body.Bytes(), &group)
		assert.NoError(t, err)

		assert.Equal(t, group1, group.Id)
		assert.Equal(t, "group_a", group.Name)
		assert.Equal(t, `label1 CONTAINS "xyz"`, group.Query)
		assert.ElementsMatch(t, []string{"object1", "object3"}, group.Objects)
	})

	t.Run("GetReportEntities", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports/"+reportId.String()+"/entities", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var entities []api.Entity
		err := json.Unmarshal(rec.Body.Bytes(), &entities)
		assert.NoError(t, err)

		assert.ElementsMatch(t, []api.Entity{
			{Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
			{Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
			{Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
		}, entities)
	})

	t.Run("GetReportEntitiesPaged", func(t *testing.T) {
		var entities []api.Entity

		for {
			url := fmt.Sprintf("/reports/%s/entities?limit=2&offset=%d", reportId.String(), len(entities))
			req := httptest.NewRequest(http.MethodGet, url, nil)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			assert.Equal(t, http.StatusOK, rec.Code)
			var responseEntities []api.Entity
			err := json.Unmarshal(rec.Body.Bytes(), &responseEntities)
			assert.NoError(t, err)

			assert.GreaterOrEqual(t, 2, len(responseEntities))
			entities = append(entities, responseEntities...)

			if len(responseEntities) == 0 {
				break
			}
		}

		assert.ElementsMatch(t, []api.Entity{
			{Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
			{Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
			{Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
		}, entities)

	})
}

func TestReportSearch(t *testing.T) {
	modelId, reportId := uuid.New(), uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Report{
			Id:             reportId,
			ModelId:        modelId,
			SourceS3Bucket: "test-bucket",
			SourceS3Prefix: sql.NullString{String: "test-prefix", Valid: true},
		},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
		&database.ObjectEntity{ReportId: reportId, Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
		&database.ObjectEntity{ReportId: reportId, Object: "object3", Start: 2, End: 3, Label: "label3", Text: "abc"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 4, End: 5, Label: "label4", Text: "12xyz34"},
		&database.ObjectEntity{ReportId: reportId, Object: "object3", Start: 4, End: 5, Label: "label4", Text: "12xyz34"},
		&database.ObjectEntity{ReportId: reportId, Object: "object4", Start: 4, End: 5, Label: "label3", Text: "12xyz34"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024)
	router := chi.NewRouter()
	service.AddRoutes(router)

	query := `label4 CONTAINS "xyz" AND (COUNT(label2) > 0 OR label3 = "abc")`
	url := fmt.Sprintf("/reports/%s/search?query=%s", reportId.String(), url.QueryEscape(query))

	fmt.Println(url)
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	var response api.SearchResponse
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)

	assert.ElementsMatch(t, []string{"object1", "object3"}, response.Objects)
}
