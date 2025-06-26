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
	"gorm.io/datatypes"
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
	storage.ObjectStore
}

func (m *mockStorage) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func TestListModels(t *testing.T) {
	id1, id2 := uuid.New(), uuid.New()
	fixedTime := time.Date(2025, 6, 11, 12, 0, 0, 0, time.UTC)
	db := createDB(t,
		&database.Model{Id: id1, Name: "Model1", Type: "regex", Status: database.ModelTrained, CreationTime: fixedTime},
		&database.Model{Id: id2, Name: "Model2", Type: "bolt_udt", Status: database.ModelTraining, CreationTime: fixedTime},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
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
		{Id: id1, Name: "Model1", Status: database.ModelTrained, CreationTime: fixedTime, Finetunable: false},
		{Id: id2, Name: "Model2", Status: database.ModelTraining, CreationTime: fixedTime, Finetunable: true},
	}, response)
}

func TestGetModel(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: uuid.New(), Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Model{Id: modelId, Name: "Model2", Type: "bolt_udt", Status: database.ModelTraining},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/models/"+modelId.String(), nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	var response api.Model
	err := json.Unmarshal(rec.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, api.Model{Id: modelId, Name: "Model2", Status: database.ModelTraining, Tags: []string{"name", "phone"}, Finetunable: true}, response)
}

func TestFinetuneModel(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	var response api.FinetuneResponse
	t.Run("Finetuning", func(t *testing.T) {
		tp := "Finetuning test"
		payload := api.FinetuneRequest{
			Name:       "FinetunedModel",
			TaskPrompt: &tp,
			Tags:       []api.TagInfo{{Name: "tag1"}},
			Samples: []api.Sample{
				{
					Tokens: []string{"example"},
					Labels: []string{"tag1"},
				},
			},
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

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	storageParams, _ := json.Marshal(storage.S3ConnectorParams{Region: "us-east-2", Bucket: "thirdai-corp-public", Prefix: "sample-pdfs/MACH.pdf"})

	payload := api.CreateReportRequest{
		ReportName:     "test-report",
		ModelId:        modelId,
		StorageType:     storage.S3ConnectorType,
		StorageParams:   json.RawMessage(storageParams),
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
		Status: database.ModelTrained,
	}, report.Model)
	assert.Equal(t, "s3", report.StorageType)
	assert.Equal(t, json.RawMessage(storageParams), report.StorageParams)
	assert.ElementsMatch(t, []string{"name", "phone"}, report.Tags)
	assert.Equal(t, map[string]string{"tag1": "pattern1", "tag2": "pattern2"}, report.CustomTags)
	assert.Equal(t, 2, len(report.Groups))
	assert.Equal(t, database.JobQueued, report.ShardDataTaskStatus)
}

func TestCreateReport_InvalidS3(t *testing.T) {
	modelId := uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "email"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	storageParams, _ := json.Marshal(storage.S3ConnectorParams{Bucket: "test-bucket", Prefix: "test-prefix"})

	payload := api.CreateReportRequest{
		ReportName:     "test-report",
		ModelId:        modelId,
		StorageType:     storage.S3ConnectorType,
		StorageParams:   json.RawMessage(storageParams),
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

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.Contains(t, rec.Body.String(), "failed to verify access to s3")
}

func TestGetReport(t *testing.T) {
	modelId, reportId, group1, group2 := uuid.New(), uuid.New(), uuid.New(), uuid.New()

	storageParams, _ := json.Marshal(storage.S3ConnectorParams{Bucket: "test-bucket", Prefix: "test-prefix"})

	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.ModelTag{ModelId: modelId, Tag: "name"},
		&database.ModelTag{ModelId: modelId, Tag: "email"},
		&database.ModelTag{ModelId: modelId, Tag: "phone"},
		&database.Report{
			Id:             reportId,
			ModelId:        modelId,
			StorageType:     storage.S3ConnectorType,
			StorageParams:   datatypes.JSON(storageParams),
			Groups: []database.Group{
				{Id: group1, Name: "group_a", ReportId: reportId, Query: `label1 CONTAINS "xyz"`},
				{Id: group2, Name: "group_b", ReportId: reportId, Query: `label1 = "xyz"`},
			},
		},
		&database.ReportTag{ReportId: reportId, Tag: "name"},
		&database.ReportTag{ReportId: reportId, Tag: "phone"},
		&database.CustomTag{ReportId: reportId, Tag: "tag1", Pattern: "pattern1"},
		&database.ShardDataTask{ReportId: reportId, Status: database.JobCompleted},
		&database.InferenceTask{ReportId: reportId, TaskId: 1, Status: database.JobCompleted, StorageParams: datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`))},
		&database.InferenceTask{ReportId: reportId, TaskId: 2, Status: database.JobRunning, StorageParams: datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`))},
		&database.InferenceTask{ReportId: reportId, TaskId: 3, Status: database.JobRunning, StorageParams: datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`))},
		&database.ObjectGroup{ReportId: reportId, GroupId: group1, Object: "object1"},
		&database.ObjectGroup{ReportId: reportId, GroupId: group2, Object: "object2"},
		&database.ObjectGroup{ReportId: reportId, GroupId: group1, Object: "object3"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
		&database.ObjectEntity{ReportId: reportId, Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
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
			Status: database.ModelTrained,
		}, report.Model)
		assert.Equal(t, "s3", report.StorageType)
		assert.Equal(t, json.RawMessage(storageParams), report.StorageParams)
		assert.ElementsMatch(t, []string{"name", "phone"}, report.Tags)
		assert.Equal(t, map[string]string{"tag1": "pattern1"}, report.CustomTags)
		assert.Equal(t, 2, len(report.Groups))
		assert.Equal(t, database.JobCompleted, report.ShardDataTaskStatus)
		assert.GreaterOrEqual(t, report.TotalInferenceTimeSeconds, 0.0)
		assert.GreaterOrEqual(t, report.ShardDataTimeSeconds, 0.0)
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

func TestDeleteReport(t *testing.T) {
	modelId, reportId := uuid.New(), uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Report{Id: reportId, ModelId: modelId, StorageType: storage.S3ConnectorType, StorageParams: datatypes.JSON(json.RawMessage(`{"Bucket": "test-bucket", "Prefix": "test-prefix"}`))},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	t.Run("DeleteReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodDelete, "/reports/"+reportId.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("GetDeletedReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports/"+reportId.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusNotFound, rec.Code)
	})

	t.Run("ListDeletedReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var reports []api.Report
		err := json.Unmarshal(rec.Body.Bytes(), &reports)
		assert.NoError(t, err)

		assert.Empty(t, reports)
	})
}

func TestStopReport(t *testing.T) {
	modelId, reportId := uuid.New(), uuid.New()
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Report{Id: reportId, ModelId: modelId, StorageType: storage.S3ConnectorType, StorageParams: datatypes.JSON(json.RawMessage(`{"Bucket": "test-bucket", "Prefix": "test-prefix"}`))},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	t.Run("StopReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/reports/"+reportId.String()+"/stop", nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("GetStoppedReport", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/reports/"+reportId.String(), nil)
		rec := httptest.NewRecorder()

		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var report api.Report
		err := json.Unmarshal(rec.Body.Bytes(), &report)
		assert.NoError(t, err)
		assert.Equal(t, report.Id, reportId)
		assert.True(t, report.Stopped)
	})
}

func TestReportSearch(t *testing.T) {
	modelId, reportId := uuid.New(), uuid.New()

	storageParams, _ := json.Marshal(storage.S3ConnectorParams{Bucket: "test-bucket", Prefix: "test-prefix"})
	
	db := createDB(t,
		&database.Model{Id: modelId, Name: "Model1", Type: "regex", Status: database.ModelTrained},
		&database.Report{
			Id:             reportId,
			ModelId:        modelId,
			StorageType:     storage.S3ConnectorType,
			StorageParams:   datatypes.JSON(storageParams),
		},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 1, End: 2, Label: "label1", Text: "text1"},
		&database.ObjectEntity{ReportId: reportId, Object: "object2", Start: 1, End: 1, Label: "label2", Text: "text2"},
		&database.ObjectEntity{ReportId: reportId, Object: "object3", Start: 2, End: 3, Label: "label3", Text: "abc"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 2, End: 3, Label: "label3", Text: "text3"},
		&database.ObjectEntity{ReportId: reportId, Object: "object1", Start: 4, End: 5, Label: "label4", Text: "12xyz34"},
		&database.ObjectEntity{ReportId: reportId, Object: "object3", Start: 4, End: 5, Label: "label4", Text: "12xyz34"},
		&database.ObjectEntity{ReportId: reportId, Object: "object4", Start: 4, End: 5, Label: "label3", Text: "12xyz34"},
	)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
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

func TestGetReportPreviews(t *testing.T) {
	reportId := uuid.New()
	payload := struct {
		Tokens []string `json:"tokens"`
		Tags   []string `json:"tags"`
	}{
		Tokens: []string{"foo", "bar", "baz"},
		Tags:   []string{"O", "TAG1", "O"},
	}
	b, err := json.Marshal(payload)
	require.NoError(t, err)

	p1 := &database.ObjectPreview{
		ReportId:  reportId,
		Object:    "doc1.txt",
		TokenTags: datatypes.JSON(b),
	}
	p2 := &database.ObjectPreview{
		ReportId:  reportId,
		Object:    "doc2.txt",
		TokenTags: datatypes.JSON(b),
	}
	p3 := &database.ObjectPreview{
		ReportId:  reportId,
		Object:    "doc3.txt",
		TokenTags: datatypes.JSON([]byte(`{"tokens":["foo"],"tags":["TAG2"]}`)),
	}

	e1 := &database.ObjectEntity{
		ReportId: reportId,
		Object:   "doc1.txt",
		Start:    0, End: 1,
		Label: "TAG1",
		Text:  "",
	}
	e2 := &database.ObjectEntity{
		ReportId: reportId,
		Object:   "doc2.txt",
		Start:    0, End: 1,
		Label: "TAG1",
		Text:  "",
	}
	e3 := &database.ObjectEntity{
		ReportId: reportId,
		Object:   "doc3.txt",
		Start:    0, End: 1,
		Label: "TAG2",
		Text:  "",
	}

	db := createDB(t, p1, p2, p3, e1, e2, e3)

	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	// Test: Get previews with tag filter "TAG1"
	t.Run("GetPreviewsWithTagFilter", func(t *testing.T) {
		url := fmt.Sprintf("/reports/%s/objects?limit=10&offset=0&tags=TAG1", reportId.String())
		req := httptest.NewRequest(http.MethodGet, url, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var resp []api.ObjectPreviewResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

		want := []api.ObjectPreviewResponse{
			{Object: "doc1.txt", Tokens: []string{"foo", "bar", "baz"}, Tags: []string{"O", "TAG1", "O"}},
			{Object: "doc2.txt", Tokens: []string{"foo", "bar", "baz"}, Tags: []string{"O", "TAG1", "O"}},
		}
		assert.ElementsMatch(t, want, resp)
	})

	// Test: Get previews with tag filter "TAG2"
	t.Run("GetPreviewsWithTag2Filter", func(t *testing.T) {
		url := fmt.Sprintf("/reports/%s/objects?limit=10&offset=0&tags=TAG2", reportId.String())
		req := httptest.NewRequest(http.MethodGet, url, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var resp []api.ObjectPreviewResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

		want := []api.ObjectPreviewResponse{
			{Object: "doc3.txt", Tokens: []string{"foo"}, Tags: []string{"TAG2"}},
		}
		assert.ElementsMatch(t, want, resp)
	})

	// Test: Get previews with no matching tags
	t.Run("GetPreviewsWithNoMatchingTags", func(t *testing.T) {
		url := fmt.Sprintf("/reports/%s/objects?limit=10&offset=0&tags=TAG3", reportId.String())
		req := httptest.NewRequest(http.MethodGet, url, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var resp []api.ObjectPreviewResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		assert.Len(t, resp, 0)
	})

	// Test: Get previews with object filtering
	t.Run("GetPreviewsWithObjectFilter", func(t *testing.T) {
		url := fmt.Sprintf("/reports/%s/objects?limit=10&offset=0&object=doc1.txt", reportId.String())
		req := httptest.NewRequest(http.MethodGet, url, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var resp []api.ObjectPreviewResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))

		want := []api.ObjectPreviewResponse{
			{Object: "doc1.txt", Tokens: []string{"foo", "bar", "baz"}, Tags: []string{"O", "TAG1", "O"}},
		}
		assert.ElementsMatch(t, want, resp)
	})
}

func TestGetInferenceMetrics_NoTasks(t *testing.T) {
	db := createDB(t)

	svc := backend.NewBackendService(db, &mockStorage{}, nil /*chunkTargetBytes*/, 0, nil)
	router := chi.NewRouter()
	svc.AddRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var m api.InferenceMetricsResponse
	err := json.NewDecoder(rec.Body).Decode(&m)
	require.NoError(t, err)

	assert.Equal(t, int64(0), m.Completed)
	assert.Equal(t, int64(0), m.InProgress)
	assert.Equal(t, float64(0), m.DataProcessedMB)
	assert.Equal(t, int64(0), m.TokensProcessed)
}

func TestGetInferenceMetrics_WithTasks(t *testing.T) {
	db := createDB(t)
	now := time.Now().UTC()

	// A completed task 1 hour ago, 2 MiB, 200 tokens
	db.Create(&database.InferenceTask{
		ReportId:       uuid.New(),
		TaskId:         1,
		Status:         database.JobCompleted,
		CreationTime:   now.Add(-1 * time.Hour),
		CompletionTime: sql.NullTime{Time: now.Add(-1 * time.Hour), Valid: true},
		TotalSize:      2 * 1024 * 1024,
		TokenCount:     200,
		StorageParams:  datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`)),
	})

	// A running task 30 min ago, 4 MiB, 300 tokens
	db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TaskId:       1,
		Status:       database.JobRunning,
		CreationTime: now.Add(-30 * time.Minute),
		TotalSize:    4 * 1024 * 1024,
		TokenCount:   300,
		StorageParams:  datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`)),
	})

	svc := backend.NewBackendService(db, &mockStorage{}, nil, 0, nil)
	router := chi.NewRouter()
	svc.AddRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var m api.InferenceMetricsResponse
	err := json.NewDecoder(rec.Body).Decode(&m)
	require.NoError(t, err)

	// We expect 1 completed, 1 in‐progress
	assert.Equal(t, int64(1), m.Completed)
	assert.Equal(t, int64(1), m.InProgress)
	// DataProcessedMB = (2 + 4) MiB = 6.0
	assert.InEpsilon(t, 6.0, m.DataProcessedMB, 1e-6)
	// TokensProcessed = 200 + 300 = 500
	assert.Equal(t, int64(500), m.TokensProcessed)
	t.Run("GetThroughputMetrics", func(t *testing.T) {
		modelID := uuid.New()
		reportID := uuid.New()
		require.NoError(t, db.Create(&database.Model{
			Id:     modelID,
			Name:   "M",
			Type:   "regex",
			Status: database.ModelTrained,
		}).Error)
		require.NoError(t, db.Create(&database.Report{
			Id:      reportID,
			ModelId: modelID,
			StorageType: storage.S3ConnectorType,
			StorageParams: datatypes.JSON(json.RawMessage(`{"Bucket": "test-bucket", "Prefix": "test-prefix"}`)),
		}).Error)

		now := time.Now().UTC()
		require.NoError(t, db.Create(&database.InferenceTask{
			ReportId:       reportID,
			TaskId:         42,
			Status:         database.JobCompleted,
			CreationTime:   now.Add(-1 * time.Hour),
			StartTime:      sql.NullTime{Time: now.Add(-1 * time.Hour), Valid: true},
			CompletionTime: sql.NullTime{Time: now, Valid: true},
			TotalSize:      1 * 1024 * 1024,
			TokenCount:     0,
			StorageParams:  datatypes.JSON(json.RawMessage(`{"ChunkKeys": ["test-chunk-key"]}`)),
		}).Error)

		url := fmt.Sprintf(
			"/metrics/throughput?model_id=%s&report_id=%s",
			modelID.String(), reportID.String(),
		)
		req := httptest.NewRequest(http.MethodGet, url, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		assert.Equal(t, http.StatusOK, rec.Code)
		var resp api.ThroughputResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		assert.Equal(t, modelID, resp.ModelID)
		assert.Equal(t, reportID, resp.ReportID)
		assert.InEpsilon(t, 1.0, resp.ThroughputMBPerHour, 1e-6)
	})
}

func TestValidateGroupDefinition_ValidDefinition(t *testing.T) {
	service := backend.NewBackendService(nil, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	url := "/validate/group?GroupQuery=" + url.QueryEscape("COUNT(label1) > 0")
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestValidateGroupDefinition_InvalidDefinition(t *testing.T) {
	service := backend.NewBackendService(nil, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	url := "/validate/group?GroupQuery=FAKEQUERY"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusUnprocessableEntity, rec.Code)
	assert.Contains(t, rec.Body.String(), "invalid query")
}

func TestValidateS3Bucket_PublicBucket(t *testing.T) {
	service := backend.NewBackendService(nil, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	url := "/validate/s3?S3Endpoint=&Region=us-east-2&SourceS3Bucket=thirdai-corp-public&SourceS3Prefix="
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestValidateS3Bucket_InvalidBucket(t *testing.T) {
	service := backend.NewBackendService(nil, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	url := "/validate/s3?S3Endpoint=&Region=us-east-1&SourceS3Bucket=test-bucket&SourceS3Prefix="
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.Contains(t, rec.Body.String(), "failed to verify access to s3")
}

func TestStoreAndGetFileNameToPath(t *testing.T) {
	db := createDB(t)
	service := backend.NewBackendService(db, &mockStorage{}, messaging.NewInMemoryQueue(), 1024, nil)
	router := chi.NewRouter()
	service.AddRoutes(router)

	// First test storing
	uploadId := uuid.New()
	testMap := map[string]string{
		"file1.txt": "path/to/file1.txt",
		"file2.txt": "path/to/file2.txt",
	}

	url := fmt.Sprintf("/file-name-to-path/%s", uploadId.String())
	body, err := json.Marshal(api.FileNameToPath{Mapping: testMap})
	assert.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Then test getting
	req = httptest.NewRequest(http.MethodGet, url, nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var resp api.FileNameToPath
	err = json.Unmarshal(rec.Body.Bytes(), &resp)
	assert.NoError(t, err)
	assert.Equal(t, testMap, resp.Mapping)
}
