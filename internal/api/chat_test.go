package api

import (
	"bytes"
	"encoding/json"
	"log"
	"ner-backend/internal/core"
	"ner-backend/pkg/api"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

var router chi.Router

func init() {
	db, err := gorm.Open(sqlite.Open("file::memory:"), &gorm.Config{})
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	// presidio model doesn't require python path or plugin path
	loaders := core.NewModelLoaders("dummy python path executable ", "dummy python plugin path executable")
	nerModel, err := loaders["presidio"]("")
	if err != nil {
		log.Fatalf("could not load NER model: %v", err)
	}
	chatService := NewChatService(db, nerModel)
	router = chi.NewRouter()
	chatService.AddRoutes(router)
}

func TestChatEndpoint(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY must be set for TestChatEndpoint")
	}
	// Create a new session first
	startPayload := api.StartSessionRequest{
		Model: "gpt-3",
	}
	startPayloadBytes, err := json.Marshal(startPayload)
	if err != nil {
		t.Fatalf("Failed to marshal start session payload: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/chat/sessions", bytes.NewReader(startPayloadBytes))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var startSessionResponse api.StartSessionResponse
	if err := json.NewDecoder(rec.Body).Decode(&startSessionResponse); err != nil {
		t.Fatalf("Failed to decode start-session response: %v", err)
	}
	sessionID := startSessionResponse.SessionID

	reqBody := api.ChatRequest{
		Model:   "gpt-3",
		APIKey:  apiKey,
		Message: "Hello, I am Gautam sharma, How are you?",
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("Failed to marshal request body: %v", err)
	}
	req = httptest.NewRequest(http.MethodPost, "/chat/sessions/"+sessionID+"/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)

	var chatResponse api.ChatResponse
	if err := json.NewDecoder(rec.Body).Decode(&chatResponse); err != nil {
		t.Fatalf("Failed to decode chat response: %v", err)
	}
	assert.Equal(t, chatResponse.InputText, "Hello, I am [NAME_1], Ask me how I am doing by my mentioned name")
	assert.Equal(t, chatResponse.TagMap, map[string]string{
		"[NAME_1]": "Gautam sharma",
	})

	// Test GetHistory endpoint
	req = httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID+"/history", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
	var historyResponse []api.ChatHistoryItem
	if err := json.NewDecoder(rec.Body).Decode(&historyResponse); err != nil {
		t.Fatalf("Failed to decode history: %v", err)
	}
	assert.Equal(t, len(historyResponse), 2)
	assert.Equal(t, historyResponse[0].MessageType, "user")
	assert.Equal(t, historyResponse[0].Content, "Hello, I am [NAME_1], Ask me how I am doing by my mentioned name")
	assert.Equal(t, historyResponse[0].Metadata, map[string]string{
		"[NAME_1]": "Gautam sharma",
	})
	assert.Equal(t, historyResponse[1].MessageType, "ai")
	assert.Contains(t, historyResponse[1].Content, "[NAME_1]")
	assert.Equal(t, historyResponse[1].Metadata, map[string]string{
		"[NAME_1]": "Gautam sharma",
	})
}
