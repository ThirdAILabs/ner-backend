package api

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"ner-backend/internal/core"
	"ner-backend/internal/database"
	pkgapi "ner-backend/pkg/api"

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

	if err := database.GetMigrator(db).Migrate(); err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
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

	// Test setting API key
	setKeyPayload := pkgapi.ApiKey{ApiKey: apiKey}
	setKeyBody, _ := json.Marshal(setKeyPayload)
	req := httptest.NewRequest(http.MethodPost, "/chat/api-key", bytes.NewReader(setKeyBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Test getting API key
	req = httptest.NewRequest(http.MethodGet, "/chat/api-key", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var getKeyResp pkgapi.ApiKey
	if err := json.NewDecoder(rec.Body).Decode(&getKeyResp); err != nil {
		t.Fatalf("decode get-api-key response: %v", err)
	}
	assert.Equal(t, apiKey, getKeyResp.ApiKey)

	// Test Chat Session

	startPayload := pkgapi.StartSessionRequest{Model: "gpt-3", Title: "Test Session"}
	startBody, _ := json.Marshal(startPayload)
	req = httptest.NewRequest(http.MethodPost, "/chat/sessions", bytes.NewReader(startBody))
	req.Header.Set("Content-Type", "application/json")

	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var startResp pkgapi.StartSessionResponse
	if err := json.NewDecoder(rec.Body).Decode(&startResp); err != nil {
		t.Fatalf("decode start-session response: %v", err)
	}
	sessionID := startResp.SessionID

	// Get all sessions and verify our new session is there
	req = httptest.NewRequest(http.MethodGet, "/chat/sessions", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var sessionsResp pkgapi.GetSessionsResponse
	if err := json.NewDecoder(rec.Body).Decode(&sessionsResp); err != nil {
		t.Fatalf("decode get-sessions response: %v", err)
	}
	assert.Equal(t, 1, len(sessionsResp.Sessions))
	assert.Equal(t, "Test Session", sessionsResp.Sessions[0].Title)

	// Get specific session
	req = httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID, nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var sessionResp pkgapi.ChatSessionMetadata
	if err := json.NewDecoder(rec.Body).Decode(&sessionResp); err != nil {
		t.Fatalf("decode get-session response: %v", err)
	}
	assert.Equal(t, "Test Session", sessionResp.Title)

	// Rename the session
	renamePayload := pkgapi.RenameSessionRequest{Title: "Renamed Test Session"}
	renameBody, _ := json.Marshal(renamePayload)
	req = httptest.NewRequest(http.MethodPost, "/chat/sessions/"+sessionID+"/rename", bytes.NewReader(renameBody))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Verify rename worked
	req = httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID, nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	if err := json.NewDecoder(rec.Body).Decode(&sessionResp); err != nil {
		t.Fatalf("decode get-session response: %v", err)
	}
	assert.Equal(t, "Renamed Test Session", sessionResp.Title)

	// Send a message to the session
	chatPayload := pkgapi.ChatRequest{
		Model:   "gpt-3",
		APIKey:  apiKey,
		Message: "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is yash@thirdai.com",
	}
	chatBody, _ := json.Marshal(chatPayload)
	req = httptest.NewRequest(http.MethodPost, "/chat/sessions/"+sessionID+"/messages/stream", bytes.NewReader(chatBody))
	req.Header.Set("Content-Type", "application/json")

	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Read the streaming response
	reader := bufio.NewReader(rec.Body)
	var chatResp pkgapi.ChatResponse
	var tagMap map[string]string
	var inputText string
	var replyBuilder string

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("read stream: %v", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if err := json.Unmarshal([]byte(line), &chatResp); err != nil {
			t.Fatalf("decode chat response: %v", err)
		}

		if chatResp.TagMap != nil {
			tagMap = chatResp.TagMap
		}
		if chatResp.InputText != "" {
			inputText = chatResp.InputText
		}
		if chatResp.Reply != "" {
			replyBuilder += chatResp.Reply
		}
	}

	expectedRedacted := "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is [EMAIL_1]"

	assert.Equal(t, expectedRedacted, inputText)

	assert.Equal(t,
		map[string]string{"[EMAIL_1]": "yash@thirdai.com"},
		tagMap,
	)

	assert.NotEmpty(t, replyBuilder)

	// Get history
	req = httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID+"/history", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var history []pkgapi.ChatHistoryItem
	if err := json.NewDecoder(rec.Body).Decode(&history); err != nil {
		t.Fatalf("decode history: %v", err)
	}

	if len(history) != 2 {
		t.Fatalf("expected 2 history items, got %d", len(history))
	}

	userItem := history[0]
	assert.Equal(t, "user", userItem.MessageType)
	assert.Equal(t, expectedRedacted, userItem.Content)

	req = httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID, nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Get the session again to check that the tag map has been updated
	if err := json.NewDecoder(rec.Body).Decode(&sessionResp); err != nil {
		t.Fatalf("decode get-session response: %v", err)
	}
	assert.Equal(t, "yash@thirdai.com", sessionResp.TagMap["[EMAIL_1]"])
	
	aiItem := history[1]
	assert.Equal(t, "ai", aiItem.MessageType)
	assert.Equal(t, replyBuilder, aiItem.Content)

	// Clean up
	req = httptest.NewRequest(http.MethodDelete, "/chat/api-key", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Test that the api key is deleted
	req = httptest.NewRequest(http.MethodGet, "/chat/api-key", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	if err := json.NewDecoder(rec.Body).Decode(&getKeyResp); err != nil {
		t.Fatalf("decode get-api-key response: %v", err)
	}
	assert.Equal(t, "", getKeyResp.ApiKey)
}
