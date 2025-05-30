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

func initializeChatService() chi.Router {
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
	router := chi.NewRouter()
	chatService.AddRoutes(router)

	return router
}

func setOpenAIAPIKey(t *testing.T, router chi.Router, apiKey string) {
	setKeyPayload := pkgapi.ApiKey{ApiKey: apiKey}
	setKeyBody, _ := json.Marshal(setKeyPayload)
	req := httptest.NewRequest(http.MethodPost, "/chat/api-key", bytes.NewReader(setKeyBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func getOpenAIAPIKey(t *testing.T, router chi.Router) string {
	req := httptest.NewRequest(http.MethodGet, "/chat/api-key", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var getKeyResp pkgapi.ApiKey
	if err := json.NewDecoder(rec.Body).Decode(&getKeyResp); err != nil {
		t.Fatalf("decode get-api-key response: %v", err)
	}

	return getKeyResp.ApiKey
}

func deleteOpenAIAPIKey(t *testing.T, router chi.Router) {
	req := httptest.NewRequest(http.MethodDelete, "/chat/api-key", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
}

func startSession(t *testing.T, router chi.Router, title string) string {
	startPayload := pkgapi.StartSessionRequest{Model: "gpt-3", Title: title}
	startBody, _ := json.Marshal(startPayload)
	req := httptest.NewRequest(http.MethodPost, "/chat/sessions", bytes.NewReader(startBody))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var startResp pkgapi.StartSessionResponse
	if err := json.NewDecoder(rec.Body).Decode(&startResp); err != nil {
		t.Fatalf("decode start-session response: %v", err)
	}
	sessionID := startResp.SessionID

	return sessionID
}

func getSessions(t *testing.T, router chi.Router) []pkgapi.ChatSessionMetadata {
	req := httptest.NewRequest(http.MethodGet, "/chat/sessions", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var sessionsResp pkgapi.GetSessionsResponse
	if err := json.NewDecoder(rec.Body).Decode(&sessionsResp); err != nil {
		t.Fatalf("decode get-sessions response: %v", err)
	}

	return sessionsResp.Sessions
}

func getSession(t *testing.T, router chi.Router, sessionID string) pkgapi.ChatSessionMetadata {
	req := httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var sessionResp pkgapi.ChatSessionMetadata
	if err := json.NewDecoder(rec.Body).Decode(&sessionResp); err != nil {
		t.Fatalf("decode get-session response: %v", err)
	}

	return sessionResp
}

func deleteSession(t *testing.T, router chi.Router, sessionID string) {
	req := httptest.NewRequest(http.MethodDelete, "/chat/sessions/"+sessionID, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func renameSession(t *testing.T, router chi.Router, sessionID string, title string) {
	renamePayload := pkgapi.RenameSessionRequest{Title: title}
	renameBody, _ := json.Marshal(renamePayload)
	req := httptest.NewRequest(http.MethodPost, "/chat/sessions/"+sessionID+"/rename", bytes.NewReader(renameBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)
}

func sendMessage(t *testing.T, router chi.Router, sessionID string, message string, apiKey string) (redactedMessage string, reply string, tagMap map[string]string) {
	chatPayload := pkgapi.ChatRequest{
		Model:   "gpt-3",
		Message: message,
	}
	chatBody, _ := json.Marshal(chatPayload)
	req := httptest.NewRequest(http.MethodPost, "/chat/sessions/"+sessionID+"/messages", bytes.NewReader(chatBody))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	// Read the streaming response
	reader := bufio.NewReader(rec.Body)
	var streamResp StreamMessage

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

		// Parse response.data as a ChatResponse
		var chatResp pkgapi.ChatResponse
		if err := json.Unmarshal([]byte(line), &streamResp); err != nil {
			t.Fatalf("decode chat response: %v", err)
		}

		if streamResp.Error != "" {
			t.Fatalf("stream response error: %s", streamResp.Error)
		}

		if streamResp.Data == nil {
			t.Fatalf("stream response data is nil")
		}

		jsonData, err := json.Marshal(streamResp.Data)
		if err != nil {
			t.Fatalf("marshal stream data: %v", err)
		}

		if err := json.Unmarshal(jsonData, &chatResp); err != nil {
			t.Fatalf("unmarshal chat response: %v", err)
		}

		if chatResp.TagMap != nil {
			tagMap = chatResp.TagMap
		}
		if chatResp.InputText != "" {
			redactedMessage = chatResp.InputText
		}
		if chatResp.Reply != "" {
			reply += chatResp.Reply
		}
	}

	return redactedMessage, reply, tagMap
}

func getHistory(t *testing.T, router chi.Router, sessionID string) []pkgapi.ChatHistoryItem {
	req := httptest.NewRequest(http.MethodGet, "/chat/sessions/"+sessionID+"/history", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	assert.Equal(t, http.StatusOK, rec.Code)

	var history []pkgapi.ChatHistoryItem
	if err := json.NewDecoder(rec.Body).Decode(&history); err != nil {
		t.Fatalf("decode history: %v", err)
	}

	return history
}

func TestChatEndpoint(t *testing.T) {

	router := initializeChatService()

	t.Run("TestGetOpenAIAPIKey_KeyUnset", func(t *testing.T) {
		apiKey := getOpenAIAPIKey(t, router)
		assert.Equal(t, "", apiKey)
	})

	t.Run("TestDeleteOpenAIAPIKey", func(t *testing.T) {
		apiKey := os.Getenv("OPENAI_API_KEY")

		setOpenAIAPIKey(t, router, apiKey)

		apiKey = getOpenAIAPIKey(t, router)
		assert.Equal(t, apiKey, apiKey)

		deleteOpenAIAPIKey(t, router)

		apiKey = getOpenAIAPIKey(t, router)
		assert.Equal(t, "", apiKey)
	})

	t.Run("TestGetOpenAIAPIKey_KeySet", func(t *testing.T) {
		apiKey := os.Getenv("OPENAI_API_KEY")

		setOpenAIAPIKey(t, router, apiKey)
		defer deleteOpenAIAPIKey(t, router)

		apiKey = getOpenAIAPIKey(t, router)
		assert.Equal(t, apiKey, apiKey)
	})

	t.Run("TestGetSessions", func(t *testing.T) {
		sessionID1 := startSession(t, router, "Test Session 1")
		defer deleteSession(t, router, sessionID1)
		assert.NotEmpty(t, sessionID1)

		sessionID2 := startSession(t, router, "Test Session 2")
		defer deleteSession(t, router, sessionID2)
		assert.NotEmpty(t, sessionID2)

		sessions := getSessions(t, router)
		assert.Equal(t, 2, len(sessions))
		assert.Equal(t, sessionID1, sessions[0].ID.String())
		assert.Equal(t, "Test Session 1", sessions[0].Title)
		assert.Equal(t, sessionID2, sessions[1].ID.String())
		assert.Equal(t, "Test Session 2", sessions[1].Title)
	})

	t.Run("TestDeleteSession", func(t *testing.T) {
		sessionID := startSession(t, router, "Test Session")
		deleteSession(t, router, sessionID)

		sessions := getSessions(t, router)
		assert.Equal(t, 0, len(sessions))
	})

	t.Run("TestStartSession", func(t *testing.T) {
		sessionID := startSession(t, router, "Test Session")
		defer deleteSession(t, router, sessionID)
		assert.NotEmpty(t, sessionID)

		session := getSession(t, router, sessionID)
		assert.Equal(t, "Test Session", session.Title)
	})

	t.Run("TestRenameSession", func(t *testing.T) {
		sessionID := startSession(t, router, "Test Session")
		defer deleteSession(t, router, sessionID)

		session := getSession(t, router, sessionID)
		assert.Equal(t, "Test Session", session.Title)

		renameSession(t, router, sessionID, "Renamed Test Session")

		session = getSession(t, router, sessionID)
		assert.Equal(t, "Renamed Test Session", session.Title)
	})

	t.Run("TestSendMessage", func(t *testing.T) {
		apiKey := os.Getenv("OPENAI_API_KEY")

		setOpenAIAPIKey(t, router, apiKey)
		defer deleteOpenAIAPIKey(t, router)

		sessionID := startSession(t, router, "Test Session")
		defer deleteSession(t, router, sessionID)
		assert.NotEmpty(t, sessionID)

		message := "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is yash@thirdai.com"
		redactedMessage, reply, tagMap := sendMessage(t, router, sessionID, message, apiKey)

		expectedRedacted := "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is [EMAIL_1]"
		assert.Equal(t, expectedRedacted, redactedMessage)
		assert.NotEmpty(t, reply)
		assert.Equal(t,
			map[string]string{"[EMAIL_1]": "yash@thirdai.com"},
			tagMap,
		)
	})

	t.Run("TestSendMessage_UpdatesHistory", func(t *testing.T) {
		apiKey := os.Getenv("OPENAI_API_KEY")

		setOpenAIAPIKey(t, router, apiKey)
		defer deleteOpenAIAPIKey(t, router)

		sessionID := startSession(t, router, "Test Session")
		defer deleteSession(t, router, sessionID)
		assert.NotEmpty(t, sessionID)

		message := "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is yash@thirdai.com"
		redactedMessage, reply, _ := sendMessage(t, router, sessionID, message, apiKey)

		history := getHistory(t, router, sessionID)
		assert.Equal(t, 2, len(history))
		assert.Equal(t, "user", history[0].MessageType)
		assert.Equal(t, redactedMessage, history[0].Content)
		assert.Equal(t, "ai", history[1].MessageType)
		assert.Equal(t, reply, history[1].Content)
	})

	t.Run("TestSendMessage_UpdatesSession", func(t *testing.T) {
		apiKey := os.Getenv("OPENAI_API_KEY")

		setOpenAIAPIKey(t, router, apiKey)
		defer deleteOpenAIAPIKey(t, router)

		sessionID := startSession(t, router, "Test Session")
		defer deleteSession(t, router, sessionID)
		assert.NotEmpty(t, sessionID)

		message1 := "Hello, how are you today? I am Yashwanth and I work at ThirdAI and my email is yash@thirdai.com"
		message2 := "Hello, how are you today? I am Tharun and I work at ThirdAI and my email is tharun@thirdai.com"
		_, _, _ = sendMessage(t, router, sessionID, message1, apiKey)
		_, _, _ = sendMessage(t, router, sessionID, message2, apiKey)

		session := getSession(t, router, sessionID)
		assert.Equal(t, "yash@thirdai.com", session.TagMap["[EMAIL_1]"])
		assert.Equal(t, "tharun@thirdai.com", session.TagMap["[EMAIL_2]"])
	})
}
