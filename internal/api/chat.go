package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"

	"ner-backend/internal/chat"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/pkg/api"
)

type SessionLock struct {
	mu sync.Mutex
	sessions map[uuid.UUID]bool
}

func NewSessionLock() *SessionLock {
	return &SessionLock{
		sessions: make(map[uuid.UUID]bool),
	}
}

func (l *SessionLock) Lock(sessionID uuid.UUID) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if _, ok := l.sessions[sessionID]; ok {
		return fmt.Errorf("session %s is currently in use", sessionID)
	}
	l.sessions[sessionID] = true
	return nil
}

func (l *SessionLock) Unlock(sessionID uuid.UUID) {
	l.mu.Lock()
	defer l.mu.Unlock()

	delete(l.sessions, sessionID)
}

type ChatService struct {
	db      *chat.ChatDB
	manager *chat.ChatSessionManager
	model   core.Model
	lock    *SessionLock
}

func NewChatService(db *gorm.DB, model core.Model) *ChatService {
	return &ChatService{
		db:      chat.NewChatDB(db),
		manager: chat.NewChatSessionManager(db),
		model:   model,
		lock:    NewSessionLock(),
	}
}

func (s *ChatService) AddRoutes(r chi.Router) {
	r.Route("/chat", func(r chi.Router) {
		r.Get("/sessions", RestHandler(s.GetSessions))
		r.Post("/sessions", RestHandler(s.StartSession))
		r.Get("/sessions/{session_id}", RestHandler(s.GetSession))
		r.Post("/sessions/{session_id}/rename", RestHandler(s.RenameSession))
		r.Post("/sessions/{session_id}/messages", RestStreamHandler(s.SendMessageStream))
		r.Get("/sessions/{session_id}/history", RestHandler(s.GetHistory))
		r.Get("/api-key", RestHandler(s.GetOpenAIApiKey))
		r.Post("/api-key", RestHandler(s.SetOpenAIApiKey))
		r.Post("/api-key/validate", RestHandler(s.ValidateOpenAIApiKey))
		r.Delete("/api-key", RestHandler(s.DeleteOpenAIApiKey))
		r.Delete("/sessions/{session_id}", RestHandler(s.DeleteSession))
	})
}

func (s *ChatService) GetSessions(r *http.Request) (any, error) {
	sessions, err := s.db.GetSessions()
	if err != nil {
		return nil, err
	}

	apiSessions := make([]api.ChatSessionMetadata, len(sessions))
	for i, session := range sessions {
		apiSessions[i] = api.ChatSessionMetadata{
			ID:      session.ID,
			Title:   session.Title,
		}
	}

	return api.GetSessionsResponse{Sessions: apiSessions}, nil
}

func (s *ChatService) StartSession(r *http.Request) (any, error) {
	req, err := ParseRequest[api.StartSessionRequest](r)
	if err != nil {
		return nil, err
	}

	if err := s.manager.ValidateModel(req.Model); err != nil {
		return nil, err
	}

	tagMetadata := chat.NewTagMetadata()
	tagMetadataJSON, err := json.Marshal(tagMetadata)
	if err != nil {
		return nil, err
	}
	
	sessionID := uuid.New()
	err = s.db.CreateSession(&database.ChatSession{
		ID:          sessionID,
		Title:       req.Title,
		TagMetadata: tagMetadataJSON,
	})
	if err != nil {
		return nil, err
	}

	return api.StartSessionResponse{SessionID: sessionID.String()}, nil
}

func (s *ChatService) GetSession(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := s.lock.Lock(sessionID); err != nil {
		return nil, err
	}
	defer s.lock.Unlock(sessionID)

	session, err := s.db.GetSession(sessionID)
	if err != nil {
		slog.Error("Error getting session", "error", err)
		return nil, fmt.Errorf("error getting session: %v", err)
	}

	var tagMetadata chat.TagMetadata
	if err := json.Unmarshal(session.TagMetadata, &tagMetadata); err != nil {
		slog.Error("Error getting session", "error", err)
		return nil, fmt.Errorf("error getting session tag metadata: %v", err)
	}

	return api.ChatSessionMetadata{
		ID:      session.ID,
		Title:   session.Title,
		TagMap:  tagMetadata.TagMap,
	}, nil
}

func (s *ChatService) RenameSession(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}
	
	if err := s.lock.Lock(sessionID); err != nil {
		return nil, err
	}
	defer s.lock.Unlock(sessionID)

	req, err := ParseRequest[api.RenameSessionRequest](r)
	if err != nil {
		return nil, err
	}

	if err := s.db.UpdateSessionTitle(sessionID, req.Title); err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *ChatService) SendMessageStream(r *http.Request) (StreamResponse, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := s.lock.Lock(sessionID); err != nil {
		return nil, err
	}
	defer s.lock.Unlock(sessionID)

	req, err := ParseRequest[api.ChatRequest](r)
	if err != nil {
		return nil, err
	}

	if err := s.manager.ValidateModel(req.Model); err != nil {
		return nil, err
	}

	engine, err := s.manager.EngineName(req.Model)
	if err != nil {
		return nil, err
	}

	apiKey := s.getOpenAIApiKey()
	session, err := chat.NewChatSession(s.db, sessionID, engine, apiKey, s.model)
	if err != nil {
		return nil, err
	}
	
	chatIterator, err := session.ChatStream(req.Message)
	if err != nil {
		return nil, err
	}

	response := func(yield func(any, error) bool) {
		for item, err := range chatIterator {
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(api.ChatResponse{InputText: item.RedactedText, Reply: item.Reply, TagMap: item.TagMap}, nil) {
				break
			}
		}
	}

	return response, nil
}

func (s *ChatService) GetHistory(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := s.lock.Lock(sessionID); err != nil {
		return nil, err
	}
	defer s.lock.Unlock(sessionID)

	history, err := s.db.GetChatHistory(sessionID)
	if err != nil {
		return nil, err
	}

	resp := make([]api.ChatHistoryItem, len(history))
	for i, item := range history {
		resp[i] = api.ChatHistoryItem{
			MessageType: item.MessageType,
			Content:     item.Content,
			Timestamp:   item.Timestamp,
			Metadata:    item.Metadata,
		}
	}

	return resp, nil
}

func (s *ChatService) DeleteSession(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := s.lock.Lock(sessionID); err != nil {
		return nil, err
	}
	defer s.lock.Unlock(sessionID)

	if err := s.db.DeleteSession(sessionID); err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *ChatService) getOpenAIApiKey() string {
	// TODO: Store in a more secure way.
	apiKey := ""
	if data, err := os.ReadFile("api-key.txt"); err == nil {
		apiKey = strings.TrimSpace(string(data));
	}
	return apiKey
}

func (s *ChatService) GetOpenAIApiKey(r *http.Request) (any, error) {
	return api.ApiKey{ApiKey: s.getOpenAIApiKey()}, nil
}

func (s *ChatService) SetOpenAIApiKey(r *http.Request) (any, error) {
	// TODO: Store in a more secure way.
	req, err := ParseRequest[api.ApiKey](r)
	if err != nil {
		return nil, err
	}

	slog.Info("Setting OpenAI API key", "key_length", len(req.ApiKey))
	
	err = os.WriteFile("api-key.txt", []byte(req.ApiKey), 0600)
	if err != nil {
		slog.Error("Failed to write API key file", "error", err)
		return nil, err
	}
	
	// Also set it as environment variable immediately
	os.Setenv("OPENAI_API_KEY", req.ApiKey)
	slog.Info("OpenAI API key saved and set as environment variable")

	return nil, nil
}

func (s *ChatService) DeleteOpenAIApiKey(r *http.Request) (any, error) {
	err := os.Remove("api-key.txt")
	if err != nil {
		return nil, err
	}
	return nil, nil
}

func (s *ChatService) ValidateOpenAIApiKey(r *http.Request) (any, error) {
	req, err := ParseRequest[api.ApiKey](r)
	if err != nil {
		return nil, err
	}

	// Basic validation
	apiKey := strings.TrimSpace(req.ApiKey)
	if apiKey == "" {
		return api.ValidationResponse{Valid: false, Message: "API key is empty"}, nil
	}

	// Check if it starts with expected prefix
	if !strings.HasPrefix(apiKey, "sk-") {
		return api.ValidationResponse{Valid: false, Message: "API key format is invalid"}, nil
	}

	// For now, just do basic format validation
	// A full implementation would make a test API call to OpenAI
	return api.ValidationResponse{Valid: true, Message: "API key format is valid"}, nil
}
