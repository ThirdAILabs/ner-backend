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
	sessions map[string]bool
}

func NewSessionLock() *SessionLock {
	return &SessionLock{
		sessions: make(map[string]bool),
	}
}

func (l *SessionLock) Lock(sessionID string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if _, ok := l.sessions[sessionID]; ok {
		return fmt.Errorf("session %s is currently in use", sessionID)
	}
	l.sessions[sessionID] = true
	return nil
}

func (l *SessionLock) Unlock(sessionID string) {
	l.mu.Lock()
	defer l.mu.Unlock()

	delete(l.sessions, sessionID)
}

var sessionLock = NewSessionLock()

type ChatService struct {
	db      *gorm.DB
	manager *chat.ChatSessionManager
	model   core.Model
}

func NewChatService(db *gorm.DB, model core.Model) *ChatService {
	return &ChatService{
		db:      db,
		manager: chat.NewChatSessionManager(db),
		model:   model,
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
		r.Delete("/api-key", RestHandler(s.DeleteOpenAIApiKey))
		r.Delete("/sessions/{session_id}", RestHandler(s.DeleteSession))
	})
}

func (s *ChatService) GetSessions(r *http.Request) (any, error) {
	sessions, err := chat.GetSessions(s.db)
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
	err = chat.CreateSession(s.db, &database.ChatSession{
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

	if err := sessionLock.Lock(sessionID.String()); err != nil {
		return nil, err
	}
	defer sessionLock.Unlock(sessionID.String())

	session, err := chat.GetSession(s.db, sessionID)
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
	
	if err := sessionLock.Lock(sessionID.String()); err != nil {
		return nil, err
	}
	defer sessionLock.Unlock(sessionID.String())

	req, err := ParseRequest[api.RenameSessionRequest](r)
	if err != nil {
		return nil, err
	}

	if err := chat.UpdateSessionTitle(s.db, sessionID, req.Title); err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *ChatService) SendMessageStream(r *http.Request) (StreamResponse, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := sessionLock.Lock(sessionID.String()); err != nil {
		return nil, err
	}
	defer sessionLock.Unlock(sessionID.String())

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

	if err := sessionLock.Lock(sessionID.String()); err != nil {
		return nil, err
	}
	defer sessionLock.Unlock(sessionID.String())

	history, err := chat.GetChatHistory(s.db, sessionID)
	if err != nil {
		return nil, err
	}

	var resp []api.ChatHistoryItem
	for _, msg := range history {
		resp = append(resp, api.ChatHistoryItem{
			MessageType: msg.MessageType,
			Content:     msg.Content,
			Timestamp:   msg.Timestamp,
			Metadata:    msg.Metadata,
		})
	}

	return resp, nil
}

func (s *ChatService) DeleteSession(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	if err := sessionLock.Lock(sessionID.String()); err != nil {
		return nil, err
	}
	defer sessionLock.Unlock(sessionID.String())
	
	if err := chat.DeleteSession(s.db, sessionID); err != nil {
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

	err = os.WriteFile("api-key.txt", []byte(req.ApiKey), 0600)
	if err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *ChatService) DeleteOpenAIApiKey(r *http.Request) (any, error) {
	err := os.Remove("api-key.txt")
	if err != nil {
		return nil, err
	}
	return nil, nil
}
