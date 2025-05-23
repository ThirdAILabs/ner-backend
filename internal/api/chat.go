package api

import (
	"log/slog"
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gorm.io/gorm"

	"ner-backend/internal/chat"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/pkg/api"
)

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
		r.Post("/sessions/{session_id}", RestHandler(s.GetSession))
		r.Post("/sessions/{session_id}/rename", RestHandler(s.RenameSession))
		r.Post("/sessions/{session_id}/messages", RestHandler(s.SendMessage))
		r.Get("/sessions/{session_id}/history", RestHandler(s.GetHistory))
		r.Get("/api-key", RestHandler(s.GetOpenAIApiKey))
		r.Post("/api-key", RestHandler(s.SetOpenAIApiKey))
	})
}

func (s *ChatService) GetSessions(r *http.Request) (any, error) {
	var sessions []database.ChatSession
	err := s.db.Find(&sessions).Error
	if err != nil {
		return nil, err
	}

	return sessions, nil
}

func (s *ChatService) StartSession(r *http.Request) (any, error) {
	req, err := ParseRequest[api.StartSessionRequest](r)
	if err != nil {
		return nil, err
	}

	if err := s.manager.ValidateModel(req.Model); err != nil {
		return nil, err
	}
	
	sessionID := uuid.New()
	err = s.db.Create(&database.ChatSession{
		ID:      sessionID,
		Title:   req.Title,
	}).Error;
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

	var session database.ChatSession
	err = s.db.Where("id = ?", sessionID).First(&session).Error
	if err != nil {
		return nil, err
	}

	return session, nil
}

func (s *ChatService) RenameSession(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}
	req, err := ParseRequest[api.RenameSessionRequest](r)
	if err != nil {
		return nil, err
	}

	err = s.db.Model(&database.ChatSession{}).Where("id = ?", sessionID).Update("title", req.Title).Error
	if err != nil {
		return nil, err
	}

	return nil, nil
}

func (s *ChatService) SendMessage(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

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

	session, err := chat.NewChatSession(s.db, sessionID, engine, req.APIKey, s.model)
	if err != nil {
		return nil, err
	}

	redactedIpText, reply, tagMap, err := session.Chat(req.Message)
	if err != nil {
		return nil, err
	}

	return api.ChatResponse{InputText: redactedIpText, Reply: reply, TagMap: tagMap}, nil
}

func (s *ChatService) GetHistory(r *http.Request) (any, error) {
	sessionID, err := URLParamUUID(r, "session_id")
	if err != nil {
		return nil, err
	}

	var history []database.ChatHistory
	err = s.db.
		Where("session_id = ?", sessionID).
		Order("timestamp ASC").
		Find(&history).
		Error
	if err != nil {
		return nil, err
	}

	var resp []api.ChatHistoryItem
	for _, msg := range history {
		ts := msg.Timestamp.Format("2006-01-02 15:04:05")
		resp = append(resp, api.ChatHistoryItem{
			MessageType: msg.MessageType,
			Content:     msg.Content,
			Timestamp:   ts,
			Metadata:    msg.Metadata,
		})
	}

	return resp, nil
}

func (s *ChatService) GetOpenAIApiKey(r *http.Request) (any, error) {
	// TODO: Implement
	apiKey := "test-api-key"
	return api.ApiKey{ApiKey: apiKey}, nil
}

func (s *ChatService) SetOpenAIApiKey(r *http.Request) (any, error) {
	// TODO: Implement
	req, err := ParseRequest[api.ApiKey](r)
	if err != nil {
		return nil, err
	}
	slog.Info("setting openai api key", "api_key", req.ApiKey)

	return nil, nil
}
