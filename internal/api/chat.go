package api

import (
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
		r.Post("/sessions", RestHandler(s.StartSession))
		r.Post("/sessions/{session_id}/messages", RestHandler(s.SendMessage))
		r.Get("/sessions/{session_id}/history", RestHandler(s.GetHistory))
	})
}

func (s *ChatService) StartSession(r *http.Request) (any, error) {
	req, err := ParseRequest[api.StartSessionRequest](r)
	if err != nil {
		return nil, err
	}

	sessionID := uuid.New().String()
	if err := s.manager.ValidateModel(req.Model); err != nil {
		return nil, err
	}

	return api.StartSessionResponse{SessionID: sessionID}, nil
}

func (s *ChatService) SendMessage(r *http.Request) (any, error) {
	sessionID := chi.URLParam(r, "session_id")
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

	reply, tagMap, err := session.Chat(req.Message)
	if err != nil {
		return nil, err
	}

	return api.ChatResponse{Reply: reply, TagMap: tagMap}, nil
}

func (s *ChatService) GetHistory(r *http.Request) (any, error) {
	sessionID := chi.URLParam(r, "session_id")

	var history []database.ChatHistory
	err := s.db.
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
