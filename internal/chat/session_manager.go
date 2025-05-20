package chat

import (
	"fmt"

	"gorm.io/gorm"
)

type ChatSessionManager struct {
	db     *gorm.DB
	models map[string]string
}

func NewChatSessionManager(db *gorm.DB) *ChatSessionManager {
	return &ChatSessionManager{
		db: db,
		models: map[string]string{
			"gpt-4": "gpt-4",
			"gpt-3": "gpt-3.5",
		},
	}
}

func (manager *ChatSessionManager) StartSession(model, sessionID, apiKey string) (*ChatSession, error) {
	if _, exists := manager.models[model]; !exists {
		return nil, fmt.Errorf("model %s not supported", model)
	}

	session, err := NewChatSession(manager.db, sessionID, model, apiKey)
	if err != nil {
		return nil, err
	}

	return session, nil
}
