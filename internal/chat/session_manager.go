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
			"gpt-4": "gpt-4o-mini",
			"gpt-3": "gpt-3.5-turbo",
		},
	}
}

func (manager *ChatSessionManager) ValidateModel(model string) error {
	if _, exists := manager.models[model]; !exists {
		return fmt.Errorf("model %s not supported", model)
	}

	return nil
}

func (m *ChatSessionManager) EngineName(key string) (string, error) {
	if eng, ok := m.models[key]; ok {
		return eng, nil
	}
	return "", fmt.Errorf("model %q not supported", key)
}
