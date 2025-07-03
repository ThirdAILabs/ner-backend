package chat

import (
	"encoding/json"
	"log/slog"
	"ner-backend/internal/database"
	"sync"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// ChatDB encapsulates the database connection and mutex for thread-safe operations
type ChatDB struct {
	db    *gorm.DB
	mutex sync.Mutex
}

// NewChatDB creates a new ChatDB instance
func NewChatDB(db *gorm.DB) *ChatDB {
	return &ChatDB{
		db: db,
	}
}

func (c *ChatDB) GetSessions() ([]database.ChatSession, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	var sessions []database.ChatSession
	err := c.db.Find(&sessions).Error
	return sessions, err
}

func (c *ChatDB) CreateSession(sessionID uuid.UUID, title string, extensionSessionID uuid.NullUUID) error {
	tagMetadata := NewTagMetadata()
	tagMetadataJSON, err := json.Marshal(tagMetadata)
	if err != nil {
		return err
	}

	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.db.Create(&database.ChatSession{
		ID: sessionID,
		Title: title,
		TagMetadata: tagMetadataJSON,
		ExtensionSessionId: extensionSessionID,
	}).Error
}

func (c *ChatDB) GetSession(sessionID uuid.UUID) (database.ChatSession, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	var session database.ChatSession
	err := c.db.First(&session, "id = ?", sessionID).Error
	return session, err
}

func (c *ChatDB) GetSessionByExtensionId(extensionID uuid.UUID) (database.ChatSession, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	var session database.ChatSession
	err := c.db.First(&session, "extension_session_id = ?", extensionID).Error
	return session, err
}

func (c *ChatDB) UpdateSessionTitle(sessionID uuid.UUID, title string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.db.Model(&database.ChatSession{ID: sessionID}).Update("title", title).Error
}

func (c *ChatDB) UpdateSessionTagMetadata(sessionID uuid.UUID, tagMetadata []byte) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.db.Model(&database.ChatSession{ID: sessionID}).Update("tag_metadata", tagMetadata).Error
}

func (c *ChatDB) DeleteSession(sessionID uuid.UUID) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if err := c.db.Delete(&database.ChatHistory{}, "session_id = ?", sessionID).Error; err != nil {
		return err
	}
	return c.db.Delete(&database.ChatSession{}, "id = ?", sessionID).Error
}

func (c *ChatDB) GetChatHistory(sessionID uuid.UUID) ([]database.ChatHistory, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	var history []database.ChatHistory
	err := c.db.Where("session_id = ?", sessionID).Order("timestamp ASC").Find(&history).Error
	return history, err
}

func (c *ChatDB) SaveChatMessage(message *database.ChatHistory) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.db.Create(message).Error
}

func (c *ChatDB) UpdateSessionExtensionId(oldExtensionID uuid.UUID, newExtensionID uuid.UUID) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	slog.Info("Updating session extension ID", "oldExtensionID", oldExtensionID, "newExtensionID", newExtensionID)
	return c.db.Model(&database.ChatSession{}).Where("extension_session_id = ?", uuid.NullUUID{UUID: oldExtensionID, Valid: true}).Update("extension_session_id", uuid.NullUUID{UUID: newExtensionID, Valid: true}).Error
}