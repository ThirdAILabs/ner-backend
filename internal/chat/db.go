package chat

import (
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

func (c *ChatDB) CreateSession(session *database.ChatSession) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.db.Create(session).Error
}

func (c *ChatDB) GetSession(sessionID uuid.UUID) (database.ChatSession, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	var session database.ChatSession
	err := c.db.First(&session, "id = ?", sessionID).Error
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
