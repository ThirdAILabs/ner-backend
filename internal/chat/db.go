package chat

import (
	"ner-backend/internal/database"
	"sync"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

// SQLite only supports one writer at a time, so we need a lock
// whenever we write to the database
var dbMutex sync.Mutex

func GetSessions(db *gorm.DB) ([]database.ChatSession, error) {
	var sessions []database.ChatSession
	err := db.Find(&sessions).Error
	return sessions, err
}

func CreateSession(db *gorm.DB, session *database.ChatSession) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()
	return db.Create(session).Error
}

func GetSession(db *gorm.DB, sessionID uuid.UUID) (database.ChatSession, error) {
	var session database.ChatSession
	err := db.First(&session, "id = ?", sessionID).Error
	return session, err
}

func UpdateSessionTitle(db *gorm.DB, sessionID uuid.UUID, title string) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()
	return db.Model(&database.ChatSession{ID: sessionID}).Update("title", title).Error
}

func UpdateSessionTagMetadata(db *gorm.DB, sessionID uuid.UUID, tagMetadata []byte) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()
	return db.Model(&database.ChatSession{ID: sessionID}).Update("tag_metadata", tagMetadata).Error
}

func DeleteSession(db *gorm.DB, sessionID uuid.UUID) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()
	if err := db.Delete(&database.ChatHistory{}, "session_id = ?", sessionID).Error; err != nil {
		return err
	}
	return db.Delete(&database.ChatSession{}, "id = ?", sessionID).Error
}

func GetChatHistory(db *gorm.DB, sessionID uuid.UUID) ([]database.ChatHistory, error) {
	var history []database.ChatHistory
	err := db.Where("session_id = ?", sessionID).Order("timestamp ASC").Find(&history).Error
	return history, err
}

func SaveChatMessage(db *gorm.DB, message *database.ChatHistory) error {
	dbMutex.Lock()
	defer dbMutex.Unlock()
	return db.Create(message).Error
}
