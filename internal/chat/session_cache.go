package chat

import (
	"ner-backend/internal/core"
	"sync"
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type sessionEntry struct {
	session      *ChatSession
	lastAccessed time.Time
}

type SessionCache struct {
	lock     sync.Mutex
	sessions map[uuid.UUID]*sessionEntry
	maxSize  int
}

func NewSessionPool(maxSize int) *SessionCache {
	return &SessionCache{
		sessions: make(map[uuid.UUID]*sessionEntry, maxSize),
		maxSize:  maxSize,
	}
}

func (pool *SessionCache) GetSession(db *gorm.DB, sessionID uuid.UUID, model, apiKey string, ner core.Model) (*ChatSession, error) {
	pool.lock.Lock()
	defer pool.lock.Unlock()

	session, exists := pool.sessions[sessionID]
	if !exists {
		oldestSessionID := uuid.Nil
		var oldestTime time.Time
		for id, entry := range pool.sessions {
			if oldestSessionID == uuid.Nil || entry.lastAccessed.Before(oldestTime) {
				oldestSessionID = id
				oldestTime = entry.lastAccessed
			}
		}

		oldestSession := pool.sessions[oldestSessionID]
		oldestSession.session.mu.Lock()
		delete(pool.sessions, oldestSessionID)
		oldestSession.session.mu.Unlock()

		session, err := NewChatSession(db, sessionID, model, apiKey, ner)
		if err != nil {
			return nil, err
		}
		pool.sessions[sessionID] = &sessionEntry{
			session:      session,
			lastAccessed: time.Now(),
		}

		return session, nil
	}

	// Update last accessed time
	session.lastAccessed = time.Now()
	return session.session, nil
}
