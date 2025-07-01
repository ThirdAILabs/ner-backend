package api

import (
	"time"

	"github.com/google/uuid"
)

type StartSessionRequest struct {
	Model string
	Title string
}

type ChatSessionMetadata struct {
	ID      uuid.UUID
	Title   string
	TagMap  map[string]string
}

type GetSessionsResponse struct {
	Sessions []ChatSessionMetadata
}

type StartSessionResponse struct {
	SessionID string
}

type RenameSessionRequest struct {
	Title string
}

type ChatRequest struct {
	Model   string
	Message string
}

type ChatResponse struct {
	InputText string
	Reply     string
	TagMap    map[string]string
}

type ChatHistoryItem struct {
	MessageType string
	Content     string
	Timestamp   time.Time
	Metadata    any
}

type ApiKey struct {
	ApiKey string
}

type ChatMessage struct {
	Message string
}
