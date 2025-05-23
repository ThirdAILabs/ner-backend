package api

import "github.com/google/uuid"

type StartSessionRequest struct {
	Model string `json:"model"`
	Title string `json:"title"`
}

type ChatSessionMetadata struct {
	ID      uuid.UUID `json:"id"`
	Title   string `json:"title"`
}

type GetSessionsResponse struct {
	Sessions []ChatSessionMetadata `json:"sessions"`
}

type StartSessionResponse struct {
	SessionID string `json:"session_id"`
}

type RenameSessionRequest struct {
	Title string `json:"title"`
}

type ChatRequest struct {
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
	Message string `json:"message"`
}

type ChatResponse struct {
	InputText string            `json:"input_text"`
	Reply     string            `json:"reply"`
	TagMap    map[string]string `json:"tag_map"`
}

type ChatHistoryItem struct {
	MessageType string `json:"message_type"` // "user" or "ai"
	Content     string `json:"content"`
	Timestamp   string `json:"timestamp"`
	Metadata    any    `json:"metadata,omitempty"` // Optional metadata field
}

type ApiKey struct {
	ApiKey string `json:"api_key"`
}