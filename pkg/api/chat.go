package api

type StartSessionRequest struct {
	Model string `json:"model"`
}

type StartSessionResponse struct {
	SessionID string `json:"session_id"`
}

type ChatRequest struct {
	Model   string `json:"model"`
	APIKey  string `json:"api_key"`
	Message string `json:"message"`
}

type ChatResponse struct {
	Reply  string            `json:"reply"`
	TagMap map[string]string `json:"tag_map"`
}

type ChatHistoryItem struct {
	MessageType string `json:"message_type"` // "user" or "ai"
	Content     string `json:"content"`
	Timestamp   string `json:"timestamp"`
	Metadata    any    `json:"metadata,omitempty"` // Optional metadata field
}
