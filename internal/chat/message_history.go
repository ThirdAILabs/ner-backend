package chat

type ChatHistory struct {
	ID          uint   `gorm:"primary_key"`
	SessionID   string `gorm:"index"`
	MessageType string // 'user' or 'ai'
	Content     string
	Timestamp   string `gorm:"default:CURRENT_TIMESTAMP"`
}
