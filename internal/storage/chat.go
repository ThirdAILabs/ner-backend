package storage

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/fsnotify/fsnotify"
	"github.com/google/uuid"
)

var MAX_CHAT_HISTORY = 1000
var BASE_PATH = "chat/"

type ChatQueue struct {
	path string
}

func NewChatQueue(uuid uuid.UUID) (*ChatQueue, error) {
	path := filepath.Join(BASE_PATH, uuid.String())
	// Create a new folder at path if it doesn't already exist.
	err := os.MkdirAll(path, 0755)
	if err != nil {
		slog.Error("Failed to create directory", "error", err)
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}
	
	// If path / 0.txt does not exist, create it.
	filePath := filepath.Join(path, "0.txt")
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		_, err = os.Create(filePath)
		if err != nil {
			slog.Error("Failed to create file", "error", err)
			return nil, fmt.Errorf("failed to create file: %w", err)
		}
	}

	return &ChatQueue{path: path}, nil
}

func (s *ChatQueue) indexToObjName(index int) string {
	return fmt.Sprintf("%d.txt", index)
}

func (s *ChatQueue) objNameToIndex(objName string) (int, error) {
	index, err := strconv.Atoi(strings.TrimSuffix(objName, ".txt"))
	if err != nil {
		slog.Error("Failed to convert file name to int", "error", err)
		return 0, fmt.Errorf("failed to convert file name to int: %w", err)
	}
	return index, nil
}

func (s *ChatQueue) Append(message string) error {
	// List all files in the directory. Files are named 0.txt, 1.txt, etc.
	files, err := os.ReadDir(s.path)
	if err != nil {
		slog.Error("Failed to read directory", "error", err)
		return fmt.Errorf("failed to read directory: %w", err)
	}
	
	// Remove file extension and convert to int to find the highest index.
	highestIndex := -1
	for _, file := range files {
		index, err := s.objNameToIndex(file.Name())
		if err != nil {
			slog.Error("Failed to convert file name to int", "error", err)
			continue
		}
		if index > highestIndex {
			highestIndex = index
		}
	}

	// Do not write to MAX_CHAT_HISTORY.txt.
	if highestIndex >= MAX_CHAT_HISTORY {
		slog.Error("Chat history limit reached")
		return fmt.Errorf("chat history limit reached")
	}
	
	// Write message to highestIndex.txt, then create highestIndex + 1.txt.
	filePath := filepath.Join(s.path, s.indexToObjName(highestIndex))
	file, err := os.OpenFile(filePath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		slog.Error("Failed to open file", "error", err)
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	_, err = file.Write([]byte(message))
	if err != nil {
		slog.Error("Failed to write to file", "error", err)
		return fmt.Errorf("failed to write to file: %w", err)
	}

	// Create highestIndex + 1.txt to mark that highestIndex.txt is ready.
	filePath = filepath.Join(s.path, s.indexToObjName(highestIndex + 1))
	_, err = os.Create(filePath)
	if err != nil {
		slog.Error("Failed to create file", "error", err)
		return fmt.Errorf("failed to create file: %w", err)
	}

	return nil
}

func (s *ChatQueue) EndChat() error {
	filePath := filepath.Join(s.path, "end.txt")
	_, err := os.Create(filePath)
	if err != nil {
		slog.Error("Failed to create file", "error", err)
		return fmt.Errorf("failed to create file: %w", err)
	}
	return nil
}

// Returns (message, endOfChat, error)
func (s *ChatQueue) Pop(newlyCreatedFile string, removeAfterRead bool) (string, bool, error) {
	slog.Info("Pop", "newlyCreatedFile", filepath.Base(newlyCreatedFile))
	// First check if we have reached the end of the chat.
	if filepath.Base(newlyCreatedFile) == "end.txt" {
		return "", true, nil
	}

	// In AppendChatMessage, we write the latest message to
	// highestIndex.txt and then we create highestIndex + 1.txt.
	// Hence the lastMessageIndex is the extensionless part of newlyCreatedFile - 1.
	newlyCreatedFileIndex, err := s.objNameToIndex(filepath.Base(newlyCreatedFile))
	if err != nil {
		slog.Error("Failed to convert file name to int", "error", err)
		return "", false, fmt.Errorf("failed to convert file name to int: %w", err)
	}
	lastMessageIndex := newlyCreatedFileIndex - 1
	
	slog.Info("Pop", "lastMessageIndex", lastMessageIndex)
	
	// Read and return the message in path/index.txt
	filePath := filepath.Join(s.path, s.indexToObjName(lastMessageIndex))
	slog.Info("Pop", "filePath", filePath)
	message, err := os.ReadFile(filePath)
	if err != nil {
		slog.Error("Failed to read file", "error", err)
		return "", false, fmt.Errorf("failed to read file: %w", err)
	}

	slog.Info("Pop", "message", string(message))

	if removeAfterRead {
		err = os.Remove(filePath)
		if err != nil {
			slog.Error("Failed to remove file", "error", err)
			return "", false, fmt.Errorf("failed to remove file: %w", err)
		}
	}

	return string(message), false, nil
}


type ChatProvider struct {
	queue *ChatQueue
	messages []string
	newMessageReceived chan struct{}
	done bool
}

func NewChatProvider(uuid uuid.UUID) (*ChatProvider, error) {
	queue, err := NewChatQueue(uuid)
	if err != nil {
		return nil, fmt.Errorf("failed to create chat storage: %w", err)
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		slog.Error("Failed to create watcher", "error", err)
		return nil, fmt.Errorf("failed to create watcher: %w", err)
	}

	messages := make([]string, 0, MAX_CHAT_HISTORY)
	newMessageReceived := make(chan struct{})

	provider := &ChatProvider{
		queue: queue,
		messages: messages,
		newMessageReceived: newMessageReceived,
		done: false,
	}

    // Start listening for events.
    go func() {
		seen := make(map[string]bool)
        for {
            select {
            case event, ok := <-watcher.Events:
                if !ok {
                    return
                }
				slog.Info("event", "event", event)
                if event.Has(fsnotify.Create) {
					slog.Info("Handling event", "file", event.Name)
					if seen[event.Name] {
						slog.Info("Already seen", "file", event.Name)
						continue
					}
					seen[event.Name] = true
					message, endOfChat, err := queue.Pop(event.Name, false)
					if err != nil {
						slog.Error("Failed to get last message", "error", err)
						continue
					}
					if endOfChat {
						watcher.Close()
						provider.done = true
						provider.newMessageReceived <- struct{}{}
						slog.Info("Done")
						return
					} else {
						slog.Info("Appending message", "message", message)
						// TODO Is this going to update correctly?
						provider.messages = append(provider.messages, message)
						provider.newMessageReceived <- struct{}{}
						slog.Info("Notified")
					}
                }
            case err, ok := <-watcher.Errors:
                if !ok {
                    return
                }
				slog.Error("error", "error", err)
			}
		}
	}()

	err = watcher.Add(queue.path)
	if err != nil {
		slog.Error("Failed to add path to watcher", "path", queue.path, "error", err)
		return nil, fmt.Errorf("failed to add path %s to watcher: %w", queue.path, err)
	}

	return provider, nil
}

// These methods are needed to process inference tasks.

func (p *ChatProvider) GetObjectStream(bucket, key string) (io.Reader, error) {
	slog.Info("GetObjectStream", "bucket", bucket, "key", key)
	index, err := p.queue.objNameToIndex(key)
	if err != nil {
		return nil, err
	}

	// Wait for the message to be received.
	for ; index >= len(p.messages) && !p.done; {
		slog.Info("Waiting for message", "index", index, "current", len(p.messages), "done", p.done)
		<- p.newMessageReceived
		slog.Info("Notified and done waiting", "index", index, "current", len(p.messages), "done", p.done)
	}

	slog.Info("Done waiting", "index", index, "current", len(p.messages), "done", p.done)

	if p.done && index >= len(p.messages) {
		slog.Info("Done", "index", index, "current", len(p.messages), "done", p.done)
		return bytes.NewReader([]byte{}), nil
	}

	message := p.messages[index]
	slog.Info("Returning message", "message", message)
	return bytes.NewReader([]byte(message)), nil
}

func (p *ChatProvider) IterObjects(ctx context.Context, bucket, dir string) ObjectIterator {
	return func(yield func(obj Object, err error) bool) {
		for i := range MAX_CHAT_HISTORY {
			// Size = 1 is a placeholder.
			if !yield(Object{Name: fmt.Sprintf("%d.txt", i), Size: 1}, nil) {
				return
			}
		}	
	}
}

// The other methods are not supported.

func (p *ChatProvider) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func (p *ChatProvider) GetObject(ctx context.Context, bucket, key string) ([]byte, error) {
	return nil, nil
}

func (p *ChatProvider) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	return nil
}

func (p *ChatProvider) DownloadDir(ctx context.Context, bucket, prefix, dest string) error {
	return nil
}

func (p *ChatProvider) UploadDir(ctx context.Context, bucket, prefix, src string) error {
	return nil
}

func (p *ChatProvider) ListObjects(ctx context.Context, bucket, dir string) ([]Object, error) {
	return nil, nil
}
