package utils

import (
	"fmt"
	"log/slog"
	"sync"
)

type MutexMap struct {
	edit sync.Mutex
	queueLengths map[string]int
	mutexes map[string]*sync.Mutex
	maxSize int
}

func NewMutexMap(maxSize int) MutexMap {
	return MutexMap{
		queueLengths: make(map[string]int),
		mutexes: make(map[string]*sync.Mutex),
		maxSize: maxSize,
	}
}

func (m *MutexMap) Lock(key string) error {
	m.edit.Lock()

	if m.mutexes[key] == nil {
		slog.Info("Mutex not found, creating new one", "key", key)
		if len(m.mutexes) >= m.maxSize {
			m.edit.Unlock()
			return fmt.Errorf("max size reached")
		}

		m.mutexes[key] = &sync.Mutex{}
		m.queueLengths[key] = 0
	} else {
		slog.Info("Mutex found", "key", key)
	}

	m.queueLengths[key]++
	m.edit.Unlock()
	
	m.mutexes[key].Lock()

	return nil
}

func (m *MutexMap) Unlock(key string) error {
	m.edit.Lock()

	if m.mutexes[key] == nil {
		m.edit.Unlock()
		return fmt.Errorf("key %s not found", key)
	}

	m.mutexes[key].Unlock()
	m.queueLengths[key]--

	if m.queueLengths[key] == 0 {
		delete(m.mutexes, key)
		delete(m.queueLengths, key)
	}

	m.edit.Unlock()
	
	return nil
}
