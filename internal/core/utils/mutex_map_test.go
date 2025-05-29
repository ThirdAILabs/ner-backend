package utils_test

import (
	"log/slog"
	"ner-backend/internal/core/utils"
	"testing"
	"time"
)

func TestMutexMap_RunSequentiallyWhenSameKey(t *testing.T) {
	m := utils.NewMutexMap(10)
	key := "test"

	sleep_duration := 500 * time.Millisecond
	
	routine := func(wait chan bool) {
		slog.Info("First routine started", "key", key)
		err := m.Lock(key)
		if err != nil {
			t.Errorf("Error locking key: %v", err)
		}
		
		time.Sleep(sleep_duration)
		m.Unlock(key)
		wait <- true
	}
	
	wait1 := make(chan bool)
	wait2 := make(chan bool)
	
	start := time.Now()
	go routine(wait1)
	go routine(wait2)

	<-wait1
	<-wait2

	elapsed := time.Since(start)
	if elapsed < 2 * sleep_duration {
		t.Errorf("Routines are not running sequentially, expected > %v elapsed, got %v", 2 * sleep_duration, elapsed)
	}
}

func TestMutexMap_RunConcurrentlyWhenDifferentKeys(t *testing.T) {
	m := utils.NewMutexMap(10)
	
	sleep_duration := 500 * time.Millisecond
	
	routine := func(key string,wait chan bool) {
		slog.Info("First routine started", "key", key)
		err := m.Lock(key)
		if err != nil {
			t.Errorf("Error locking key: %v", err)
		}
		
		time.Sleep(sleep_duration)
		m.Unlock(key)
		wait <- true
	}
	
	wait1 := make(chan bool)
	wait2 := make(chan bool)

	start := time.Now()
	go routine("key1", wait1)
	go routine("key2", wait2)

	<-wait1
	<-wait2

	elapsed := time.Since(start)
	
	if elapsed > 750 * time.Millisecond {
		t.Errorf("Routines are not running concurrently, expected around %v elapsed, got %v", sleep_duration, elapsed)
	}
}

func TestMutexMap_MixOfSameAndDifferentKeys(t *testing.T) {
	m := utils.NewMutexMap(10)
	
	sleep_duration := 500 * time.Millisecond
	
	routine := func(key string,wait chan bool) {
		slog.Info("First routine started", "key", key)
		err := m.Lock(key)
		if err != nil {
			t.Errorf("Error locking key: %v", err)
		}
		
		time.Sleep(sleep_duration)
		m.Unlock(key)
		wait <- true
	}
	
	wait1 := make(chan bool)
	wait2 := make(chan bool)
	wait3 := make(chan bool)
	
	start := time.Now()
	go routine("key1", wait1)
	go routine("key1", wait2)
	go routine("key2", wait3)

	<-wait1
	<-wait2

	elapsed := time.Since(start)
	if elapsed < 2 * sleep_duration {
		t.Errorf("Same-key routines are not running sequentially, expected > %v elapsed, got %v", 2 * sleep_duration, elapsed)
	}

	<-wait3
	
	if elapsed > 1250 * time.Millisecond {
		t.Errorf("Different-key routines are not running concurrently, expected around %v elapsed, got %v", 2 * sleep_duration, elapsed)
	}
}

func TestMutexMap_ErrorWhenMaxSizeReached(t *testing.T) {
	m := utils.NewMutexMap(1)
	key1 := "test1"
	key2 := "test2"

	err := m.Lock(key1)
	if err != nil {
		t.Errorf("Error locking key1: %v", err)
	}

	err = m.Lock(key2)
	if err == nil {
		t.Errorf("Expected error when max size reached, got nil")
	}
}

func TestMutexMap_UnlockErrorWhenKeyNotFound(t *testing.T) {
	m := utils.NewMutexMap(10)
	key := "test"

	err := m.Unlock(key)
	if err == nil {
		t.Errorf("Expected error when unlocking key not found, got nil")
	}
}