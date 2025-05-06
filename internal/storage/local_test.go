package storage

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalProvider_GetObject(t *testing.T) {
	dir := t.TempDir()
	provider, err := NewLocalProvider(dir)
	require.NoError(t, err)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Hello, World!")

	// Create the file
	err = provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	// Retrieve the file
	data, err := provider.GetObject(context.Background(), bucket, key)
	require.NoError(t, err)

	if !bytes.Equal(data, content) {
		t.Errorf("expected %q, got %q", content, data)
	}
}

func TestLocalProvider_PutObject(t *testing.T) {
	dir := t.TempDir()
	provider, err := NewLocalProvider(dir)
	require.NoError(t, err)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Test content")

	err = provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	// Verify the file exists
	path := filepath.Join(dir, bucket, key)
	data, err := os.ReadFile(path)
	require.NoError(t, err)

	if !bytes.Equal(data, content) {
		t.Errorf("expected %q, got %q", content, data)
	}
}

func TestLocalProvider_ListObjects(t *testing.T) {
	dir := t.TempDir()
	provider, err := NewLocalProvider(dir)
	require.NoError(t, err)

	bucket := "test-bucket"
	files := []string{"test/file1.txt", "test/file2.txt", "test/file3.txt"}
	for _, file := range files {
		err := provider.PutObject(context.Background(), bucket, file, strings.NewReader("content"))
		require.NoError(t, err)
	}

	err = provider.PutObject(context.Background(), bucket, "haha/test.txt", strings.NewReader("content"))
	require.NoError(t, err)

	objects, err := provider.ListObjects(context.Background(), bucket, "test")
	require.NoError(t, err)

	listFiles := []string{}
	for _, obj := range objects {
		listFiles = append(listFiles, obj.Name)
	}
	assert.ElementsMatch(t, files, listFiles)

	iterFiles := []string{}
	for obj, err := range provider.IterObjects(context.Background(), bucket, "test") {
		assert.NoError(t, err)
		iterFiles = append(iterFiles, obj.Name)
	}
	assert.ElementsMatch(t, files, iterFiles)
}

func TestLocalProvider_GetObjectStream(t *testing.T) {
	dir := t.TempDir()

	provider, err := NewLocalProvider(dir)
	require.NoError(t, err)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Stream content")

	err = provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	stream, err := provider.GetObjectStream(bucket, key)
	require.NoError(t, err)

	data, err := io.ReadAll(stream)
	require.NoError(t, err)

	if !bytes.Equal(data, content) {
		t.Errorf("expected %q, got %q", content, data)
	}
}
