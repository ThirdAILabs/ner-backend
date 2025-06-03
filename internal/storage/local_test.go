package storage

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestProvider(t *testing.T) (*LocalProvider, string) {
	t.Helper()
	dir := t.TempDir()
	provider, err := NewLocalProvider(dir)
	require.NoError(t, err)
	return provider, dir
}

func TestProvider_GetObject(t *testing.T) {
	provider, _ := setupTestProvider(t)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Hello, World!")

	err := provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	data, err := provider.GetObject(context.Background(), bucket, key)
	require.NoError(t, err)
	assert.Equal(t, content, data)
}

func TestProvider_GetObjectStream(t *testing.T) {
	provider, _ := setupTestProvider(t)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Stream content")

	err := provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	stream, err := provider.GetObjectStream(bucket, key)
	require.NoError(t, err)

	data, err := io.ReadAll(stream)
	require.NoError(t, err)
	assert.Equal(t, content, data)
}

func TestProvider_PutObject(t *testing.T) {
	provider, baseDir := setupTestProvider(t)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Test content")

	err := provider.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	filePath := filepath.Join(baseDir, bucket, key)
	data, err := os.ReadFile(filePath)
	require.NoError(t, err)
	assert.Equal(t, content, data)
}

func TestProvider_ListObjects(t *testing.T) {
	provider, _ := setupTestProvider(t)

	bucket := "test-bucket"
	files := []string{"file1.txt", "file2.txt", "file3.txt"}
	for _, file := range files {
		err := provider.PutObject(context.Background(), bucket, file, bytes.NewReader([]byte("content")))
		require.NoError(t, err)
	}

	objects, err := provider.ListObjects(context.Background(), bucket, "")
	require.NoError(t, err)

	var objectNames []string
	for _, obj := range objects {
		objectNames = append(objectNames, obj.Name)
	}
	assert.ElementsMatch(t, files, objectNames)
}

func TestProvider_IterObjects(t *testing.T) {
	provider, _ := setupTestProvider(t)

	bucket := "test-bucket"
	files := []string{"dir1/file1.txt", "dir1/file2.txt", "dir2/file3.txt"}
	for _, file := range files {
		err := provider.PutObject(context.Background(), bucket, file, bytes.NewReader([]byte("content")))
		require.NoError(t, err)
	}

	iterFiles := []string{}
	iter := provider.IterObjects(context.Background(), bucket, "dir1")
	iter(func(obj Object, err error) bool {
		require.NoError(t, err)
		iterFiles = append(iterFiles, obj.Name)
		return true
	})

	expected := []string{"dir1/file1.txt", "dir1/file2.txt"}
	assert.ElementsMatch(t, expected, iterFiles)
}

func TestProvider_UploadDir(t *testing.T) {
	provider, baseDir := setupTestProvider(t)

	bucket := "test-bucket"
	prefix := "uploaded"
	srcDir := t.TempDir()

	// Create test files in the source directory
	files := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, file := range files {
		filePath := filepath.Join(srcDir, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("content"), os.ModePerm))
	}

	err := provider.UploadDir(context.Background(), bucket, prefix, srcDir)
	require.NoError(t, err)

	// Verify files were uploaded
	for _, file := range files {
		uploadedPath := filepath.Join(baseDir, bucket, prefix, file)
		data, err := os.ReadFile(uploadedPath)
		require.NoError(t, err)
		assert.Equal(t, "content", string(data))
	}
}

func TestProvider_DownloadDir(t *testing.T) {
	provider, baseDir := setupTestProvider(t)

	bucket := "test-bucket"
	prefix := "to-download"
	destDir := t.TempDir()

	// Create test files in the provider
	files := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, file := range files {
		filePath := filepath.Join(baseDir, bucket, prefix, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("content"), os.ModePerm))
	}

	err := provider.DownloadDir(context.Background(), bucket, prefix, destDir, false)
	require.NoError(t, err)

	// Verify files were downloaded
	for _, file := range files {
		downloadedPath := filepath.Join(destDir, file)
		data, err := os.ReadFile(downloadedPath)
		require.NoError(t, err)
		assert.Equal(t, "content", string(data))
	}
}

func TestProvider_DownloadDir_Overwrite(t *testing.T) {
	provider, baseDir := setupTestProvider(t)

	bucket := "test-bucket"
	prefix := "to-download"
	destDir := t.TempDir()

	// First create a file in the destination
	destFile := filepath.Join(destDir, "file1.txt")
	require.NoError(t, os.WriteFile(destFile, []byte("original"), os.ModePerm))

	// Create test files in the provider
	files := []string{"file1.txt", "file2.txt"}
	for _, file := range files {
		filePath := filepath.Join(baseDir, bucket, prefix, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("new"), os.ModePerm))
	}

	// Try without overwrite first
	err := provider.DownloadDir(context.Background(), bucket, prefix, destDir, false)
	require.Error(t, err)
	data, err := os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "original", string(data), "File should not be overwritten when overwrite=false")

	// Now try with overwrite
	err = provider.DownloadDir(context.Background(), bucket, prefix, destDir, true)
	require.NoError(t, err)
	data, err = os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "new", string(data), "File should be overwritten when overwrite=true")
}
