package storage

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestObjectStore(t *testing.T) (*LocalObjectStore, string) {
	t.Helper()
	dir := t.TempDir()
	objectStore, err := NewLocalObjectStore(dir)
	require.NoError(t, err)
	return objectStore, dir
}

func setupTestConnector(t *testing.T) (*LocalConnector, *LocalObjectStore, string) {
	t.Helper()
	objectStore, dir := setupTestObjectStore(t)
	connector := NewLocalConnector(
		LocalConnectorParams{	
			BaseDir: dir,
			Bucket:  "test-bucket",
			Prefix:  "test-prefix",
		},
	)
	return connector, objectStore, dir
}


func TestLocalObjectStore_PutObject(t *testing.T) {
	objectStore, baseDir := setupTestObjectStore(t)

	bucket := "test-bucket"
	key := "test-file.txt"
	content := []byte("Test content")

	err := objectStore.PutObject(context.Background(), bucket, key, bytes.NewReader(content))
	require.NoError(t, err)

	filePath := filepath.Join(baseDir, bucket, key)
	data, err := os.ReadFile(filePath)
	require.NoError(t, err)
	assert.Equal(t, content, data)
}

func TestLocalObjectStore_CreateBucket(t *testing.T) {
	objectStore, _ := setupTestObjectStore(t)

	bucket := "test-bucket"
	err := objectStore.CreateBucket(context.Background(), bucket)
	require.NoError(t, err)
	// CreateBucket is a no-op for LocalObjectStore, so we just verify it doesn't error
}

func TestLocalObjectStore_DeleteObjects(t *testing.T) {
	objectStore, baseDir := setupTestObjectStore(t)

	bucket := "test-bucket"
	prefix := "test-dir"
	
	// Create some test files
	files := []string{"test-dir/file1.txt", "test-dir/file2.txt", "other-dir/file3.txt"}
	for _, file := range files {
		filePath := filepath.Join(baseDir, bucket, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("content"), os.ModePerm))
	}

	err := objectStore.DeleteObjects(context.Background(), bucket, prefix)
	require.NoError(t, err)

	// Verify files in the prefix were deleted
	for _, file := range []string{"test-dir/file1.txt", "test-dir/file2.txt"} {
		filePath := filepath.Join(baseDir, bucket, file)
		_, err := os.Stat(filePath)
		assert.True(t, os.IsNotExist(err), "File %s should not exist", file)
	}

	// Verify files outside the prefix still exist
	otherFilePath := filepath.Join(baseDir, bucket, "other-dir/file3.txt")
	_, err = os.Stat(otherFilePath)
	assert.NoError(t, err, "File outside prefix should still exist")
}

func TestLocalObjectStore_UploadDir(t *testing.T) {
	objectStore, baseDir := setupTestObjectStore(t)

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

	err := objectStore.UploadDir(context.Background(), bucket, prefix, srcDir)
	require.NoError(t, err)

	// Verify files were uploaded by checking content
	for _, file := range files {
		uploadedPath := filepath.Join(baseDir, bucket, prefix, file)
		data, err := os.ReadFile(uploadedPath)
		require.NoError(t, err)
		assert.Equal(t, "content", string(data))
	}
}

func TestLocalObjectStore_DownloadDir(t *testing.T) {
	objectStore, baseDir := setupTestObjectStore(t)

	bucket := "test-bucket"
	prefix := "to-download"
	destDir := filepath.Join(t.TempDir(), "download-target")

	// Create test files in the object store
	files := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, file := range files {
		filePath := filepath.Join(baseDir, bucket, prefix, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("content"), os.ModePerm))
	}

	err := objectStore.DownloadDir(context.Background(), bucket, prefix, destDir, false)
	require.NoError(t, err)

	// Verify files were downloaded by checking content
	for _, file := range files {
		downloadedPath := filepath.Join(destDir, file)
		data, err := os.ReadFile(downloadedPath)
		require.NoError(t, err)
		assert.Equal(t, "content", string(data))
	}
}

func TestLocalObjectStore_DownloadDir_Overwrite(t *testing.T) {
	objectStore, baseDir := setupTestObjectStore(t)

	bucket := "test-bucket"
	prefix := "to-download"
	destDir := t.TempDir()

	destFile := filepath.Join(destDir, "file1.txt")
	require.NoError(t, os.WriteFile(destFile, []byte("original"), os.ModePerm))

	// Create test files in the object store
	files := []string{"file1.txt", "file2.txt"}
	for _, file := range files {
		filePath := filepath.Join(baseDir, bucket, prefix, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("new"), os.ModePerm))
	}

	// Try without overwrite first
	err := objectStore.DownloadDir(context.Background(), bucket, prefix, destDir, false)
	require.Error(t, err)
	data, err := os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "original", string(data), "File should not be overwritten when overwrite=false")

	// Now try with overwrite
	err = objectStore.DownloadDir(context.Background(), bucket, prefix, destDir, true)
	require.NoError(t, err)
	data, err = os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "new", string(data), "File should be overwritten when overwrite=true")
}

func TestLocalObjectStore_GetUploadConnector(t *testing.T) {
	objectStore, _ := setupTestObjectStore(t)

	bucket := "test-bucket"
	uploadId := uuid.New()

	connector, err := objectStore.GetUploadConnector(context.Background(), bucket, UploadParams{ UploadId: uploadId })
	require.NoError(t, err)
	require.NotNil(t, connector)
}

func TestLocalConnector_CreateInferenceTasks(t *testing.T) {
	connector, objectStore, _ := setupTestConnector(t)

	// Create some test files
	files := []string{"test-prefix/file1.txt", "test-prefix/file2.txt", "test-prefix/subdir/file3.txt"}
	for _, file := range files {
		err := objectStore.PutObject(context.Background(), "test-bucket", file, bytes.NewReader([]byte("content")))
		require.NoError(t, err)
	}

	targetBytes := int64(100) // Small target to force multiple tasks
	tasks, totalObjects, err := connector.CreateInferenceTasks(context.Background(), targetBytes)
	require.NoError(t, err)
	assert.Greater(t, len(tasks), 0)
	assert.Equal(t, int64(len(files)), totalObjects)

	// Verify task structure
	for _, task := range tasks {
		assert.Greater(t, task.TotalSize, int64(0))
		
		var taskParams LocalConnectorTaskParams
		err := json.Unmarshal(task.Params, &taskParams)
		require.NoError(t, err)
		assert.Greater(t, len(taskParams.ChunkKeys), 0)
	}
}

func TestLocalConnector_IterTaskChunks(t *testing.T) {
	connector, objectStore, _ := setupTestConnector(t)

	// Create test files with different content
	testFiles := map[string]string{
		"test-prefix/file1.txt": "Hello world",
		"test-prefix/file2.txt": "Test content",
	}

	for file, content := range testFiles {
		err := objectStore.PutObject(context.Background(), "test-bucket", file, bytes.NewReader([]byte(content)))
		require.NoError(t, err)
	}

	taskParams := LocalConnectorTaskParams{
		ChunkKeys: []string{"test-prefix/file1.txt", "test-prefix/file2.txt"},
	}
	paramsBytes, err := json.Marshal(taskParams)
	require.NoError(t, err)

	chunkStreams, err := connector.IterTaskChunks(context.Background(), paramsBytes)
	require.NoError(t, err)

	// Process the streams and verify its contents
	i := 0
	results := make(map[string]string)
	for stream := range chunkStreams {
		require.NoError(t, stream.Error)
		assert.Equal(t, taskParams.ChunkKeys[i], stream.Name)

		content := ""
		for chunk := range stream.Chunks {
			require.NoError(t, chunk.Error)
			content += chunk.Text
		}
		results[stream.Name] = content
		i++
	}

	for file, expectedContent := range testFiles {
		assert.Equal(t, expectedContent, results[file])
	}
}

func TestLocalConnector_IterTaskChunks_WithErrors(t *testing.T) {
	connector, _, _ := setupTestConnector(t)

	// Create task params with non-existent files
	taskParams := LocalConnectorTaskParams{
		ChunkKeys: []string{"non-existent-file.txt"},
	}
	paramsBytes, err := json.Marshal(taskParams)
	require.NoError(t, err)

	chunkStreams, err := connector.IterTaskChunks(context.Background(), paramsBytes)
	require.NoError(t, err)

	errorCount := 0
	for stream := range chunkStreams {
		if stream.Error != nil {
			errorCount++
		}
	}

	assert.Equal(t, 1, errorCount, "Should have one error for non-existent file")
}