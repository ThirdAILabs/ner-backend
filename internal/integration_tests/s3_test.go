package integrationtests

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"ner-backend/internal/storage"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	bucketName = "test-bucket"
	subdir     = "test-subdir"
)

func setupTestObjectStore(t *testing.T, ctx context.Context) (*storage.S3ObjectStore, string) {
	t.Helper()

	endpoint := setupMinioContainer(t, ctx)

	objectStore, err := storage.NewS3ObjectStore(bucketName, storage.S3ClientConfig{
		Endpoint:        endpoint,
		AccessKeyID:     minioUsername,
		SecretAccessKey: minioPassword,
	})
	require.NoError(t, err)
	return objectStore, endpoint
}

func setupTestConnector(t *testing.T, ctx context.Context) (*storage.S3ObjectStore, *storage.S3Connector) {
	t.Helper()
	objectStore, endpoint := setupTestObjectStore(t, ctx)

	connector, err := storage.NewS3ConnectorWithAccessKey(
		ctx,
		storage.S3ConnectorParams{
			Endpoint: endpoint,
			Bucket:   bucketName,
			Prefix:   subdir,
		},
		minioUsername,
		minioPassword,
	)
	require.NoError(t, err)

	return objectStore, connector
}

func TestS3ObjectStore_PutObject(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	key := "test-dir/test-file.txt"
	content := []byte("Test content")

	err := objectStore.PutObject(ctx, key, bytes.NewReader(content))
	require.NoError(t, err)

	obj, err := objectStore.GetObject(ctx, key)
	require.NoError(t, err)
	defer obj.Close()

	data, err := io.ReadAll(obj)
	require.NoError(t, err)
	assert.Equal(t, content, data)
}

func TestS3ObjectStore_DeleteObjects(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	prefix := "test-dir"

	// Create some test files
	files := []string{"test-dir/file1.txt", "test-dir/subdir/file2.txt", "other-dir/file3.txt"}
	for _, file := range files {
		require.NoError(t, objectStore.PutObject(ctx, file, bytes.NewReader([]byte("content: "+file))))
	}

	objs, err := objectStore.ListObjects(ctx, prefix)
	require.NoError(t, err)
	assert.Len(t, objs, 2)

	require.NoError(t, objectStore.DeleteObjects(context.Background(), prefix))

	newObjs, err := objectStore.ListObjects(ctx, prefix)
	require.NoError(t, err)
	assert.Len(t, newObjs, 0)
}

func TestS3ObjectStore_UploadDir(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	srcDir := t.TempDir()
	dest := "uploaded"

	// Create test files in the source directory
	files := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, file := range files {
		filePath := filepath.Join(srcDir, file)
		require.NoError(t, os.MkdirAll(filepath.Dir(filePath), os.ModePerm))
		require.NoError(t, os.WriteFile(filePath, []byte("content: "+file), os.ModePerm))
	}

	err := objectStore.UploadDir(context.Background(), srcDir, dest)
	require.NoError(t, err)

	// Verify files were uploaded by checking content
	for _, file := range files {
		uploadedPath := filepath.Join(dest, file)

		obj, err := objectStore.GetObject(ctx, uploadedPath)
		require.NoError(t, err)
		defer obj.Close()

		data, err := io.ReadAll(obj)
		require.NoError(t, err)
		assert.Equal(t, "content: "+file, string(data))
	}
}

func TestS3ObjectStore_DownloadDir(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	src := "to-download"
	destDir := filepath.Join(t.TempDir(), "download-target")

	// Create test files in the object store
	files := []string{"file1.txt", "file2.txt", "subdir/file3.txt"}
	for _, file := range files {
		require.NoError(t, objectStore.PutObject(ctx, filepath.Join(src, file), strings.NewReader("content: "+file)))
	}

	err := objectStore.DownloadDir(context.Background(), src, destDir, false)
	require.NoError(t, err)

	// Verify files were downloaded by checking content
	for _, file := range files {
		downloadedPath := filepath.Join(destDir, file)
		data, err := os.ReadFile(downloadedPath)
		require.NoError(t, err)
		assert.Equal(t, "content: "+file, string(data))
	}
}

func TestS3ObjectStore_DownloadDir_Overwrite(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	src := "to-download"
	destDir := t.TempDir()

	destFile := filepath.Join(destDir, "file1.txt")
	require.NoError(t, os.WriteFile(destFile, []byte("original"), os.ModePerm))

	// Create test files in the object store
	files := []string{"file1.txt", "file2.txt"}
	for _, file := range files {
		require.NoError(t, objectStore.PutObject(ctx, filepath.Join(src, file), strings.NewReader("new content")))
	}

	// Try without overwrite first
	err := objectStore.DownloadDir(context.Background(), src, destDir, false)
	require.Error(t, err)
	data, err := os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "original", string(data), "File should not be overwritten when overwrite=false")

	// Now try with overwrite
	err = objectStore.DownloadDir(context.Background(), src, destDir, true)
	require.NoError(t, err)
	data, err = os.ReadFile(destFile)
	require.NoError(t, err)
	assert.Equal(t, "new content", string(data), "File should be overwritten when overwrite=true")
}

func TestS3ObjectStore_GetUploadConnector(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, _ := setupTestObjectStore(t, ctx)

	uploadDir := "test-uploads"
	uploadId := uuid.New()

	connector, err := objectStore.GetUploadConnector(context.Background(), uploadDir, storage.UploadParams{UploadId: uploadId})
	require.NoError(t, err)
	require.NotNil(t, connector)
}

func TestS3Connector_CreateInferenceTasks(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, connector := setupTestConnector(t, ctx)

	// Create some test files
	files := []string{"test-subdir/file1.txt", "test-subdir/file2.txt", "test-subdir/subdir/file3.txt"}
	for _, file := range files {
		require.NoError(t, objectStore.PutObject(ctx, file, bytes.NewReader(make([]byte, 20))))
	}

	targetBytes := int64(45) // Small target to force multiple tasks
	tasks, totalObjects, err := connector.CreateInferenceTasks(context.Background(), targetBytes)
	require.NoError(t, err)
	assert.Len(t, tasks, 2)
	assert.Equal(t, int64(len(files)), totalObjects)

	allChunkKeys := []string{}
	// Verify task structure
	for _, task := range tasks {
		assert.Greater(t, task.TotalSize, int64(0))

		var taskParams storage.S3ConnectorTaskParams
		err := json.Unmarshal(task.Params, &taskParams)
		require.NoError(t, err)
		assert.Greater(t, len(taskParams.ChunkKeys), 0)

		allChunkKeys = append(allChunkKeys, taskParams.ChunkKeys...)
	}

	assert.ElementsMatch(t, files, allChunkKeys)
}

func TestS3Connector_IterTaskChunks(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	objectStore, connector := setupTestConnector(t, ctx)

	// Create test files with different content
	testFiles := map[string]string{
		"test-subdir/file1.txt": "Hello world",
		"test-subdir/file2.txt": "Test content",
	}

	for file, content := range testFiles {
		err := objectStore.PutObject(context.Background(), file, bytes.NewReader([]byte(content)))
		require.NoError(t, err)
	}

	taskParams := storage.S3ConnectorTaskParams{
		ChunkKeys: []string{"test-subdir/file1.txt", "test-subdir/file2.txt"},
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

func TestS3Connector_IterTaskChunks_WithErrors(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	_, connector := setupTestConnector(t, ctx)

	// Create task params with non-existent files
	taskParams := storage.S3ConnectorTaskParams{
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
		for chunk := range stream.Chunks {
			if chunk.Error != nil {
				errorCount++
			}
		}
	}

	assert.Equal(t, 1, errorCount, "Should have one error for non-existent file")
}
