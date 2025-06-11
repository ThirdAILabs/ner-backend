package cmd

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"ner-backend/internal/storage"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/joho/godotenv"
	"gorm.io/gorm"
)

func LoadEnvFile() {
	var configPath string

	flag.StringVar(&configPath, "env", "", "path to load env from")
	flag.Parse()

	if configPath == "" {
		log.Printf("no env file specified, using os.Environ only")
		return
	}

	log.Printf("loading env from file %s", configPath)
	err := godotenv.Load(configPath)
	if err != nil {
		log.Fatalf("error loading .env file '%s': %v", configPath, err)
	}
}

func RemoveExcludedTagsFromAllModels(db *gorm.DB) error {
	excludedTags := make([]string, 0, len(core.ExcludedTags))
	for tag := range core.ExcludedTags {
		excludedTags = append(excludedTags, tag)
	}

	if err := db.
		Where("tag IN ?", excludedTags).
		Delete(&database.ModelTag{}).Error; err != nil {
		return fmt.Errorf("failed to remove excluded tags %v: %w", excludedTags, err)
	}

	log.Printf("Removed excluded tags from model_tags: %v", excludedTags)
	return nil
}

func InitializePresidioModel(db *gorm.DB) {
	presidio, err := core.NewPresidioModel()
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}

	modelId := uuid.New()

	var tags []database.ModelTag
	for _, tag := range presidio.GetTags() {
		if _, exists := core.ExcludedTags[tag]; !exists {
			tags = append(tags, database.ModelTag{
				ModelId: modelId,
				Tag:     tag,
			})
		}
	}

	var model database.Model

	if err := db.Where(database.Model{Name: "basic"}).Attrs(database.Model{
		Id:           modelId,
		Type:         "presidio",
		Status:       database.ModelTrained,
		CreationTime: time.Now().UTC(),
		Tags:         tags,
	}).FirstOrCreate(&model).Error; err != nil {
		log.Fatalf("Failed to create model record: %v", err)
	}
}

var commonModelTags = []string{
	"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
	"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
	"LOCATION", "NAME", "PHONENUMBER", "SERVICE_CODE", "SEXUAL_ORIENTATION",
	"SSN", "URL", "VIN", "O",
}

func filterExcludedTags(tags []string) []string {
	var filteredTags []string
	for _, tag := range tags {
		if _, exists := core.ExcludedTags[tag]; !exists {
			filteredTags = append(filteredTags, tag)
		}
	}
	return filteredTags
}

func initializeModel(
	ctx context.Context,
	db *gorm.DB,
	s3p storage.Provider,
	bucket,
	name,
	modelType,
	localDir string,
	tags []string,
) error {
	tags = filterExcludedTags(tags)
	var model database.Model

	// The following section overwrite attributes of an existing model if it exists,
	// or creates a new one if it doesn't. This is to handle the case of model type
	// changes when we build our desktop application.
	// First get the existing model by name
	result := db.Preload("Tags").Where("name = ?", name).First(&model)
	if result.Error != nil && result.Error != gorm.ErrRecordNotFound {
		return fmt.Errorf("error finding model %q: %w", name, result.Error)
	}

	if result.Error == gorm.ErrRecordNotFound {
		model = database.Model{
			Id:           uuid.New(),
			Name:         name,
			Type:         modelType,
			Status:       database.ModelQueued,
			CreationTime: time.Now(),
		}
		// Create the model
		if err := db.Create(&model).Error; err != nil {
			return fmt.Errorf("failed to create model: %w", err)
		}
	} else {
		slog.Info("model already exists, will overwrite", "model_id", model.Id)
		// Update existing model
		result = db.Model(&model).Updates(database.Model{
			Type:         modelType,
			Status:       database.ModelQueued,
			CreationTime: time.Now(),
		})
		if result.Error != nil {
			return fmt.Errorf("failed to update model: %w", result.Error)
		}
	}

	// Update tags
	modelTags := make([]database.ModelTag, len(tags))
	for i, tag := range tags {
		modelTags[i] = database.ModelTag{
			ModelId: model.Id,
			Tag:     tag,
		}
	}
	if err := db.Model(&model).Association("Tags").Replace(modelTags); err != nil {
		return fmt.Errorf("failed to update tags for model %q: %w", name, err)
	}

	info, err := os.Stat(localDir)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("local model dir does not exist, skipping upload", "dir", localDir)
			return nil
		}
		database.UpdateModelStatus(ctx, db, model.Id, database.ModelFailed) // nolint:errcheck
		return fmt.Errorf("failed to stat local model dir %s: %w", localDir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("local model dir %s is not a directory", localDir)
	}

	if err := s3p.UploadDir(ctx, bucket, model.Id.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, db, model.Id, database.ModelFailed) // nolint:errcheck
		slog.Warn("failed to upload model to S3", "model_id", model.Id, "error", err)
		return fmt.Errorf("error uploading model %s: %w", name, err)
	}

	if err := database.UpdateModelStatus(ctx, db, model.Id, database.ModelTrained); err != nil {
		slog.Error("failed to update model status after upload", "model_id", model.Id, "error", err)
		return fmt.Errorf("failed to update model status after upload: %w", err)
	}

	slog.Info("successfully uploaded model to S3", "model_id", model.Id)
	return nil
}

func InitializeCnnNerExtractor(ctx context.Context, db *gorm.DB, s3p storage.Provider, bucket, name, hostModelDir string) error {
	slog.Info("initializing CNN model", "model_name", "cnn_model", "local_model_path", filepath.Join(hostModelDir, "cnn_model"))
	return initializeModel(ctx, db, s3p, bucket,
		name, "cnn", filepath.Join(hostModelDir, "cnn_model"),
		commonModelTags,
	)
}

func InitializeTransformerModel(ctx context.Context, db *gorm.DB, s3p storage.Provider, bucket, name, hostModelDir string) error {
	return initializeModel(ctx, db, s3p, bucket,
		name, "transformer", filepath.Join(hostModelDir, "transformer_model"),
		commonModelTags,
	)
}

func InitializeBoltModel(db *gorm.DB, s3 storage.Provider, modelBucket, name, hostModelDir string) error {
	localModelPath := filepath.Join(hostModelDir, "udt_model", "udt_complete.model")
	slog.Info("initializing bolt model", "model_name", name, "local_model_path", localModelPath)

	tags := filterExcludedTags(commonModelTags)
	var model database.Model

	// The following section overwrite attributes of an existing model if it exists,
	// or creates a new one if it doesn't. This is to handle the case of model type
	// changes when we build our desktop application.
	// First get the existing model by name
	result := db.Where(database.Model{Name: name}).First(&model)
	if result.Error != nil && result.Error != gorm.ErrRecordNotFound {
		return fmt.Errorf("failed to query existing model: %w", result.Error)
	}

	if result.Error == gorm.ErrRecordNotFound {
		model = database.Model{
			Id:           uuid.New(),
			Name:         name,
			Type:         "bolt",
			Status:       database.ModelQueued,
			CreationTime: time.Now().UTC(),
		}
		// Create the model
		if err := db.Create(&model).Error; err != nil {
			return fmt.Errorf("failed to create model: %w", err)
		}
	} else {
		slog.Info("bolt model already exists, will overwrite", "model_id", model.Id)
		// Update existing model
		result = db.Model(&model).Updates(database.Model{
			Type:         "bolt",
			Status:       database.ModelQueued,
			CreationTime: time.Now().UTC(),
		})
		if result.Error != nil {
			return fmt.Errorf("failed to update model: %w", result.Error)
		}
	}

	// Update tags
	modelTags := make([]database.ModelTag, len(tags))
	for i, tag := range tags {
		modelTags[i] = database.ModelTag{
			ModelId: model.Id,
			Tag:     tag,
		}
	}
	if err := db.Model(&model).Association("Tags").Replace(modelTags); err != nil {
		return fmt.Errorf("failed to update model tags: %w", err)
	}

	status := database.ModelFailed
	defer func() {
		if err := database.UpdateModelStatus(context.Background(), db, model.Id, status); err != nil {
			slog.Error("failed to update model status during initialization", "model_id", model.Id, "status", status, "error", err)
		}
	}()

	if info, err := os.Stat(localModelPath); err != nil {
		if os.IsNotExist(err) {
			slog.Error("local model path for bolt model does not exist, skipping upload", "path", localModelPath)
			return fmt.Errorf("local model path does not exist: %v", err)
		}
	} else if info.IsDir() {
		slog.Error("local model path for bolt model exists but is a directory, skipping upload", "path", localModelPath)
		return fmt.Errorf("local model path exists but is a directory: %v", err)
	}

	file, err := os.Open(localModelPath)
	if err != nil {
		slog.Error("failed to open local model file", "path", localModelPath, "error", err)
		return fmt.Errorf("failed to open local model file: %w", err)
	}
	defer file.Close()

	if err := s3.PutObject(context.Background(), modelBucket, filepath.Join(model.Id.String(), "model.bin"), file); err != nil {
		slog.Error("failed to upload bolt model to S3", "model_id", model.Id, "error", err)
		return fmt.Errorf("failed to upload model to S3: %w", err)
	}

	slog.Info("successfully uploaded bolt model to S3", "model_id", model.Id)
	status = database.ModelTrained

	return nil
}

func InitializeOnnxModel(
	db *gorm.DB,
	s3 storage.Provider,
	bucket, name, modelDir string,
) error {
	return initializeModel(
		context.Background(), db, s3, bucket,
		name, "onnx", filepath.Join(modelDir, "onnx_model"), commonModelTags,
	)
}

func CreateLicenseVerifier(db *gorm.DB, license string) licensing.LicenseVerifier {
	if strings.HasPrefix(license, "local:") {
		return licensing.NewFileLicenseVerifier([]byte(licensing.FileLicensePublicKey), strings.TrimSpace(strings.TrimPrefix(license, "local:")))
	} else if license != "" {
		return licensing.NewKeygenLicenseVerifier(license)
	} else {
		slog.Warn(fmt.Sprintf("License key not provided. Using free license verifier with %.2fGB total file size limit", float64(licensing.DefaultFreeLicenseMaxBytes)/(1024*1024*1024)))
		return licensing.NewFreeLicenseVerifier(db, licensing.DefaultFreeLicenseMaxBytes)
	}
}
