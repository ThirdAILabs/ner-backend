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
	s3p *storage.S3Provider,
	bucket,
	name,
	modelType,
	localDir string,
	tags []string,
) error {
	tags = filterExcludedTags(tags)
	var model database.Model
	result := db.
		Preload("Tags").
		Where("name = ?", name).
		Attrs(database.Model{
			Id:           uuid.New(),
			Name:         name,
			Type:         modelType,
			Status:       database.ModelQueued,
			CreationTime: time.Now(),
		}).
		FirstOrCreate(&model)

	if result.Error != nil {
		return fmt.Errorf("error finding or creating model %q: %w", name, result.Error)
	}

	if result.RowsAffected == 0 && model.Status == database.ModelTrained {
		slog.Info("model already exists, skipping initialization", "model_id", model.Id)
		return nil
	}

	if result.RowsAffected > 0 {
		modelTags := make([]database.ModelTag, len(tags))
		for i, tag := range tags {
			modelTags[i] = database.ModelTag{
				ModelId: model.Id,
				Tag:     tag,
			}
		}

		if err := db.Model(&model).Association("Tags").Replace(modelTags); err != nil {
			return fmt.Errorf("failed to attach tags to new model %q: %w", name, err)
		}
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
	database.UpdateModelStatus(ctx, db, model.Id, database.ModelTrained) // nolint:errcheck
	slog.Info("successfully uploaded model to S3", "model_id", model.Id)
	return nil
}

func InitializeCnnNerExtractor(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string, hostModelDir string) error {
	return initializeModel(ctx, db, s3p, bucket,
		"advanced", "cnn", filepath.Join(hostModelDir, "cnn_model"),
		commonModelTags,
	)
}

func InitializeTransformerModel(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string, hostModelDir string) error {
	return initializeModel(ctx, db, s3p, bucket,
		"ultra", "transformer", filepath.Join(hostModelDir, "transformer_model"),
		commonModelTags,
	)
}

func InitializeBoltModel(db *gorm.DB, s3 storage.Provider, modelBucket, name, localModelPath string) error {
	slog.Info("initializing bolt model", "model_name", name, "local_model_path", localModelPath)

	modelId := uuid.New()

	tags := filterExcludedTags(commonModelTags)

	var modelTags []database.ModelTag
	for _, tag := range tags {
		modelTags = append(modelTags, database.ModelTag{
			ModelId: modelId,
			Tag:     tag,
		})
	}

	var model database.Model

	result := db.
		Where(database.Model{Name: name}).
		Attrs(database.Model{
			Id:           modelId,
			Type:         "bolt",
			Status:       database.ModelQueued,
			CreationTime: time.Now().UTC(),
			Tags:         modelTags,
		}).
		FirstOrCreate(&model)

	if err := result.Error; err != nil {
		return fmt.Errorf("failed to create model record: %w", err)
	}

	if result.RowsAffected == 0 && model.Status == database.ModelTrained {
		// TODO: only overwrite the blob store if there are any changes to the model
		slog.Info("bolt model already exists in db, overwriting blob store", "model_id", model.Id)
		modelId = model.Id
	}

	status := database.ModelFailed
	defer func() {
		if err := database.UpdateModelStatus(context.Background(), db, modelId, status); err != nil {
			slog.Error("failed to update model status during initialization", "model_id", modelId, "status", status, "error", err)
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

	if err := s3.PutObject(context.Background(), modelBucket, filepath.Join(modelId.String(), "model.bin"), file); err != nil {
		slog.Error("failed to upload bolt model to S3", "model_id", modelId, "error", err)
		return fmt.Errorf("failed to upload model to S3: %w", err)
	}

	slog.Info("successfully uploaded bolt model to S3", "model_id", modelId)

	status = database.ModelTrained

	return nil
}

func InitializeOnnxModel(
	db *gorm.DB,
	s3 storage.Provider,
	bucket, name, modelDir string,
) error {
	slog.Info("initializing ONNX model", "model_name", name, "model_dir", modelDir)
	ctx := context.Background()
	modelId := uuid.New()

	// attach common tags
	tags := filterExcludedTags(commonModelTags)
	var mts []database.ModelTag
	for _, t := range tags {
		mts = append(mts, database.ModelTag{
			ModelId: modelId,
			Tag:     t,
		})
	}

	// create or find existing record
	var model database.Model
	res := db.
		Where(database.Model{Name: name}).
		Attrs(database.Model{
			Id:           modelId,
			Type:         "onnx",
			Status:       database.ModelQueued,
			CreationTime: time.Now().UTC(),
			Tags:         mts,
		}).
		FirstOrCreate(&model)
	if err := res.Error; err != nil {
		return fmt.Errorf("db create/find onnx model: %w", err)
	}
	if res.RowsAffected == 0 && model.Status == database.ModelTrained {
		slog.Info("onnx model already exists, skipping upload", "model_id", model.Id)
		return nil
	}

	// upload the .onnx file
	onnxPath := filepath.Join(modelDir, "model.onnx.enc")
	f, err := os.Open(onnxPath)
	if err != nil {
		return fmt.Errorf("open onnx file %q: %w", onnxPath, err)
	}
	defer f.Close()
	if err := s3.PutObject(ctx, bucket, filepath.Join(modelId.String(), "model.onnx.enc"), f); err != nil {
		return fmt.Errorf("upload onnx to s3: %w", err)
	}

	// upload transitions.json if present
	transPath := filepath.Join(modelDir, "transitions.json")
	if tf, err := os.Open(transPath); err == nil {
		defer tf.Close()
		if err := s3.PutObject(ctx, bucket, filepath.Join(modelId.String(), "transitions.json"), tf); err != nil {
			slog.Warn("failed to upload transitions.json", "error", err)
		}
	} else {
		slog.Warn("transitions.json not found; skipping", "path", transPath)
	}

	// mark as trained
	if err := database.UpdateModelStatus(ctx, db, modelId, database.ModelTrained); err != nil {
		return fmt.Errorf("update onnx model status: %w", err)
	}
	slog.Info("uploaded ONNX model to S3", "model_id", modelId)
	return nil
}

func CreateLicenseVerifier(db *gorm.DB, license string) licensing.LicenseVerifier {
	if strings.HasPrefix(license, "local:") {
		var err error
		licenseVerifier, err := licensing.NewFileLicenseVerifier([]byte(licensing.FileLicensePublicKey), strings.TrimSpace(strings.TrimPrefix(license, "local:")))
		if err != nil {
			log.Fatalf("Failed to create file license verifier: %v", err)
		}
		return licenseVerifier
	} else if license != "" {
		licenseVerifier, err := licensing.NewKeygenLicenseVerifier(license)
		if err != nil {
			log.Fatalf("Failed to create license verifier: %v", err)
		}
		return licenseVerifier
	} else {
		slog.Warn(fmt.Sprintf("License key not provided. Using free license verifier with %.2fGB total file size limit", float64(licensing.DefaultFreeLicenseMaxBytes)/(1024*1024*1024)))
		return licensing.NewFreeLicenseVerifier(db, licensing.DefaultFreeLicenseMaxBytes)
	}
}
