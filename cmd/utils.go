package cmd

import (
	"context"
	"errors"
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

func InitializePresidioModel(db *gorm.DB) {
	presidio, err := core.NewPresidioModel()
	if err != nil {
		log.Fatalf("Failed to initialize model: %v", err)
	}

	modelId := uuid.New()

	var tags []database.ModelTag
	for _, tag := range presidio.GetTags() {
		tags = append(tags, database.ModelTag{
			ModelId: modelId,
			Tag:     tag,
		})
	}

	var model database.Model

	if err := db.Where(database.Model{Name: "basic"}).Attrs(database.Model{
		Id:           modelId,
		Type:         "presidio",
		Status:       database.ModelTrained,
		CreationTime: time.Now(),
		Tags:         tags,
	}).FirstOrCreate(&model).Error; err != nil {
		log.Fatalf("Failed to create model record: %v", err)
	}
}

func InitializeCnnNerExtractor(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string) error {
	var model database.Model
	err := db.
		Where("name = ?", "advance").
		Preload("Tags").
		First(&model).Error

	isNew := errors.Is(err, gorm.ErrRecordNotFound)
	if err != nil && !isNew {
		return fmt.Errorf("error querying model: %w", err)
	}

	if isNew {
		model.Id = uuid.New()
		model.Name = "advance"
		model.Type = "cnn"
		model.Status = database.ModelTrained
		model.CreationTime = time.Now()

		modelTags := []string{
			"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
			"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
			"LOCATION", "NAME", "PHONENUMBER", "SERVICE_CODE", "SEXUAL_ORIENTATION",
			"SSN", "URL", "VIN", "O",
		}
		for _, tag := range modelTags {
			model.Tags = append(model.Tags, database.ModelTag{
				ModelId: model.Id,
				Tag:     tag,
			})
		}

		if err := db.Create(&model).Error; err != nil {
			return fmt.Errorf("failed to create model record: %w", err)
		}
	}

	s3Prefix := model.Id.String() + "/"

	// HOST_MODEL_DIR can be used to pass models if backend is running locally
	modelBaseDir := os.Getenv("HOST_MODEL_DIR")
	if modelBaseDir == "" {
		modelBaseDir = "/app/models"
	}
	localDir := filepath.Join(modelBaseDir, "cnn_model")

	info, err := os.Stat(localDir)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("local model dir does not exist, skipping upload", "dir", localDir)
			return nil
		}
		return fmt.Errorf("failed to stat local model dir %s: %w", localDir, err)
	}
	if !info.IsDir() {
		slog.Warn("local model path exists but is not a directory, skipping upload", "path", localDir)
		return nil
	}

	objs, err := s3p.ListObjects(ctx, bucket, s3Prefix)
	if err != nil {
		slog.Error("failed to list S3 objects for model", "model_id", model.Id, "error", err)
		// we could choose to abort or continue; here we continue and try upload
	} else if len(objs) > 0 {
		slog.Info("model already uploaded to S3, skipping upload", "model_id", model.Id)
		return nil
	}

	if err := s3p.UploadDir(ctx, bucket, model.Id.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, db, model.Id, database.ModelFailed) //nolint:errcheck
		slog.Warn("failed to upload model to S3", "model_id", model.Id, "error", err)
		return nil
	}
	slog.Info("successfully uploaded model to S3", "model_id", model.Id)
	return nil
}

func InitializeTransformerModel(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string) error {
	var model database.Model
	err := db.
		Where("name = ?", "ultra").
		Preload("Tags").
		First(&model).Error

	isNew := errors.Is(err, gorm.ErrRecordNotFound)
	if err != nil && !isNew {
		return fmt.Errorf("error querying model: %w", err)
	}

	if isNew {
		model.Id = uuid.New()
		model.Name = "ultra"
		model.Type = "transformer"
		model.Status = database.ModelTrained
		model.CreationTime = time.Now()

		modelTags := []string{
			"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
			"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
			"LOCATION", "NAME", "PHONENUMBER", "SERVICE_CODE", "SEXUAL_ORIENTATION",
			"SSN", "URL", "VIN", "O",
		}
		for _, tag := range modelTags {
			model.Tags = append(model.Tags, database.ModelTag{
				ModelId: model.Id,
				Tag:     tag,
			})
		}

		if err := db.Create(&model).Error; err != nil {
			return fmt.Errorf("failed to create model record: %w", err)
		}
	}

	s3Prefix := model.Id.String() + "/"

	// HOST_MODEL_DIR can be used to pass models if backend is running locally
	modelBaseDir := os.Getenv("HOST_MODEL_DIR")
	if modelBaseDir == "" {
		modelBaseDir = "/app/models"
	}
	localDir := filepath.Join(modelBaseDir, "transformer_model")

	info, err := os.Stat(localDir)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("local model dir does not exist, skipping upload", "dir", localDir)
			return nil
		}
		return fmt.Errorf("failed to stat local model dir %s: %w", localDir, err)
	}
	if !info.IsDir() {
		slog.Warn("local model path exists but is not a directory, skipping upload", "path", localDir)
		return nil
	}

	objs, err := s3p.ListObjects(ctx, bucket, s3Prefix)
	if err != nil {
		slog.Error("failed to list S3 objects for model", "model_id", model.Id, "error", err)
		// we could choose to abort or continue; here we continue and try upload
	} else if len(objs) > 4 {
		slog.Info("model already uploaded to S3, skipping upload", "model_id", model.Id)
		return nil
	}

	if err := s3p.UploadDir(ctx, bucket, model.Id.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, db, model.Id, database.ModelFailed) //nolint:errcheck
		return fmt.Errorf("error uploading model to S3: %w", err)
	}
	slog.Info("successfully uploaded model to S3", "model_id", model.Id)
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
