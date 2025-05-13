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
		Type:         "basic",
		Status:       database.ModelTrained,
		CreationTime: time.Now().UTC(),
		Tags:         tags,
	}).FirstOrCreate(&model).Error; err != nil {
		log.Fatalf("Failed to create model record: %v", err)
	}
}

func InitializeBoltModel(db *gorm.DB, s3 storage.Provider, modelBucket, name, localModelPath string) error {
	slog.Info("initializing bolt model", "model_name", name, "local_model_path", localModelPath)

	tags := []string{"SSN", "PHONENUMBER", "CREDIT_SCORE", "LOCATION", "SEXUAL_ORIENTATION",
		"VIN", "NAME", "URL", "ETHNICITY", "CARD_NUMBER", "SERVICE_CODE", "ADDRESS", "COMPANY",
		"DATE", "EMAIL", "GENDER", "LICENSE_PLATE", "ID_NUMBER", "O"}

	modelId := uuid.New()

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

	if result.RowsAffected == 0 {
		slog.Info("bolt model already exists, skipping initialization", "model_id", modelId)
		return nil
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
