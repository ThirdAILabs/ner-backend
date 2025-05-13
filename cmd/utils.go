package cmd

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
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

func initializeModel(
	ctx context.Context,
	db *gorm.DB,
	s3p *storage.S3Provider,
	bucket,
	name,
	modelType,
	localSubdir string,
	tags []string,
	skipIfExistsCount int,
	modelBase string,
) error {
	var model database.Model
	err := db.Where("name = ?", name).Preload("Tags").First(&model).Error
	isNew := errors.Is(err, gorm.ErrRecordNotFound)
	if err != nil && !isNew {
		return fmt.Errorf("error querying model %s: %w", name, err)
	}

	if isNew {
		model.Id = uuid.New()
		model.Name = name
		model.Type = modelType
		model.Status = database.ModelTrained
		model.CreationTime = time.Now()
		for _, tag := range tags {
			model.Tags = append(model.Tags, database.ModelTag{ModelId: model.Id, Tag: tag})
		}
		if err := db.Create(&model).Error; err != nil {
			return fmt.Errorf("failed to create model record %s: %w", name, err)
		}
	}

	s3Prefix := model.Id.String() + "/"
	localDir := filepath.Join(modelBase, localSubdir)

	info, err := os.Stat(localDir)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("local model dir does not exist, skipping upload", "dir", localDir)
			return nil
		}
		return fmt.Errorf("failed to stat local model dir %s: %w", localDir, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("local model dir %s is not a directory", localDir)
	}

	objs, err := s3p.ListObjects(ctx, bucket, s3Prefix)
	if err != nil {
		slog.Error("failed to list S3 objects for model", "model_id", model.Id, "error", err)
	} else if len(objs) > skipIfExistsCount {
		slog.Info("model already uploaded to S3, skipping upload", "model_id", model.Id)
		return nil
	}

	if err := s3p.UploadDir(ctx, bucket, model.Id.String(), localDir); err != nil {
		database.UpdateModelStatus(ctx, db, model.Id, database.ModelFailed) // nolint:errcheck
		slog.Warn("failed to upload model to S3", "model_id", model.Id, "error", err)
		return fmt.Errorf("error uploading model %s: %w", name, err)
	}
	slog.Info("successfully uploaded model to S3", "model_id", model.Id)
	return nil
}

func InitializeCnnNerExtractor(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string, hostModelDir string) error {
	return initializeModel(ctx, db, s3p, bucket,
		"advanced", "cnn", "cnn_model",
		commonModelTags, 0, hostModelDir,
	)
}

func InitializeTransformerModel(ctx context.Context, db *gorm.DB, s3p *storage.S3Provider, bucket string, hostModelDir string) error {
	// transformer may have multiple files; skip if >4 objects exist
	return initializeModel(ctx, db, s3p, bucket,
		"ultra", "transformer", "transformer_model",
		commonModelTags, 4, hostModelDir,
	)
}

func InitializeBoltModel(db *gorm.DB, s3 storage.Provider, modelBucket, name, localModelPath string) error {
	slog.Info("initializing bolt model", "model_name", name, "local_model_path", localModelPath)

	modelId := uuid.New()

	var modelTags []database.ModelTag
	for _, tag := range commonModelTags {
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

func NewModelLoaders(pythonExec, pluginScript string) map[string]core.ModelLoader {

	return map[string]core.ModelLoader{
		"bolt": func(modelDir string) (core.Model, error) {
			return bolt.LoadNER(filepath.Join(modelDir, "model.bin"))
		},
		"transformer": func(modelDir string) (core.Model, error) {
			cfgJSON := fmt.Sprintf(`{"model_path":"%s","threshold":0.5}`, modelDir)
			return python.LoadPythonModel(
				pythonExec,
				pluginScript,
				"python_combined_ner_model",
				cfgJSON,
			)
		},
		"cnn": func(modelDir string) (core.Model, error) {
			cfgJSON := fmt.Sprintf(`{"model_path":"%s/cnn_model.pth"}`, modelDir)
			return python.LoadPythonModel(
				pythonExec,
				pluginScript,
				"python_cnn_ner_model",
				cfgJSON,
			)
		},
		"presidio": func(_ string) (core.Model, error) {
			return core.NewPresidioModel()
		},
	}
}
