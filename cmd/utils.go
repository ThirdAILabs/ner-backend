package cmd

import (
	"flag"
	"fmt"
	"log"
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
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
