package licensing

import (
	"log/slog"
	"ner-backend/internal/database"

	"gorm.io/gorm"
)

type FreeLicenseVerifier struct {
	db       *gorm.DB
	maxBytes int
}

func NewFreeLicenseVerifier(db *gorm.DB, maxBytes int) *FreeLicenseVerifier {
	return &FreeLicenseVerifier{db: db, maxBytes: maxBytes}
}

func (verifier *FreeLicenseVerifier) VerifyLicense() error {
	var totalBytes int
	if err := verifier.db.Model(&database.InferenceTask{}).Select("SUM(total_size)").First(&totalBytes).Error; err != nil {
		slog.Error("error getting total usage", "error", err)
		return ErrLicenseVerificationFailed
	}

	if totalBytes > verifier.maxBytes {
		return ErrQuotaExceeded
	}

	return nil
}
