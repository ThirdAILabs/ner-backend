package licensing

import (
	"context"
	"database/sql"
	"log/slog"
	"ner-backend/internal/database"

	"gorm.io/gorm"
)

const DefaultFreeLicenseMaxBytes = 1 * 1024 * 1024 * 1024 // 1 GB

type FreeLicenseVerifier struct {
	db       *gorm.DB
	maxBytes int
}

func NewFreeLicenseVerifier(db *gorm.DB, maxBytes int) *FreeLicenseVerifier {
	return &FreeLicenseVerifier{db: db, maxBytes: maxBytes}
}

func (verifier *FreeLicenseVerifier) VerifyLicense(ctx context.Context) error {
	var totalBytes sql.NullInt64
	if err := verifier.db.Debug().WithContext(ctx).Model(&database.InferenceTask{}).Select("SUM(total_size)").Scan(&totalBytes).Error; err != nil {
		slog.Error("error getting total usage", "error", err)
		return ErrLicenseVerificationFailed
	}

	if totalBytes.Valid && int(totalBytes.Int64) > verifier.maxBytes {
		return ErrQuotaExceeded
	}

	return nil
}
