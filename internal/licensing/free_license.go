package licensing

import (
	"context"
	"database/sql"
	"log/slog"
	"ner-backend/internal/database"

	"fmt"

	"gorm.io/gorm"
)

const DefaultFreeLicenseMaxBytes = 5 * 1024 * 1024 * 1024 // 5 GB

type FreeLicenseVerifier struct {
	db       *gorm.DB
	maxBytes int
}

func NewFreeLicenseVerifier(db *gorm.DB, maxBytes int) *FreeLicenseVerifier {
	return &FreeLicenseVerifier{db: db, maxBytes: maxBytes}
}

func (verifier *FreeLicenseVerifier) VerifyLicense(ctx context.Context) (LicenseType, LicenseInfo, error) {
	var totalBytes sql.NullInt64
	if err := verifier.db.WithContext(ctx).Model(&database.InferenceTask{}).Select("SUM(total_size)").Scan(&totalBytes).Error; err != nil {
		slog.Error("error getting total usage", "error", err)
		return FreeLicense, nil, ErrLicenseVerificationFailed
	}

	info := LicenseInfo{
		"max_bytes":  fmt.Sprint(verifier.maxBytes),
		"used_bytes": fmt.Sprint(totalBytes.Int64),
	}

	if totalBytes.Valid && int(totalBytes.Int64) > verifier.maxBytes {
		return FreeLicense, info, ErrQuotaExceeded
	}

	return FreeLicense, info, nil
}
