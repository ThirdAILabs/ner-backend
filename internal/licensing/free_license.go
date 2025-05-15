package licensing

import (
	"context"
	"database/sql"
	"log/slog"
	"ner-backend/internal/database"
	"time"

	"gorm.io/gorm"
)

const DefaultFreeLicenseMaxBytes = 5 * 1024 * 1024 * 1024 // 5 GB

type FreeLicenseVerifier struct {
	db       *gorm.DB
	maxBytes int64
	timeNow  func() time.Time
}

func NewFreeLicenseVerifier(db *gorm.DB, maxBytes int64) *FreeLicenseVerifier {
	return &FreeLicenseVerifier{
		db:       db,
		maxBytes: maxBytes,
		timeNow:  time.Now,
	}
}

func (verifier *FreeLicenseVerifier) SetTimeNow(timeNow func() time.Time) {
	verifier.timeNow = timeNow
}

func (verifier *FreeLicenseVerifier) VerifyLicense(ctx context.Context) (LicenseType, LicenseInfo, error) {
	var totalBytes sql.NullInt64

	// Get first day of current month, we apply the quota on a monthly basis
	now := verifier.timeNow().UTC()
	currentMonth := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, time.UTC)

	if err := verifier.db.WithContext(ctx).Model(&database.InferenceTask{}).
		Select("SUM(total_size)").
		Where("creation_time >= ?", currentMonth).
		Scan(&totalBytes).Error; err != nil {
		slog.Error("error getting total usage", "error", err)
		return FreeLicense, nil, ErrLicenseVerificationFailed
	}

	info := LicenseInfo{
		"maxBytes":  verifier.maxBytes,
		"usedBytes": totalBytes.Int64,
	}

	if totalBytes.Valid && totalBytes.Int64 > verifier.maxBytes {
		return FreeLicense, info, ErrQuotaExceeded
	}

	return FreeLicense, info, nil
}
