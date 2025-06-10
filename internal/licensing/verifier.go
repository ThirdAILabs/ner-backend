package licensing

import (
	"context"
	"errors"
	"time"
)

var (
	ErrLicenseVerificationFailed = errors.New("license verification failed")
	ErrLicenseNotFound           = errors.New("license not found")
	ErrInvalidLicense            = errors.New("invalid license")
	ErrExpiredLicense            = errors.New("expired license")
	ErrQuotaExceeded             = errors.New("quota exceeded")
)

type LicenseType string

type LicenseUsage struct {
	MaxBytes  int64
	UsedBytes int64
}

type LicenseInfo struct {
	LicenseType LicenseType
	Expiry      *time.Time
	Usage       *LicenseUsage
}

const (
	LocalLicense  LicenseType = "local"
	KeygenLicense LicenseType = "keygen"
	FreeLicense   LicenseType = "free"
)

type LicenseVerifier interface {
	VerifyLicense(ctx context.Context) (LicenseInfo, error)
}
