package licensing

import (
	"context"
	"errors"
)

var (
	ErrLicenseVerificationFailed = errors.New("license verification failed")
	ErrLicenseNotFound           = errors.New("license not found")
	ErrInvalidLicense            = errors.New("invalid license")
	ErrExpiredLicense            = errors.New("expired license")
	ErrQuotaExceeded             = errors.New("quota exceeded")
)

type LicenseType string
type LicenseInfo map[string]string

const (
	LocalLicense   LicenseType = "local"
	KeygenLicense  LicenseType = "keygen"
	FreeLicense    LicenseType = "free"
	InvalidLicense LicenseType = ""
)

type LicenseVerifier interface {
	VerifyLicense(ctx context.Context) (LicenseType, LicenseInfo, error)
}
