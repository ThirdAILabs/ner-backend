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

type LicenseVerifier interface {
	VerifyLicense(ctx context.Context) error
}
