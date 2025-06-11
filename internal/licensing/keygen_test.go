package licensing_test

import (
	"context"
	"ner-backend/internal/licensing"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

const (
	goodLicense                = "AC013F-FD0B48-00B160-64836E-76E88D-V3"
	expiredLicense             = "78BF4E-1EACCA-3432A5-D633E2-7B182B-V3"
	nonexistentLicense         = "000000-000000-000000-000000-000000-V3"
	suspendedLicense           = "9R3F-KLNJ-M3M4-KWLW-9E9E-7TNT-4FXH-V7R9"
	missingEntitlementsLicense = "6E8D1E-B1AD6A-F8D318-5BDF53-02295A-V3"
)

func TestKeygenLicensing(t *testing.T) {
	t.Run("GoodLicense", func(t *testing.T) {
		verifier := licensing.NewKeygenLicenseVerifier(goodLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, licensing.KeygenLicense, licenseInfo.LicenseType)
		assert.NotNil(t, licenseInfo.Expiry)
		assert.True(t, licenseInfo.Expiry.After(time.Now()))
	})

	t.Run("ExpiredLicense", func(t *testing.T) {
		verifier := licensing.NewKeygenLicenseVerifier(expiredLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, err, licensing.ErrExpiredLicense)
		assert.Equal(t, licensing.KeygenLicense, licenseInfo.LicenseType)
		assert.NotNil(t, licenseInfo.Expiry)
		assert.True(t, licenseInfo.Expiry.Before(time.Now()))
	})

	t.Run("NonexistentLicense", func(t *testing.T) {
		verifier := licensing.NewKeygenLicenseVerifier(nonexistentLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, err, licensing.ErrLicenseNotFound)
		assert.Equal(t, licensing.KeygenLicense, licenseInfo.LicenseType)
	})

	t.Run("SuspendedLicense", func(t *testing.T) {
		verifier := licensing.NewKeygenLicenseVerifier(suspendedLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, err, licensing.ErrExpiredLicense)
		assert.Equal(t, licensing.KeygenLicense, licenseInfo.LicenseType)
		assert.NotNil(t, licenseInfo.Expiry)
	})

	t.Run("MissingEntitlements", func(t *testing.T) {
		verifier := licensing.NewKeygenLicenseVerifier(missingEntitlementsLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, err, licensing.ErrInvalidLicense)
		assert.Equal(t, licensing.KeygenLicense, licenseInfo.LicenseType)
	})
}
