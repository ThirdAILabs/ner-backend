package licensing_test

import (
	"ner-backend/internal/licensing"
	"testing"
)

const (
	goodLicense                = "AC013F-FD0B48-00B160-64836E-76E88D-V3"
	expiredLicense             = "78BF4E-1EACCA-3432A5-D633E2-7B182B-V3"
	nonexistentLicense         = "000000-000000-000000-000000-000000-V3"
	suspendedLicense           = "9R3F-KLNJ-M3M4-KWLW-9E9E-7TNT-4FXH-V7R9"
	missingEntitlementsLicense = "94TN-9LUT-KXWK-K4VE-CPEW-3U9K-3R7H-HREL"
)

func TestKeygenLicensing(t *testing.T) {
	t.Run("GoodLicense", func(t *testing.T) {
		verifier, err := licensing.NewKeygenLicenseVerifier(goodLicense)
		if err != nil {
			t.Fatal(err)
		}
		if err := verifier.VerifyLicense(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("ExpiredLicense", func(t *testing.T) {
		_, err := licensing.NewKeygenLicenseVerifier(expiredLicense)
		if err != licensing.ErrExpiredLicense {
			t.Fatal(err)
		}
	})

	t.Run("NonexistentLicense", func(t *testing.T) {
		_, err := licensing.NewKeygenLicenseVerifier(nonexistentLicense)
		if err != licensing.ErrLicenseNotFound {
			t.Fatal(err)
		}
	})

	t.Run("SuspendedLicense", func(t *testing.T) {
		_, err := licensing.NewKeygenLicenseVerifier(suspendedLicense)
		if err != licensing.ErrExpiredLicense {
			t.Fatal(err)
		}
	})

	t.Run("MissingEntitlements", func(t *testing.T) {
		_, err := licensing.NewKeygenLicenseVerifier(missingEntitlementsLicense)
		if err != licensing.ErrInvalidLicense {
			t.Fatal(err)
		}
	})
}
