package licensing_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"ner-backend/internal/licensing"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileLicensing(t *testing.T) {
	privateKey, publicKey, err := licensing.GenerateKeys()
	require.NoError(t, err)

	t.Run("valid license", func(t *testing.T) {
		expiry := time.Now().Add(time.Hour)
		goodLicense, err := licensing.CreateLicense([]byte(privateKey), expiry)
		require.NoError(t, err)

		verifier := licensing.NewFileLicenseVerifier([]byte(publicKey), goodLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.NoError(t, err)
		assert.Equal(t, expiry, licenseInfo.Expiry)
		assert.Equal(t, licensing.LocalLicense, licenseInfo.LicenseType)
	})

	t.Run("expired license", func(t *testing.T) {
		expiredLicense, err := licensing.CreateLicense([]byte(privateKey), time.Now().Add(-time.Hour))
		require.NoError(t, err)

		verifier := licensing.NewFileLicenseVerifier([]byte(publicKey), expiredLicense)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, licensing.ErrExpiredLicense, err)
		assert.Equal(t, licensing.LocalLicense, licenseInfo.LicenseType)
	})

	t.Run("invalid license", func(t *testing.T) {
		original, err := licensing.CreateLicense([]byte(privateKey), time.Now().Add(time.Hour))
		require.NoError(t, err)

		licenseBytes, err := base64.StdEncoding.DecodeString(original)
		require.NoError(t, err)

		var corruptedLicense licensing.License
		require.NoError(t, gob.NewDecoder(bytes.NewReader(licenseBytes)).Decode(&corruptedLicense))

		corruptedLicense.Payload, err = json.Marshal(licensing.Payload{Expiration: time.Now().Add(2 * time.Hour)})
		require.NoError(t, err)

		buf := bytes.Buffer{}
		require.NoError(t, gob.NewEncoder(&buf).Encode(corruptedLicense))

		corruptedLicenseStr := base64.StdEncoding.EncodeToString(buf.Bytes())

		verifier := licensing.NewFileLicenseVerifier([]byte(publicKey), corruptedLicenseStr)
		licenseInfo, err := verifier.VerifyLicense(context.Background())
		assert.ErrorIs(t, licensing.ErrInvalidLicense, err)
		assert.Equal(t, licensing.LocalLicense, licenseInfo.LicenseType)
	})
}
