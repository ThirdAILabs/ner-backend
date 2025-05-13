package licensing_test

import (
	"context"
	"fmt"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func TestFreeLicensing(t *testing.T) {
	db, err := gorm.Open(sqlite.Open("file::memory:"), &gorm.Config{})
	require.NoError(t, err)
	require.NoError(t, db.AutoMigrate(&database.InferenceTask{}))

	maxBytes := 300

	verifier := licensing.NewFreeLicenseVerifier(db, maxBytes)

	licenseType, licenseInfo, err := verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprint(maxBytes), licenseInfo["maxBytes"])
	assert.Equal(t, "0", licenseInfo["usedBytes"])
	assert.Equal(t, licensing.FreeLicense, licenseType)

	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 200}).Error)
	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 50}).Error)

	licenseType, licenseInfo, err = verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprint(maxBytes), licenseInfo["maxBytes"])
	assert.Equal(t, "250", licenseInfo["usedBytes"])
	assert.Equal(t, licensing.FreeLicense, licenseType)

	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 100}).Error)

	licenseType, licenseInfo, err = verifier.VerifyLicense(context.Background())
	assert.ErrorIs(t, licensing.ErrQuotaExceeded, err)
	assert.Equal(t, fmt.Sprint(maxBytes), licenseInfo["maxBytes"])
	assert.Equal(t, "350", licenseInfo["usedBytes"])
	assert.Equal(t, licensing.FreeLicense, licenseType)
}
