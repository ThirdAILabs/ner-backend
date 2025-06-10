package licensing_test

import (
	"context"
	"ner-backend/internal/database"
	"ner-backend/internal/licensing"
	"testing"
	"time"

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

	mayTime := time.Date(2025, 5, 14, 10, 0, 0, 0, time.UTC)
	var mockNow = mayTime

	verifier := licensing.NewFreeLicenseVerifier(db, 300)
	verifier.SetTimeNow(func() time.Time {
		return mockNow
	})

	licenseInfo, err := verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)
	assert.Equal(t, licensing.FreeLicense, licenseInfo.LicenseType)
	assert.Equal(t, int64(300), licenseInfo.Usage.MaxBytes)
	assert.Equal(t, int64(0), licenseInfo.Usage.UsedBytes)

	// Create tasks in May 2025
	require.NoError(t, db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TotalSize:    200,
		CreationTime: mayTime,
	}).Error)
	require.NoError(t, db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TotalSize:    50,
		CreationTime: mayTime,
	}).Error)

	_, err = verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	require.NoError(t, db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TotalSize:    100,
		CreationTime: mayTime,
	}).Error)

	// Should exceed quota in May
	_, err = verifier.VerifyLicense(context.Background())
	assert.ErrorIs(t, err, licensing.ErrQuotaExceeded)

	// Move clock to June 2025
	juneTime := time.Date(2025, 6, 1, 10, 0, 0, 0, time.UTC)
	mockNow = juneTime

	// Create a task in June that is equal to the quota
	require.NoError(t, db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TotalSize:    300,
		CreationTime: juneTime,
	}).Error)

	// Should not exceed quota since we're only counting June's tasks
	_, err = verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	// Create a task in june that exceeds the quota
	require.NoError(t, db.Create(&database.InferenceTask{
		ReportId:     uuid.New(),
		TotalSize:    1,
		CreationTime: juneTime,
	}).Error)

	// License check should fail because quota is exceeded in June
	licenseInfo, err = verifier.VerifyLicense(context.Background())
	assert.ErrorIs(t, err, licensing.ErrQuotaExceeded)
	assert.Equal(t, licensing.FreeLicense, licenseInfo.LicenseType)
	assert.Equal(t, int64(300), licenseInfo.Usage.MaxBytes)
	assert.Equal(t, int64(301), licenseInfo.Usage.UsedBytes)
}
