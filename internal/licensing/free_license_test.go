package licensing_test

import (
	"context"
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

	verifier := licensing.NewFreeLicenseVerifier(db, 300)

	_, _, err = verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 200}).Error)
	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 50}).Error)

	_, _, err = verifier.VerifyLicense(context.Background())
	assert.NoError(t, err)

	require.NoError(t, db.Create(&database.InferenceTask{ReportId: uuid.New(), TotalSize: 100}).Error)

	_, _, err = verifier.VerifyLicense(context.Background())
	assert.ErrorIs(t, licensing.ErrQuotaExceeded, err)
}
