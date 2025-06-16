package database

import (
	"log"
	"log/slog"
	"ner-backend/internal/database/versions/migration_0"
	"ner-backend/internal/database/versions/migration_1"
	"ner-backend/internal/database/versions/migration_2"
	"ner-backend/internal/database/versions/migration_3"

	"github.com/go-gormigrate/gormigrate/v2"
	"gorm.io/gorm"
)

func GetMigrator(db *gorm.DB) *gormigrate.Gormigrate {
	migrator := gormigrate.New(db, gormigrate.DefaultOptions, []*gormigrate.Migration{
		{
			ID:      "0",
			Migrate: migration_0.Migration,
		},
		{
			ID:       "1",
			Migrate:  migration_1.Migration,
			Rollback: migration_1.Rollback,
		},
		{
			ID:       "2",
			Migrate:  migration_2.Migration,
			Rollback: migration_2.Rollback,
		},
		{
			ID:       "3",
			Migrate:  migration_3.Migration,
			Rollback: migration_3.Rollback,
		},
	})

	migrator.InitSchema(func(txn *gorm.DB) error {
		// This is run by the migrator if no previous migration is detected. It
		// allows it to bypass running all the migrations sequentially and just create
		// the latest database state.

		log.Println("clean database detected, running full schema initialization")

		dbType := db.Dialector.Name()
		if dbType == "sqlite" || dbType == "sqlite3" {
			// Sqlite does not enable foreign key constraints by default, so we need to enable them manually.
			if err := txn.Exec("PRAGMA foreign_keys = ON").Error; err != nil {
				slog.Error("error enabling foreign keys for SQLite", "error", err)
			}
		}

		return db.AutoMigrate(
			&Model{}, &ModelTag{}, &Report{}, &ReportTag{}, &CustomTag{}, &ShardDataTask{}, &InferenceTask{}, &Group{}, &ObjectGroup{}, &ObjectEntity{}, &ReportError{}, &ObjectPreview{}, &ChatHistory{}, &ChatSession{}, &FileNameToPath{},
		)
	})

	return migrator
}
