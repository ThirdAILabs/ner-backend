package database

import (
	"log"
	"ner-backend/internal/database/versions"

	"github.com/go-gormigrate/gormigrate/v2"
	"gorm.io/gorm"
)

func GetMigrator(db *gorm.DB) *gormigrate.Gormigrate {
	migrator := gormigrate.New(db, gormigrate.DefaultOptions, []*gormigrate.Migration{
		{
			ID:      "0",
			Migrate: versions.Migration0,
		},
		{
			ID:       "1",
			Migrate:  versions.Migration1,
			Rollback: versions.RollbackMigration1,
		},
		{
			ID:       "1",
			Migrate:  versions.Migration2,
			Rollback: versions.RollbackMigration2,
		},
	})

	migrator.InitSchema(func(txn *gorm.DB) error {
		// This is run by the migrator if no previous migration is detected. It
		// allows it to bypass running all the migrations sequentially and just create
		// the latest database state.

		log.Println("clean database detected, running full schema initialization")

		return db.AutoMigrate(
			&Model{}, &ModelTag{}, &Report{}, &ReportTag{}, &CustomTag{}, &ShardDataTask{}, &InferenceTask{}, &Group{}, &ObjectGroup{}, &ObjectEntity{}, &ReportError{}, &ObjectPreview{}, &ChatHistory{}, &ChatSession{},
		)
	})

	return migrator
}
