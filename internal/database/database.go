package database

import (
	"fmt"
	"log"
	"net/url"
	"strings"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func UriToDsn(uri string) string {
	parts, err := url.Parse(uri)
	if err != nil {
		log.Fatalf("error parsing db uri: %v", err)
	}
	pwd, _ := parts.User.Password()
	dbname := strings.TrimPrefix(parts.Path, "/")

	if dbname != "" {
		dbname = "dbname=" + dbname
	}

	return fmt.Sprintf("host=%v user=%v password=%v %v port=%v", parts.Hostname(), parts.User.Username(), pwd, dbname, parts.Port())
}

func NewDatabase(uri string) (*gorm.DB, error) {
	db, err := gorm.Open(postgres.Open(UriToDsn(uri)), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("error opening db: %w", err)
	}

	migrator := GetMigrator(db)

	if err := migrator.Migrate(); err != nil {
		return nil, fmt.Errorf("error migrating db: %w", err)
	}

	return db, nil
}
