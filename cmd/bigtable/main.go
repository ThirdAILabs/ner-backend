package main

import (
	"context"
	"io/ioutil"
	"log"
	"path/filepath"
	"strings"

	"cloud.google.com/go/bigtable"
)

const (
	tableName        = "customerservice"
	columnFamilyName = "call"
	columnName       = "transcript"
	dataDir         = "/Users/benitogeordie/datasets/mock-big-table/customer-service-2"
)

func sliceContains(list []string, target string) bool {
	for _, s := range list {
		if s == target {
			return true
		}
	}
	return false
}

func main() {
	project := "licensing-370721"
	instance := "customerservice"
	
	ctx := context.Background()

	adminClient, err := bigtable.NewAdminClient(ctx, project, instance)
	if err != nil {
		log.Fatalf("Could not create admin client: %v", err)
	}

	tables, err := adminClient.Tables(ctx)
	if err != nil {
		log.Fatalf("Could not fetch table list: %v", err)
	}

	if !sliceContains(tables, tableName) {
		log.Printf("Creating table %s", tableName)
		if err := adminClient.CreateTable(ctx, tableName); err != nil {
			log.Fatalf("Could not create table %s: %v", tableName, err)
		}
	}

	tblInfo, err := adminClient.TableInfo(ctx, tableName)
	if err != nil {
		log.Fatalf("Could not read info for table %s: %v", tableName, err)
	}

	if !sliceContains(tblInfo.Families, columnFamilyName) {
		if err := adminClient.CreateColumnFamily(ctx, tableName, columnFamilyName); err != nil {
			log.Fatalf("Could not create column family %s: %v", columnFamilyName, err)
		}
	}

	client, err := bigtable.NewClient(ctx, project, instance)
	if err != nil {
		log.Fatalf("Could not create data operations client: %v", err)
	}

	tbl := client.Open(tableName)

	// Delete all existing rows
	log.Printf("Deleting all existing rows")
	if err := adminClient.DropAllRows(ctx, tableName); err != nil {
		log.Fatalf("Could not delete rows: %v", err)
	}

	// Read all files from the directory
	files, err := ioutil.ReadDir(dataDir)
	if err != nil {
		log.Fatalf("Could not read directory: %v", err)
	}

	// Prepare mutations for bulk write
	muts := make([]*bigtable.Mutation, 0, len(files))
	rowKeys := make([]string, 0, len(files))

	log.Printf("Reading files and preparing mutations")
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".txt" {
			content, err := ioutil.ReadFile(filepath.Join(dataDir, file.Name()))
			if err != nil {
				log.Printf("Warning: Could not read file %s: %v", file.Name(), err)
				continue
			}

			mut := bigtable.NewMutation()
			mut.Set(columnFamilyName, columnName, bigtable.Now(), content)

			muts = append(muts, mut)
			rowKeys = append(rowKeys, columnName + strings.TrimSuffix(file.Name(), ".txt"))
		}
	}

	// Apply mutations in bulk
	log.Printf("Writing %d rows to table", len(muts))
	rowErrs, err := tbl.ApplyBulk(ctx, rowKeys, muts)
	if err != nil {
		log.Fatalf("Could not apply bulk row mutation: %v", err)
	}
	if rowErrs != nil {
		for _, rowErr := range rowErrs {
			log.Printf("Error writing row: %v", rowErr)
		}
		log.Fatalf("Could not write some rows")
	}

	log.Printf("Successfully inserted %d rows", len(muts))

	if err = client.Close(); err != nil {
		log.Fatalf("Could not close data operations client: %v", err)
	}

	if err = adminClient.Close(); err != nil {
		log.Fatalf("Could not close admin client: %v", err)
	}
}