package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"ner-backend/internal/licensing"
	"os"
	"time"
)

func generateKeys(output string) {
	privateKey, publicKey, err := licensing.GenerateKeys()
	if err != nil {
		log.Fatalf("Error generating keys: %v", err)
	}

	privateKeyFile := output + "_private_key.pem"
	if err := os.WriteFile(privateKeyFile, []byte(privateKey), 0644); err != nil {
		log.Fatalf("Error writing private key to file '%s': %v", privateKeyFile, err)
	}

	publicKeyFile := output + "_public_key.pem"
	if err := os.WriteFile(publicKeyFile, []byte(publicKey), 0644); err != nil {
		log.Fatalf("Error writing public key to file '%s': %v", publicKeyFile, err)
	}
}

func createLicense(privateKeyPath string, days int) {
	expiration := time.Now().UTC().AddDate(0, 0, days)

	privateKeyPem, err := os.ReadFile(privateKeyPath)
	if err != nil {
		log.Fatalf("error reading private key file: %v", err)
	}

	license, err := licensing.CreateLicense(privateKeyPem, expiration)
	if err != nil {
		log.Fatalf("Error creating license: %v", err)
	}

	fmt.Println(license)
}

func validateLicense(publicKeyPath, license string) {
	publicKeyPem, err := os.ReadFile(publicKeyPath)
	if err != nil {
		log.Fatalf("error reading private key file: %v", err)
	}

	verifier := licensing.NewFileLicenseVerifier(publicKeyPem, license)

	if _, err := verifier.VerifyLicense(context.Background()); err != nil {
		log.Fatalf("License verification failed: %v", err)
	}
}

func main() {
	keysArgs := flag.NewFlagSet("keys", flag.ExitOnError)
	output := keysArgs.String("output", "", "Name of output files for the generated keys")

	createArgs := flag.NewFlagSet("create", flag.ExitOnError)
	privateKey := createArgs.String("private-key", "", "Path to private key file")
	days := createArgs.Int("days", 0, "Days until expiration")

	validateArgs := flag.NewFlagSet("validate", flag.ExitOnError)
	publicKey := validateArgs.String("public-key", "", "Path to public key file")
	license := validateArgs.String("license", "", "License key to validate")

	if len(os.Args) < 2 {
		log.Fatalf("expected 'keys' or 'create' or 'validate' subcommands")
	}

	switch os.Args[1] {
	case "keys":
		if err := keysArgs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("Error parsing arguments: %v", err)
		}
		generateKeys(*output)
	case "create":
		if err := createArgs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("Error parsing arguments: %v", err)
		}
		createLicense(*privateKey, *days)
	case "validate":
		if err := validateArgs.Parse(os.Args[2:]); err != nil {
			log.Fatalf("Error parsing arguments: %v", err)
		}
		validateLicense(*publicKey, *license)
	}
}
