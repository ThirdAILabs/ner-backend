package licensing

import (
	"bytes"
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"time"
)

type Payload struct {
	Expiration time.Time
}

type License struct {
	Payload   []byte
	Signature []byte
}

type FileLicenseVerifier struct {
	publicKeyPem []byte
	licenseStr   string
}

func NewFileLicenseVerifier(publicKeyPem []byte, licenseStr string) *FileLicenseVerifier {

	return &FileLicenseVerifier{
		publicKeyPem: publicKeyPem,
		licenseStr:   licenseStr,
	}
}

func (v *FileLicenseVerifier) VerifyLicense(ctx context.Context) (LicenseInfo, error) {

	licenseInfo := LicenseInfo{
		LicenseType: LocalLicense,
	}

	publicKey, err := parseRsaPublicKey(v.publicKeyPem)
	if err != nil {
		return licenseInfo, ErrInvalidLicense
	}

	payload, err := DecodeLicense(publicKey, v.licenseStr)
	if err != nil {
		return licenseInfo, ErrInvalidLicense
	}

	if payload.Expiration.IsZero() {
		return licenseInfo, ErrInvalidLicense
	}

	if payload.Expiration.Before(time.Now().UTC()) {
		return licenseInfo, ErrExpiredLicense
	}

	licenseInfo.Expiry = &payload.Expiration

	return licenseInfo, nil
}

func DecodeLicense(publicKey *rsa.PublicKey, licenseStr string) (Payload, error) {
	licenseBytes, err := base64.StdEncoding.DecodeString(licenseStr)
	if err != nil {
		return Payload{}, ErrInvalidLicense
	}

	var license License
	if err := gob.NewDecoder(bytes.NewReader(licenseBytes)).Decode(&license); err != nil {
		return Payload{}, ErrInvalidLicense
	}

	if err := verifySignature(publicKey, license.Payload, license.Signature); err != nil {
		return Payload{}, ErrInvalidLicense
	}

	var payload Payload
	if err := json.Unmarshal(license.Payload, &payload); err != nil {
		return Payload{}, ErrInvalidLicense
	}

	return payload, nil
}

func CreateLicense(privateKeyPem []byte, expiration time.Time) (string, error) {
	privateKey, err := parseRsaPrivateKey(privateKeyPem)
	if err != nil {
		return "", fmt.Errorf("error parsing private key: %w", err)
	}

	payload := Payload{
		Expiration: expiration,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("error marshaling payload: %w", err)
	}

	signature, err := signMessage(privateKey, payloadBytes)
	if err != nil {
		return "", fmt.Errorf("error signing payload: %w", err)
	}

	license := License{
		Payload:   payloadBytes,
		Signature: signature,
	}

	licenseBytes := bytes.Buffer{}
	if err := gob.NewEncoder(&licenseBytes).Encode(license); err != nil {
		return "", fmt.Errorf("error encoding license: %w", err)
	}

	return base64.StdEncoding.EncodeToString(licenseBytes.Bytes()), nil
}

func GenerateKeys() ([]byte, []byte, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, fmt.Errorf("error generating private key: %w", err)
	}

	publicKey := &privateKey.PublicKey

	privateKeyBytes, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling private key: %w", err)
	}

	privateKeyPem, err := encodeToPem("PRIVATE KEY", privateKeyBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("error encoding private key to PEM: %w", err)
	}

	publicKeyBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling public key: %w", err)
	}
	publicKeyPem, err := encodeToPem("PUBLIC KEY", publicKeyBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("error encoding public key to PEM: %w", err)
	}

	return privateKeyPem, publicKeyPem, nil
}

func encodeToPem(blockType string, bytes []byte) ([]byte, error) {
	pemBlock := &pem.Block{
		Type:  blockType,
		Bytes: bytes,
	}

	pemBytes := pem.EncodeToMemory(pemBlock)
	if pemBytes == nil {
		return nil, fmt.Errorf("failed to encode PEM block")
	}

	return pemBytes, nil
}

func parseRsaPublicKey(pemBytes []byte) (*rsa.PublicKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM block")
	}

	publicKey, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("error parsing public key: %w", err)
	}

	rsaPublicKey, ok := publicKey.(*rsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("public key is not of type RSA")
	}

	return rsaPublicKey, nil
}

func parseRsaPrivateKey(pemBytes []byte) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM block")
	}

	privateKey, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("error parsing private key: %w", err)
	}

	rsaPrivateKey, ok := privateKey.(*rsa.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("private key is not of type RSA")
	}

	return rsaPrivateKey, nil
}

func signMessage(privateKey *rsa.PrivateKey, message []byte) ([]byte, error) {
	hash := crypto.SHA256.New()
	hash.Write(message)

	signature, err := rsa.SignPKCS1v15(rand.Reader, privateKey, crypto.SHA256, hash.Sum(nil))
	if err != nil {
		return nil, fmt.Errorf("error signing message: %w", err)
	}

	return signature, nil
}

func verifySignature(publicKey *rsa.PublicKey, message []byte, signature []byte) error {
	hash := crypto.SHA256.New()
	hash.Write(message)

	if err := rsa.VerifyPKCS1v15(publicKey, crypto.SHA256, hash.Sum(nil), signature); err != nil {
		return fmt.Errorf("signature verification failed: %w", err)
	}
	return nil
}

const FileLicensePublicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3F0ujjsyp5CtfRcSvVfr
JzUOqCbrOelAceR9JFSt7GDWcafTeaUXnA4gxN2UXSvr0//kYsKSmJHscTg9cZ+y
4WSg6BREI9opkwjeadvDECXJEGxkjtLHCp5RwOvkkehLYIAIvgHf6AY86/1T3hT3
YrVq/vETXqOA1bmEcjttybfnzdTjVQqxgPWdonXaLefLY/NZieMarSd49s/n993L
3gYaoh4oCHSwGZ/4UuYMyPCQM1J6YxcCG7rS6YjAiKziyChWjWfcPU3Crtkoe/L3
NosO9EHLcekjQwmYPHEufNbULymI+B+7cdJeH6U+hLG3V4VC7BxvvZMCko2XTPhg
2QIDAQAB
-----END PUBLIC KEY-----
`
