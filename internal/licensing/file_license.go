package licensing

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/gob"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"log/slog"
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
	publicKey      *rsa.PublicKey
	licensePayload Payload
}

func NewFileLicenseVerifier(publicKeyPem []byte, licenseStr string) (*FileLicenseVerifier, error) {
	publicKey, err := parseRsaPublicKey(publicKeyPem)
	if err != nil {
		return nil, fmt.Errorf("error parsing public key: %w", err)
	}

	payload, err := DecodeLicense(publicKey, licenseStr)
	if err != nil {
		return nil, err
	}

	slog.Info("file license initialized", "license_payload", payload)

	return &FileLicenseVerifier{
		publicKey:      publicKey,
		licensePayload: payload,
	}, nil
}

func (v *FileLicenseVerifier) VerifyLicense() error {
	if v.licensePayload.Expiration.Before(time.Now()) {
		return ErrExpiredLicense
	}

	return nil
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

const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwuEITPNWJHQwuEAl08AE
QxFG9mmVbJL+/WVASSyaw48ktTioHkGm3Y+uuA2ae0AREQL+xCyyPN2jm5X2IT+Y
EIW3xB4AZcLYrmzRC952Hz2PqC62HwST19HfUrZzIMukrXAozXPZn7LwwiVRDHwH
j/LUvf7ikQ9Sr6vuU1yE8HQH2YAHXd3H0fNERLWCm+GywlLHGoZ1Z9m1JWcp8y2I
D6icow4aAlK+OOUU/Eb1YwnTpFk0Lye4AO7TvWtO/4QO4nzFfz9s6NLV6MW8QkSQ
aRKkt5oWE9aJ27dP+dQdb+dWPTf99wNMdqDBHlDp3LWQYkhgHjEdMOgnQMb5Svth
eQIDAQAB
-----END PUBLIC KEY-----
`
