package licensing

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/go-resty/resty/v2"
)

type KeygenVerifier struct {
	client     *resty.Client
	licenseKey string
}

func NewKeygenLicenseVerifier(licenseKey string) (*KeygenVerifier, error) {
	verifier := &KeygenVerifier{
		client:     resty.New().SetBaseURL("https://api.keygen.sh"),
		licenseKey: licenseKey,
	}

	if _, _, err := verifier.VerifyLicense(context.Background()); err != nil {
		return nil, err
	}

	return verifier, nil
}

type licenseVerifyResponse struct {
	Meta struct {
		Valid    bool   `json:"valid"`
		Constant string `json:"constant"`
		Detail   string `json:"detail"`
	} `json:"meta"`
}

func (verifier *KeygenVerifier) VerifyLicense(ctx context.Context) (LicenseType, LicenseInfo, error) {
	rb := map[string]map[string]any{
		"meta": {
			"key": verifier.licenseKey,
			"scope": map[string]any{
				"entitlements": []string{"FULL_ACCESS"},
			},
		},
	}

	context, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	endpoint := "/v1/accounts/thirdai/licenses/actions/validate-key"
	res, err := verifier.client.R().
		SetContext(context).
		SetHeader("Content-Type", "application/vnd.api+json").
		SetHeader("Accept", "application/vnd.api+json").
		SetBody(rb).
		Post(endpoint)

	if err != nil {
		slog.Error("unable to verify license with keygen", "error", err)
		return InvalidLicense, nil, ErrLicenseVerificationFailed
	}

	if !res.IsSuccess() {
		slog.Error("keygen returned error", "status_code", res.StatusCode(), "body", res.String())
		return InvalidLicense, nil, ErrLicenseVerificationFailed
	}

	body := res.Body()

	var verified licenseVerifyResponse
	if err := json.Unmarshal(body, &verified); err != nil {
		slog.Error("error parsing response from keygen", "error", err)
		return InvalidLicense, nil, ErrLicenseVerificationFailed
	}

	if !verified.Meta.Valid {
		switch verified.Meta.Constant {
		case "NOT_FOUND":
			return InvalidLicense, nil, ErrLicenseNotFound
		case "EXPIRED", "SUSPENDED":
			return InvalidLicense, nil, ErrExpiredLicense
		default:
			slog.Error("keygen verification error", "code", verified.Meta.Constant, "error", verified.Meta.Detail)
			return InvalidLicense, nil, ErrInvalidLicense
		}
	}

	if err := verifyResponseSignature(res, endpoint); err != nil {
		slog.Error("unable to verify response signature", "error", err)
		return InvalidLicense, nil, ErrInvalidLicense
	}

	return KeygenLicense, nil, nil
}

func parseSignature(res *resty.Response) ([]byte, error) {
	header := res.Header().Get("keygen-signature")
	_, sig, _ := strings.Cut(header, `signature="`)
	sig, _, _ = strings.Cut(sig, `", `)

	sigBytes, err := base64.StdEncoding.DecodeString(sig)
	if err != nil {
		return nil, fmt.Errorf("error response decoding signature: %w", err)
	}
	return sigBytes, nil
}

func parsePublicKey() (ed25519.PublicKey, error) {
	keyDER, err := base64.StdEncoding.DecodeString(publicKeyBase64)
	if err != nil {
		return nil, fmt.Errorf("error decoding public key: %w", err)
	}

	pubKey, err := x509.ParsePKIXPublicKey(keyDER)
	if err != nil {
		return nil, fmt.Errorf("error parsing DER encoded public key: %w", err)
	}

	ed25519PubKey, ok := pubKey.(ed25519.PublicKey)
	if !ok {
		return nil, fmt.Errorf("public key is not of type ed25519")
	}
	return ed25519PubKey, nil
}

const signingDataTemplate = `(request-target): %s %s
host: api.keygen.sh
date: %s
digest: sha-256=%s`

const publicKeyBase64 = "MCowBQYDK2VwAyEAmtv9iB02PTHBVsNImWiS3QGDp+RUDcABy3wu7Fp5Zq4="

// This verifies that the response originated at the keygen server by verifying
// the response signature using the public key corresponding to the private key
// used by keygen.
// https://keygen.sh/docs/api/signatures/#response-signatures
func verifyResponseSignature(res *resty.Response, endpoint string) error {
	bodyHash := sha256.Sum256(res.Body())
	bodyHashBase64 := base64.StdEncoding.EncodeToString(bodyHash[:])

	signingData := fmt.Sprintf(signingDataTemplate, strings.ToLower(res.Request.Method), endpoint, res.Header().Get("date"), bodyHashBase64)

	publicKey, err := parsePublicKey()
	if err != nil {
		return err
	}

	signature, err := parseSignature(res)
	if err != nil {
		return err
	}

	if !ed25519.Verify(publicKey, []byte(signingData), signature) {
		return fmt.Errorf("signature verification failed")
	}

	return nil
}
