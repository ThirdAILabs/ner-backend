package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

type ReportRequest struct {
	CustomTags      map[string]interface{} `json:"CustomTags"`
	Groups          map[string]interface{} `json:"Groups"`
	ModelID         string                 `json:"ModelId"`
	S3Endpoint      string                 `json:"S3Endpoint"`
	S3Region        string                 `json:"S3Region"`
	SourceS3Bucket  string                 `json:"SourceS3Bucket"`
	SourceS3Prefix  string                 `json:"SourceS3Prefix"`
	Tags            []string               `json:"Tags"`
	ReportName      string                 `json:"report_name"`
}

type Model struct {
	Id     string `json:"Id"`
	Status string `json:"Status"`
}

type Report struct {
	Id string `json:"Id"`
}

func getFirstTrainedModelID() (string, error) {
	// Create the request
	req, err := http.NewRequest("GET", "http://localhost:16549/api/v1/models", nil)
	if err != nil {
		return "", fmt.Errorf("error creating request: %v", err)
	}

	// Set headers
	req.Header.Set("Accept", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %v", err)
	}

	// Parse the response
	var models []Model
	if err := json.Unmarshal(body, &models); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %v", err)
	}

	// Filter for trained models
	var trainedModels []Model
	for _, model := range models {
		if model.Status == "TRAINED" {
			trainedModels = append(trainedModels, model)
		}
	}

	if len(trainedModels) == 0 {
		return "", fmt.Errorf("no trained models found")
	}

	// Return the first trained model's ID
	return trainedModels[0].Id, nil
}

func listReports() ([]Report, error) {
	// Create the request
	req, err := http.NewRequest("GET", "http://localhost:16549/api/v1/reports", nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	// Set headers
	req.Header.Set("Accept", "application/json")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	// Parse the response
	var reports []Report
	if err := json.Unmarshal(body, &reports); err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %v", err)
	}

	return reports, nil
}

func deleteReport(reportID string) error {
	// Create the request
	req, err := http.NewRequest("DELETE", fmt.Sprintf("http://localhost:16549/api/v1/reports/%s", reportID), nil)
	if err != nil {
		return fmt.Errorf("error creating request: %v", err)
	}

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("error deleting report: status %d, body: %s", resp.StatusCode, string(body))
	}

	return nil
}

func main() {
	// List and delete existing reports
	reports, err := listReports()
	if err != nil {
		log.Fatalf("Error listing reports: %v", err)
	}

	fmt.Printf("Found %d existing reports\n", len(reports))
	for _, report := range reports {
		fmt.Printf("Deleting report %s...\n", report.Id)
		if err := deleteReport(report.Id); err != nil {
			log.Printf("Error deleting report %s: %v", report.Id, err)
		}
	}

	// Get the first trained model ID
	modelID, err := getFirstTrainedModelID()
	if err != nil {
		log.Fatalf("Error getting model ID: %v", err)
	}

	// Create the request body
	reqBody := ReportRequest{
		CustomTags:      map[string]interface{}{},
		Groups:          map[string]interface{}{},
		ModelID:         modelID,
		S3Endpoint:      "bigtable://licensing-370721",
		S3Region:        "customerservice",
		SourceS3Bucket:  "customerservice",
		SourceS3Prefix:  "call:transcript",
		Tags:            []string{"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE", "EMAIL", "ID_NUMBER", "LICENSE_PLATE", "LOCATION", "NAME", "PHONENUMBER", "SSN", "URL", "VIN"},
		ReportName:      "asdf",
	}

	// Convert request body to JSON
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		log.Fatalf("Error marshaling JSON: %v", err)
	}

	// Create the request
	req, err := http.NewRequest("POST", "http://localhost:16549/api/v1/reports", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Fatalf("Error creating request: %v", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/plain, */*")

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Fatalf("Error making request: %v", err)
	}
	defer resp.Body.Close()

	// Read and print the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Error reading response: %v", err)
	}

	fmt.Printf("Status Code: %d\n", resp.StatusCode)
	fmt.Printf("Response Body: %s\n", string(body))
} 