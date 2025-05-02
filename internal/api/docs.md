# API Documentation

## Health Check

### GET /health
**Description:**  
Check the health status of the service.

**Response:**  
- `200 OK`: Service is healthy.

**Example Response:**
```json
{}
```

---

## Models

### GET /models
**Description:**  
Retrieve a list of all models.

**Response:**  
- `200 OK`: Returns a list of models.
- `500 Internal Server Error`: Error retrieving model records.

**Example Response:**
```json
[
  {
    "Id": "123e4567-e89b-12d3-a456-426614174000",
    "Name": "Model A",
    "Type": "regex",
    "Status": "TRAINED"
  },
  {
    "Id": "123e4567-e89b-12d3-a456-426614174001",
    "Name": "Model B",
    "Type": "bolt",
    "Status": "TRAINING"
  }
]
```

---

### GET /models/{model_id}
**Description:**  
Retrieve details of a specific model by its ID.

**Path Parameters:**  
- `model_id` (UUID): The ID of the model.

**Response:**  
- `200 OK`: Returns the model details.
- `404 Not Found`: Model not found.
- `500 Internal Server Error`: Error retrieving model record.

**Example Response:**
```json
{
  "Id": "123e4567-e89b-12d3-a456-426614174000",
  "Name": "Model A",
  "Type": "bolt",
  "Status": "TRAINED",
  "Tags": ["PHONENUMBER", "SSN"]
}
```

---

## Reports

### GET /reports
**Description:**  
Retrieve a list of all reports. Either `UploadId` or `SourceS3Bucket` must be specified, where `UploadId` is the id returned from the `POST /uploads` endpoint.

**Response:**  
- `200 OK`: Returns a list of reports.
- `500 Internal Server Error`: Error retrieving report records.

**Example Response:**
```json
[
  {
    "Id": "123e4567-e89b-12d3-a456-426614174002",
    "Model": {
      "Id": "123e4567-e89b-12d3-a456-426614174000",
      "Name": "Model A",
      "Type": "bolt",
      "Status": "TRAINED"
    },
    "UploadId": "123e4567-e89b-12d3-a456-426614174000",
    "SourceS3Bucket": "example-bucket",
    "SourceS3Prefix": "data/",
    "CreationTime": "2023-01-01T12:00:00Z",
    "Groups": [
      {
        "Id": "123e4567-e89b-12d3-a456-426614174003",
        "Name": "Group1",
        "Query": "COUNT(label1) > 3 AND COUNT(label2) = 2"
      }
    ]
  }
]
```

---

### POST /reports
**Description:**  
Create a new report.

**Request Body:**  
- `ModelId` (UUID): The ID of the model.
- `SourceS3Bucket` (string): The S3 bucket containing the source data.
- `SourceS3Prefix` (string, optional): The S3 prefix for the source data.
- `Groups` (map[string]string): A map of group names to queries.

**Example Request Body:**
```json
{
  "ModelId": "123e4567-e89b-12d3-a456-426614174000",
  "SourceS3Bucket": "example-bucket",
  "SourceS3Prefix": "data/",
  "Groups": {
    "Group1": "COUNT(label1) > 3 AND COUNT(label2) = 2",
    "Group2": "label3 CONTAINS 'xyz'"
  },
  "Tags": ["PHONENUMBER", "SSN"]
}
```

**Response:**  
- `201 Created`: Returns the created report ID.
- `422 Unprocessable Entity`: Missing required fields or invalid query.
- `404 Not Found`: Model not found.
- `500 Internal Server Error`: Error creating the report.

**Example Response:**
```json
{
  "ReportId": "123e4567-e89b-12d3-a456-426614174002"
}
```

---

### GET /reports/{report_id}
**Description:**  
Retrieve details of a specific report by its ID.

**Path Parameters:**  
- `report_id` (UUID): The ID of the report.

**Response:**  
- `200 OK`: Returns the report details.
- `404 Not Found`: Report not found.
- `500 Internal Server Error`: Error retrieving report data.

**Example Response:**
```json
{
  "Id": "123e4567-e89b-12d3-a456-426614174002",
  "Model": {
    "Id": "123e4567-e89b-12d3-a456-426614174000",
    "Name": "Model A",
    "Type": "bolt",
    "Status": "TRAINED"
  },
  "SourceS3Bucket": "example-bucket",
  "SourceS3Prefix": "data/",
  "CreationTime": "2023-01-01T12:00:00Z",
  "Groups": [
    {
      "Id": "123e4567-e89b-12d3-a456-426614174003",
      "Name": "Group1",
      "Query": "COUNT(label1) > 3 AND COUNT(label2) = 2"
    }
  ],
  "ShardDataTaskStatus": "COMPLETED",
  "InferenceTaskStatuses": {
    "COMPLETED": {
      "TotalTasks": 10,
      "TotalSize": 1000
    },
    "RUNNING": {
      "TotalTasks": 4,
      "TotalSize": 400
    },
    "QUEUED": {
      "TotalTasks": 5,
      "TotalSize": 500
    },
    "FAILED": {
      "TotalTasks": 1,
      "TotalSize": 100
    },
    
  }
}
```

---

### GET /reports/{report_id}/groups/{group_id}
**Description:**  
Retrieve details of a specific group within a report.

**Path Parameters:**  
- `report_id` (UUID): The ID of the report.
- `group_id` (UUID): The ID of the group.

**Response:**  
- `200 OK`: Returns the group details.
- `404 Not Found`: Group not found.
- `500 Internal Server Error`: Error retrieving group data.

**Example Response:**
```json
{
  "Id": "123e4567-e89b-12d3-a456-426614174003",
  "Name": "Group1",
  "Query": "COUNT(label1) > 3 AND COUNT(label2) = 2",
  "Objects": ["Object1", "Object2"]
}
```

---

### GET /reports/{report_id}/entities
**Description:**  
Retrieve entities for a specific report.

**Path Parameters:**  
- `report_id` (UUID): The ID of the report.

**Query Parameters:**  
- `offset` (int, optional): The starting point for pagination (default: 0).
- `limit` (int, optional): The maximum number of entities to return (default: 100, max: 200).
- `object` (string, optional): Filter entities by object.

**Response:**  
- `200 OK`: Returns a list of entities.
- `400 Bad Request`: Invalid query parameters.
- `500 Internal Server Error`: Error retrieving entities.

**Example Response:**
```json
[
  {
    "Object": "Object1",
    "Start": 0,
    "End": 5,
    "Label": "Person",
    "Text": "John",
    "LContext": "my name is",
    "RContext": "and my",
  },
  {
    "Object": "Object2",
    "Start": 10,
    "End": 15,
    "Label": "Organization",
    "Text": "Acme Corp",
    "LContext": "I work at",
    "RContext": "in new york",
  }
]
```

---

### GET /reports/{report_id}/search?query=<...>
**Description:**  
Searches for a the objects in a report that match a given query.

**Path Parameters:**  
- `report_id` (UUID): The ID of the report.

**Query Parameters:**  
- `query` (string, required): The query to run to filter the objects in the report. 

**Response:**  
- `200 OK`: Returns a list of entities.
- `400 Bad Request`: Invalid query parameters.
- `500 Internal Server Error`: Error retrieving entities.

**Example Response:**
```json
{
  "Objects": [
    "object 1", 
    "object 2",
  ]
}
```
--- 

### POST /uploads
**Description:**  
Upload files to the server.

**Request Body:**  
The request must be a `multipart/form-data` containing the files to upload. Each file should be included under the `files` form field.

**Response:**  
- `200 Created`: Returns the upload ID.
- `400 Bad Request`: Invalid or missing `Content-Type` header or multipart boundary.
- `422 Unprocessable Entity`: Invalid filename detected (e.g., empty filename).
- `500 Internal Server Error`: Error saving the file.

**Example Request:**  
```http
POST /uploads HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="files"; filename="example.txt"
Content-Type: text/plain

<file content>
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

**Example Response:**  
```json
{
  "Id": "123e4567-e89b-12d3-a456-426614174000",
}
```