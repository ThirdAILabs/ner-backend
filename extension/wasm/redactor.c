#include <emscripten.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// String duplication function that works everywhere
static char* string_duplicate(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char* copy = malloc(len);
    if (copy) {
        memcpy(copy, s, len);
    }
    return copy;
}

// JavaScript interop function for HTTP requests
EM_JS(char*, http_post_sync, (const char* url, const char* json_body), {
    var urlStr = UTF8ToString(url);
    var bodyStr = UTF8ToString(json_body);
    
    try {
        var xhr = new XMLHttpRequest();
        xhr.open('POST', urlStr, false); // false for synchronous
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(bodyStr);
        
        if (xhr.status === 200) {
            var response = xhr.responseText;
            var lengthBytes = lengthBytesUTF8(response) + 1;
            var stringOnWasmHeap = _malloc(lengthBytes);
            stringToUTF8(response, stringOnWasmHeap, lengthBytes);
            return stringOnWasmHeap;
        } else {
            console.error('HTTP error! status: ' + xhr.status);
            var errorMsg = 'HTTP error! status: ' + xhr.status;
            var lengthBytes = lengthBytesUTF8(errorMsg) + 1;
            var stringOnWasmHeap = _malloc(lengthBytes);
            stringToUTF8(errorMsg, stringOnWasmHeap, lengthBytes);
            return stringOnWasmHeap;
        }
    } catch (error) {
        console.error('Network error:', error);
        var errorMsg = 'Network error: ' + error.message;
        var lengthBytes = lengthBytesUTF8(errorMsg) + 1;
        var stringOnWasmHeap = _malloc(lengthBytes);
        stringToUTF8(errorMsg, stringOnWasmHeap, lengthBytes);
        return stringOnWasmHeap;
    }
});

// Create JSON request body with proper escaping for large text
static char* create_json_body(const char* text) {
    if (!text) return NULL;
    
    size_t text_len = strlen(text);
    
    // Calculate maximum possible size needed:
    // - Each character could potentially be escaped (doubled)
    // - Add space for JSON structure: {"message":""}
    // - Add some padding for safety
    size_t max_escaped_len = text_len * 2;
    size_t json_overhead = 20; // {"message":""}
    size_t total_size = max_escaped_len + json_overhead + 1; // +1 for null terminator
    
    // Allocate memory for escaped text
    char* escaped_text = malloc(max_escaped_len + 1);
    if (!escaped_text) {
        printf("Failed to allocate memory for escaped text\n");
        return NULL;
    }
    
    // Escape JSON characters
    size_t j = 0;
    for (size_t i = 0; i < text_len && j < max_escaped_len - 1; i++) {
        if (text[i] == '"') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = '"';
        } else if (text[i] == '\\') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = '\\';
        } else if (text[i] == '\n') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = 'n';
        } else if (text[i] == '\r') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = 'r';
        } else if (text[i] == '\t') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = 't';
        } else {
            escaped_text[j++] = text[i];
        }
    }
    escaped_text[j] = '\0';
    
    // Allocate memory for final JSON body
    char* json_body = malloc(total_size);
    if (!json_body) {
        printf("Failed to allocate memory for JSON body\n");
        free(escaped_text);
        return NULL;
    }
    
    // Create JSON body
    snprintf(json_body, total_size, "{\"message\":\"%s\"}", escaped_text);
    
    free(escaped_text);
    return json_body;
}

// Parse JSON response to extract the Message field
static char* parse_message_from_json(const char* json_response) {
    if (!json_response) return NULL;
    
    // Simple JSON parsing to find "Message":"value"
    const char* message_key = "\"Message\":\"";
    const char* message_start = strstr(json_response, message_key);
    
    if (!message_start) {
        // Try alternative format with spaces
        message_key = "\"Message\": \"";
        message_start = strstr(json_response, message_key);
    }
    
    if (!message_start) return NULL;
    
    // Move past the key to the value
    message_start += strlen(message_key);
    
    // Find the closing quote
    const char* message_end = strchr(message_start, '"');
    if (!message_end) return NULL;
    
    // Calculate length and allocate memory
    int message_length = message_end - message_start;
    char* message = malloc(message_length + 1);
    
    if (message) {
        strncpy(message, message_start, message_length);
        message[message_length] = '\0';
    }
    
    return message;
}

// Initialize redactor (simplified since we're using HTTP API)
EMSCRIPTEN_KEEPALIVE
int init_redactor() {
    printf("Redactor initialized for HTTP API mode\n");
    return 1;
}

// Main redaction function using HTTP API
EMSCRIPTEN_KEEPALIVE
char* redact(const char* session_id, const char* text) {
    if (!text || !session_id) return NULL;
    
    printf("Redacting text: %s\n", text);
    printf("Redacting for session: %s\n", session_id);
    
    // Build the API URL
    char api_url[512];
    snprintf(api_url, sizeof(api_url), "http://localhost:16549/api/v1/chat/sessions/%s/redact", session_id);
    
    // Build JSON body using helper function
    char* json_body = create_json_body(text);
    if (!json_body) {
        printf("Failed to create JSON body\n");
        return string_duplicate(text);
    }
    
    printf("Making POST request to: %s\n", api_url);
    printf("Request body: %s\n", json_body);
    
    // Make the HTTP request
    char* response = http_post_sync(api_url, json_body);
    
    // Clean up JSON body
    free(json_body);
    
    if (!response) {
        printf("Failed to get response from server\n");
        return string_duplicate(text);
    }
    
    printf("Response: %s\n", response);
    
    // Parse the response to extract the Message field
    char* redacted_message = parse_message_from_json(response);
    free(response);
    
    if (!redacted_message) {
        printf("Failed to parse message from response\n");
        return string_duplicate(text);
    }
    
    printf("Redacted message: %s\n", redacted_message);
    
    return redacted_message;
}

// Simplified restore function (just returns text as-is since we use HTTP API)
EMSCRIPTEN_KEEPALIVE
char* restore(const char* session_id, const char* text) {
    if (!text || !session_id) return NULL;
    
    printf("Restoring text: %s\n", text);
    printf("Restoring for session: %s\n", session_id);
    
    // Build the API URL
    char api_url[512];
    snprintf(api_url, sizeof(api_url), "http://localhost:16549/api/v1/chat/sessions/%s/restore", session_id);
    
    // Build JSON body using helper function
    char* json_body = create_json_body(text);
    if (!json_body) {
        printf("Failed to create JSON body\n");
        return string_duplicate(text);
    }
    
    printf("Making POST request to: %s\n", api_url);
    printf("Request body: %s\n", json_body);
    
    // Make the HTTP request
    char* response = http_post_sync(api_url, json_body);
    
    // Clean up JSON body
    free(json_body);
    
    if (!response) {
        printf("Failed to get response from server\n");
        return string_duplicate(text);
    }
    
    printf("Response: %s\n", response);
    
    // Parse the response to extract the Message field
    char* restored_message = parse_message_from_json(response);
    free(response);
    
    if (!restored_message) {
        printf("Failed to parse message from response\n");
        return string_duplicate(text);
    }
    
    printf("Restored message: %s\n", restored_message);
    
    return restored_message;
}

// Clear redaction mappings (simplified - no local mappings anymore)
EMSCRIPTEN_KEEPALIVE
void clear_redaction_mappings() {
    // Nothing to clear since we don't store local mappings
    printf("Clear redaction mappings called\n");
}

// Get redaction statistics (simplified - no local stats)
EMSCRIPTEN_KEEPALIVE
char* get_redaction_stats() {
    char* stats = malloc(100);
    if (!stats) return NULL;
    
    // Return empty stats since we don't track local redactions
    snprintf(stats, 100, "{\"total_redactions\":0,\"redaction_categories\":[]}");
    
    return stats;
}

// Free memory allocated by functions
EMSCRIPTEN_KEEPALIVE
void free_string(char* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// Cleanup function (simplified)
EMSCRIPTEN_KEEPALIVE
void cleanup_redactor() {
    clear_redaction_mappings();
    printf("Redactor cleanup completed\n");
}
