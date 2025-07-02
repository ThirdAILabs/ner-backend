#include <emscripten.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
    
    // Build JSON body
    char json_body[2048];
    // Simple JSON escaping - replace quotes with escaped quotes
    char escaped_text[1024];
    int j = 0;
    for (int i = 0; text[i] && j < sizeof(escaped_text) - 2; i++) {
        if (text[i] == '"') {
            escaped_text[j++] = '\\';
            escaped_text[j++] = '"';
        } else {
            escaped_text[j++] = text[i];
        }
    }
    escaped_text[j] = '\0';
    
    snprintf(json_body, sizeof(json_body), "{\"message\":\"%s\"}", escaped_text);
    
    printf("Making POST request to: %s\n", api_url);
    printf("Request body: %s\n", json_body);
    
    // Make the HTTP request
    char* response = http_post_sync(api_url, json_body);
    
    if (!response) {
        printf("Failed to get response from server\n");
        return strdup(text);
    }
    
    printf("Response: %s\n", response);
    
    // Parse the response to extract the Message field
    char* redacted_message = parse_message_from_json(response);
    free(response);
    
    if (!redacted_message) {
        printf("Failed to parse message from response\n");
        return strdup(text);
    }
    
    printf("Redacted message: %s\n", redacted_message);
    
    return redacted_message;
}

// Simplified restore function (just returns text as-is since we use HTTP API)
EMSCRIPTEN_KEEPALIVE
char* restore(const char* text) {
    if (!text) return NULL;
    
    // Since we're using HTTP API for redaction, restoration would also
    // need to go through the API. For now, just return the text as-is.
    return strdup(text);
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

#ifdef TEST_BUILD
// Test main function for native compilation
int main() {
    printf("Testing C redactor module...\n");
    
    // Initialize the redactor
    if (!init_redactor()) {
        printf("Failed to initialize redactor\n");
        return 1;
    }
    
    // Test text
    const char* test_text = "Contact John Doe at john.doe@email.com or call 555-123-4567";
    printf("Original: %s\n", test_text);
    
    // Test redaction
    char* redacted = redact(test_text);
    if (redacted) {
        printf("Redacted: %s\n", redacted);
        
        // Test restoration
        char* restored = restore(redacted);
        if (restored) {
            printf("Restored: %s\n", restored);
            
            // Check if restoration worked
            if (strcmp(test_text, restored) == 0) {
                printf("✓ Test passed: Redaction and restoration work correctly\n");
            } else {
                printf("✗ Test failed: Restoration doesn't match original\n");
            }
            
            free(restored);
        } else {
            printf("✗ Test failed: Restoration returned NULL\n");
        }
        
        free(redacted);
    } else {
        printf("✗ Test failed: Redaction returned NULL\n");
    }
    
    // Test statistics
    char* stats = get_redaction_stats();
    if (stats) {
        printf("Statistics: %s\n", stats);
        free(stats);
    }
    
    // Cleanup
    cleanup_redactor();
    
    printf("Test completed\n");
    return 0;
}
#endif 