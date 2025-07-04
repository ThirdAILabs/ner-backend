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
static char* create_json_body(const char* key, const char* text) {
    if (!text) return NULL;
    
    size_t text_len = strlen(text);
    
    // Calculate maximum possible size needed:
    size_t max_escaped_len = text_len * 2;
    size_t json_overhead = 10 + strlen(key); // {"key":""}
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
    snprintf(json_body, total_size, "{\"%s\":\"%s\"}", key, escaped_text);
    
    free(escaped_text);
    return json_body;
}

// Unescape JSON characters - reverse of what create_json_body does
static char* unescape_json_text(const char* escaped_text) {
    if (!escaped_text) return NULL;
    size_t escaped_len = strlen(escaped_text);
    char* unescaped = malloc(escaped_len + 1); // Unescaped will be same size or smaller
    
    if (!unescaped) {
        printf("Failed to allocate memory for unescaping\n");
        return NULL;
    }
    
    size_t j = 0;
    for (size_t i = 0; i < escaped_len; i++) {
        if (escaped_text[i] == '\\' && i + 1 < escaped_len) {
            // Handle escape sequences
            switch (escaped_text[i + 1]) {
                case '"':
                    unescaped[j++] = '"';
                    i++; // Skip the next character
                    break;
                case '\\':
                    unescaped[j++] = '\\';
                    i++; // Skip the next character
                    break;
                case 'n':
                    unescaped[j++] = '\n';
                    i++; // Skip the next character
                    break;
                case 'r':
                    unescaped[j++] = '\r';
                    i++; // Skip the next character
                    break;
                case 't':
                    unescaped[j++] = '\t';
                    i++; // Skip the next character
                    break;
                default:
                    // If it's not a recognized escape sequence, keep the backslash
                    unescaped[j++] = escaped_text[i];
                    break;
            }
        } else {
            unescaped[j++] = escaped_text[i];
        }
    }
    
    unescaped[j] = '\0';
    return unescaped;
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
    
    // Find the closing quote, but skip over escaped quotes
    const char* message_end = message_start;
    while (*message_end) {
        if (*message_end == '"') {
            // Found a quote, check if it's escaped
            // Count the number of preceding backslashes
            int backslash_count = 0;
            const char* temp = message_end - 1;
            while (temp >= message_start && *temp == '\\') {
                backslash_count++;
                temp--;
            }
            
            // If even number of backslashes (including 0), the quote is not escaped
            if (backslash_count % 2 == 0) {
                // This is the actual closing quote
                break;
            }
        }
        message_end++;
    }
    
    if (!*message_end) return NULL; // Reached end of string without finding closing quote
    
    // Calculate length and allocate memory
    int message_length = message_end - message_start;
    char* escaped_message = malloc(message_length + 1);
    
    if (!escaped_message) {
        printf("Failed to allocate memory for escaped message\n");
        return NULL;
    }
    
    strncpy(escaped_message, message_start, message_length);
    escaped_message[message_length] = '\0';
    
    // Unescape the JSON-escaped text
    char* unescaped_message = unescape_json_text(escaped_message);
    free(escaped_message);
    
    return unescaped_message;
}

// Initialize redactor (simplified since we're using HTTP API)
EMSCRIPTEN_KEEPALIVE
int init_redactor() {
    printf("Redactor initialized for HTTP API mode\n");
    return 1;
}

EMSCRIPTEN_KEEPALIVE
void update_extension_id(const char* old_session_id, const char* new_session_id) {
    char api_url[512];
    snprintf(api_url, sizeof(api_url), "http://localhost:16549/api/v1/chat/sessions/%s/update-extension-id", old_session_id);
    char* json_body = create_json_body("ExtensionId", new_session_id);
    if (!json_body) {
        printf("Failed to create JSON body\n");
        return;
    }
    http_post_sync(api_url, json_body);
    free(json_body);
}

// Main redaction function using HTTP API
EMSCRIPTEN_KEEPALIVE
char* redact(const char* session_id, const char* text) {
    if (!text || !session_id) return NULL;
    
    // Build the API URL
    char api_url[512];
    snprintf(api_url, sizeof(api_url), "http://localhost:16549/api/v1/chat/sessions/%s/redact", session_id);
    
    // Build JSON body using helper function
    char* json_body = create_json_body("Message", text);
    if (!json_body) {
        printf("Failed to create JSON body\n");
        return string_duplicate(text);
    }
    
    // Make the HTTP request
    char* response = http_post_sync(api_url, json_body);
    
    // Clean up JSON body
    free(json_body);
    
    if (!response) {
        printf("Failed to get response from server\n");
        return string_duplicate(text);
    }
    
    // Parse the response to extract the Message field
    char* redacted_message = parse_message_from_json(response);
    free(response);
    
    if (!redacted_message) {
        printf("Failed to parse message from response\n");
        return string_duplicate(text);
    }
    
    return redacted_message;
}

// Simplified restore function (just returns text as-is since we use HTTP API)
EMSCRIPTEN_KEEPALIVE
char* restore(const char* session_id, const char* text) {
    if (!text || !session_id) return NULL;
    
    // Build the API URL
    char api_url[512];
    snprintf(api_url, sizeof(api_url), "http://localhost:16549/api/v1/chat/sessions/%s/restore", session_id);
    
    // Build JSON body using helper function
    char* json_body = create_json_body("Message", text);
    if (!json_body) {
        printf("Failed to create JSON body\n");
        return string_duplicate(text);
    }
    
    // Make the HTTP request
    char* response = http_post_sync(api_url, json_body);
    
    // Clean up JSON body
    free(json_body);
    
    if (!response) {
        printf("Failed to get response from server\n");
        return string_duplicate(text);
    }
    
    // Parse the response to extract the Message field
    char* restored_message = parse_message_from_json(response);
    free(response);
    
    if (!restored_message) {
        printf("Failed to parse message from response\n");
        return string_duplicate(text);
    }
    
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
