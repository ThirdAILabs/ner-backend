#include "redactor.c"  // Include the source directly for testing

int main() {
    printf("=== Testing C redactor module ===\n\n");
    
    // Initialize the redactor
    printf("1. Initializing redactor...\n");
    if (!init_redactor()) {
        printf("   ❌ Failed to initialize redactor\n");
        return 1;
    }
    printf("   ✅ Redactor initialized successfully\n\n");
    
    // Test redaction with mock data
    printf("2. Testing redaction...\n");
    const char* test_session = "test-session-123";
    const char* test_text = "Contact John Doe at john.doe@email.com or call 555-123-4567";
    printf("   Original text: %s\n", test_text);
    
    char* redacted = redact(test_session, test_text);
    if (redacted) {
        printf("   Redacted text: %s\n", redacted);
        printf("   ✅ Redaction completed\n");
    } else {
        printf("   ❌ Redaction failed\n");
        return 1;
    }
    printf("\n");
    
    // Test restoration
    printf("3. Testing restoration...\n");
    char* restored = restore(redacted);
    if (restored) {
        printf("   Restored text: %s\n", restored);
        printf("   ✅ Restoration completed\n");
    } else {
        printf("   ❌ Restoration failed\n");
        free(redacted);
        return 1;
    }
    printf("\n");
    
    // Test statistics
    printf("4. Testing statistics...\n");
    char* stats = get_redaction_stats();
    if (stats) {
        printf("   Stats: %s\n", stats);
        printf("   ✅ Statistics retrieved\n");
    } else {
        printf("   ❌ Failed to get statistics\n");
    }
    printf("\n");
    
    // Test cleanup
    printf("5. Testing cleanup...\n");
    clear_redaction_mappings();
    cleanup_redactor();
    printf("   ✅ Cleanup completed\n\n");
    
    // Free allocated memory
    if (redacted) free(redacted);
    if (restored) free(restored);
    if (stats) free(stats);
    
    printf("=== All tests completed successfully! ===\n");
    return 0;
} 