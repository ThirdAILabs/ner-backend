#include <Python.h>
#include <emscripten.h>
#include <string.h>
#include <stdlib.h>

// Global variables to hold Python module and functions
static PyObject *redactor_module = NULL;
static PyObject *redact_func = NULL;
static PyObject *restore_func = NULL;
static PyObject *clear_func = NULL;
static PyObject *stats_func = NULL;

// Initialize Python and import the redactor module
EMSCRIPTEN_KEEPALIVE
int init_redactor() {
    if (Py_IsInitialized()) {
        return 1; // Already initialized
    }
    
    // Initialize Python
    Py_Initialize();
    if (!Py_IsInitialized()) {
        return 0; // Failed to initialize
    }
    
    // Import the redactor module
    redactor_module = PyImport_ImportModule("redactor");
    if (!redactor_module) {
        PyErr_Print();
        return 0;
    }
    
    // Get function references
    redact_func = PyObject_GetAttrString(redactor_module, "redact");
    restore_func = PyObject_GetAttrString(redactor_module, "restore");
    clear_func = PyObject_GetAttrString(redactor_module, "clear_redaction_mappings");
    stats_func = PyObject_GetAttrString(redactor_module, "get_redaction_stats");
    
    if (!redact_func || !restore_func || !clear_func || !stats_func) {
        PyErr_Print();
        return 0;
    }
    
    return 1; // Success
}

// Helper function to call Python function with string argument
static char* call_python_string_func(PyObject *func, const char *input) {
    if (!func || !input) {
        return NULL;
    }
    
    // Create Python string object
    PyObject *py_input = PyUnicode_FromString(input);
    if (!py_input) {
        return NULL;
    }
    
    // Call the function
    PyObject *py_result = PyObject_CallFunctionObjArgs(func, py_input, NULL);
    Py_DECREF(py_input);
    
    if (!py_result) {
        PyErr_Print();
        return NULL;
    }
    
    // Convert result to C string
    const char *result_str = PyUnicode_AsUTF8(py_result);
    if (!result_str) {
        Py_DECREF(py_result);
        return NULL;
    }
    
    // Duplicate the string for return
    char *result = strdup(result_str);
    Py_DECREF(py_result);
    
    return result;
}

// Redact function exposed to JavaScript
EMSCRIPTEN_KEEPALIVE
char* redact(const char* text) {
    if (!redact_func) {
        if (!init_redactor()) {
            return strdup("Error: Failed to initialize redactor");
        }
    }
    
    char *result = call_python_string_func(redact_func, text);
    return result ? result : strdup(text); // Return original text if redaction fails
}

// Restore function exposed to JavaScript
EMSCRIPTEN_KEEPALIVE
char* restore(const char* text) {
    if (!restore_func) {
        if (!init_redactor()) {
            return strdup("Error: Failed to initialize redactor");
        }
    }
    
    char *result = call_python_string_func(restore_func, text);
    return result ? result : strdup(text); // Return original text if restoration fails
}

// Clear redaction mappings
EMSCRIPTEN_KEEPALIVE
void clear_redaction_mappings() {
    if (!clear_func) {
        if (!init_redactor()) {
            return;
        }
    }
    
    PyObject *result = PyObject_CallFunctionObjArgs(clear_func, NULL);
    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }
}

// Get redaction statistics
EMSCRIPTEN_KEEPALIVE
char* get_redaction_stats() {
    if (!stats_func) {
        if (!init_redactor()) {
            return strdup("{}");
        }
    }
    
    char *result = call_python_string_func(stats_func, "");
    return result ? result : strdup("{}");
}

// Free memory allocated by the WASM functions
EMSCRIPTEN_KEEPALIVE
void free_string(char* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// Cleanup function
EMSCRIPTEN_KEEPALIVE
void cleanup_redactor() {
    if (redact_func) { Py_DECREF(redact_func); redact_func = NULL; }
    if (restore_func) { Py_DECREF(restore_func); restore_func = NULL; }
    if (clear_func) { Py_DECREF(clear_func); clear_func = NULL; }
    if (stats_func) { Py_DECREF(stats_func); stats_func = NULL; }
    if (redactor_module) { Py_DECREF(redactor_module); redactor_module = NULL; }
    
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
} 