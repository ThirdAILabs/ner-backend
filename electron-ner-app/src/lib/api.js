// API utilities for communicating with the Go backend

// Get the API details from the exposed Electron API
const API_BASE_URL = window.electronAPI?.backendAPI?.baseUrl || 'http://localhost:8000';
const API_VERSION = window.electronAPI?.backendAPI?.apiVersion || 'v1';

// Base API URL
const API_URL = `${API_BASE_URL}/api/${API_VERSION}`;

/**
 * Generic API request function
 */
async function apiRequest(endpoint, method = 'GET', body = null, headers = {}) {
  try {
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(`${API_URL}/${endpoint}`, options);
    
    // Handle non-JSON responses
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || `Error: ${response.status}`);
      }
      
      return data;
    } else {
      const text = await response.text();
      
      if (!response.ok) {
        throw new Error(text || `Error: ${response.status}`);
      }
      
      return text;
    }
  } catch (error) {
    console.error(`API request failed: ${error.message}`);
    throw error;
  }
}

// API Functions
export const api = {
  // Check backend health
  checkHealth: async () => {
    try {
      const response = await apiRequest('health');
      return { ok: true, data: response };
    } catch (error) {
      return { ok: false, error: error.message };
    }
  },
  
  // Add all your API methods here...
  
  // Example functions:
  jobs: {
    getJobs: async () => apiRequest('jobs'),
    getJob: async (id) => apiRequest(`jobs/${id}`),
    createJob: async (data) => apiRequest('jobs', 'POST', data),
  },
  
  // You would add all the API functions that match your Go backend here
};

export default api; 