import axios from 'axios';

// Add TypeScript interface for the Electron API
declare global {
  interface Window {
    electronAPI?: {
      backendAPI: {
        baseUrl: string;
        apiVersion: string;
      }
    }
  }
}

// Get backend URL from Electron context if available, otherwise use default
let nerBaseUrl = '/api/v1'; // Default path for API routes

// Check if we're in Electron environment by looking for the exposed API
if (typeof window !== 'undefined' && window.electronAPI) {
  const { baseUrl, apiVersion } = window.electronAPI.backendAPI;
  nerBaseUrl = `${baseUrl}/api/${apiVersion}`;
  console.log('Using Electron API URL:', nerBaseUrl);
}

const axiosInstance = axios.create({
    baseURL: nerBaseUrl,
    headers: {
        'Content-Type': 'application/json',
    },
});

axiosInstance.interceptors.request.use(config => {
    console.log('Request URL:', config.url);
    console.log('Request Method:', config.method);
    console.log('Request Headers:', config.headers);
    return config;
  }, error => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  });

axiosInstance.interceptors.response.use(
    (response) => response,
    (error) => {
        const errorMessage =
            error.response?.data?.message || error.message || 'An unexpected error occurred';
        const errorStatus = error.response?.status || 500;
        console.error(`API Error (${errorStatus}):`, errorMessage);

        return Promise.reject(error);
    }
);

export default axiosInstance;