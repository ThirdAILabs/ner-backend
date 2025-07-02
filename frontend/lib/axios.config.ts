import axios from 'axios';

// Use our fixed port (16549) for backend communication
export const nerBaseUrl = '/api/v1';

const axiosInstance = axios.create({
  baseURL: nerBaseUrl,
  headers: {
    'Content-Type': 'application/json',
  },
});

axiosInstance.interceptors.request.use(
  (config) => {
    console.log('Request URL:', config.url);
    console.log('Request Method:', config.method);
    console.log('Request Headers:', config.headers);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Create a custom event for API errors
export const showApiErrorEvent = (message: string, status?: number) => {
  const event = new CustomEvent('api-error', {
    detail: { message, status },
  });
  window.dispatchEvent(event);
};

axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMessage =
      error.response?.data?.message || error.message || 'An unexpected error occurred';
    const errorStatus = error.response?.status || 500;

    // Dispatch custom error event
    if (typeof window !== 'undefined') {
      showApiErrorEvent(errorMessage, errorStatus);
    }

    return Promise.reject(error);
  }
);

export default axiosInstance;
