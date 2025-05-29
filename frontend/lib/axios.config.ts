import axios from 'axios';

const axiosInstance = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

export const updateNerBaseUrl = async () => {
  if (typeof window !== 'undefined') {
    // @ts-ignore
    const port = await window.electronAPI.getPort();
    console.log('port', port);
    if (port) {
      axiosInstance.defaults.baseURL = `http://localhost:${port}/api/v1`;
      return true
    }
  }
  return false;
};

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
