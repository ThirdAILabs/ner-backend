import axios from 'axios';

// Use an absolute URL directly for Electron app
const isElectron = typeof window !== 'undefined' && (window as any).electron?.isElectron;
const isProduction = process.env.NODE_ENV === 'production';

// Direct URL to local backend when running in Electron, otherwise use relative path
const nerBaseUrl = isElectron || isProduction 
  ? 'http://localhost:8000/api/v1'
  : '/api/v1';

const axiosInstance = axios.create({
    baseURL: nerBaseUrl,
    headers: {
        'Content-Type': 'application/json',
    },
});

axiosInstance.interceptors.response.use(
    (response) => response,
    (error) => {
        const errorMessage =
            error.response?.data?.message || error.message || 'An unexpected error occurred';
        const errorStatus = error.response?.status || 500;
        // For debugging - log error in development
        if (process.env.NODE_ENV !== 'production') {
            console.error(`API Error (${errorStatus}): ${errorMessage}`);
        }

        return Promise.reject(error);
    }
);

export default axiosInstance;