import axios from 'axios';

// Use the same API endpoint as the Next.js app
const nerBaseUrl = 'http://localhost:8000/api/v1'; // Updated to port 8000

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
        
        console.error(`API Error (${errorStatus}): ${errorMessage}`);
        
        return Promise.reject(error);
    }
);

export default axiosInstance; 