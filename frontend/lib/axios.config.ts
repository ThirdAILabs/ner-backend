import axios from 'axios';

// Use the Next.js API proxy instead of direct server URL
const nerBaseUrl = 'https://d532-2600-1700-cd1-f30-00-c.ngrok-free.app/api/v1/'

const axiosInstance = axios.create({
    baseURL: nerBaseUrl,
    headers: {
        'Content-Type': 'application/json',
        "ngrok-skip-browser-warning": "true"
    },
});

axiosInstance.interceptors.response.use(
    (response) => response,
    (error) => {
        const errorMessage =
            error.response?.data?.message || error.message || 'An unexpected error occurred';
        const errorStatus = error.response?.status || 500;
        //     window.location.href = `/error?message=${encodeURIComponent(errorMessage)}
        //   &status=${encodeURIComponent(errorStatus)}`;

        return Promise.reject(error);
    }
);

export default axiosInstance;