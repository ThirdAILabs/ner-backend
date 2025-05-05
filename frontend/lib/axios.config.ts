import axios from 'axios';

// Use the Next.js API proxy instead of direct server URL
const nerBaseUrl = '/api/v1'

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
        //     window.location.href = `/error?message=${encodeURIComponent(errorMessage)}
        //   &status=${encodeURIComponent(errorStatus)}`;

        return Promise.reject(error);
    }
);

export default axiosInstance;