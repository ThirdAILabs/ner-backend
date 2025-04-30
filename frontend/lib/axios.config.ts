import axios from 'axios';

export const nerBaseUrl = process.env.NEXT_PUBLIC_NER_API_URL || "http://localhost:8001";

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