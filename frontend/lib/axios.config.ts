import axios from 'axios';

// Use our fixed port (16549) for backend communication
const nerBaseUrl = 'https://e943-2600-1700-cd1-f30-00-c.ngrok-free.app/api/v1'

const axiosInstance = axios.create({
    baseURL: nerBaseUrl,
    headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
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
        //     window.location.href = `/error?message=${encodeURIComponent(errorMessage)}
        //   &status=${encodeURIComponent(errorStatus)}`;

        return Promise.reject(error);
    }
);

export default axiosInstance;