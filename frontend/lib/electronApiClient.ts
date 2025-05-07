import axios from 'axios';

// Base API URLs - adjusted for Electron environment
const BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://localhost:8001/api/v1' // Local server endpoint when in Electron
  : '/api/v1'; // Use relative path for Next.js development with API routes

// Reusable API client for token classification endpoints
export const tokenClassification = {
  // Get available models/deployments
  getDeployments: async () => {
    try {
      const response = await axios.get(`${BASE_URL}/token-classification/deployments`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch deployments:', error);
      // Return mock data for demo purposes
      return ['deployment1', 'deployment2', 'deployment3'];
    }
  },
  
  // Classify text
  classifyText: async (deploymentId: string, text: string) => {
    try {
      const response = await axios.post(`${BASE_URL}/token-classification/${deploymentId}/classify`, { text });
      return response.data;
    } catch (error) {
      console.error('Failed to classify text:', error);
      throw error;
    }
  },
  
  // Get jobs for a deployment
  getJobs: async (deploymentId: string) => {
    try {
      const response = await axios.get(`${BASE_URL}/token-classification/${deploymentId}/jobs`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch jobs:', error);
      throw error;
    }
  }
};

// Electron-specific helper to check if we're running in Electron
export const isElectron = () => {
  return typeof window !== 'undefined' && 'electron' in window;
}; 