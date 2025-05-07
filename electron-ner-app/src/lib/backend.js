import axiosInstance from './axios.config';

export const nerService = {
  checkHealth: async () => {
    const response = await axiosInstance.get('/health');
    return response.data;
  },

  listModels: async () => {
    const response = await axiosInstance.get('/models');
    return response.data;
  },

  getModel: async (modelId) => {
    const response = await axiosInstance.get(`/models/${modelId}`);
    return response.data;
  },

  getTagsFromModel: async (modelId) => {
    try {
      const model = await nerService.getModel(modelId);
      return model.Tags || [];
    } catch (error) {
      console.error("Error fetching tags from model:", error);
      return [];
    }
  },

  listReports: async () => {
    const response = await axiosInstance.get('/reports');
    return response.data;
  },

  createReport: async (data) => {
    const response = await axiosInstance.post('/reports', data);
    return response.data;
  },

  getReport: async (reportId) => {
    const response = await axiosInstance.get(`/reports/${reportId}`);
    return response.data;
  },

  getReportEntities: async (reportId, params) => {
    const response = await axiosInstance.get(`/reports/${reportId}/entities`, {
      params: {
        offset: params?.offset || 0,
        limit: params?.limit || 100,
        ...(params?.object && { object: params.object }),
      },
    });
    return response.data;
  },

  uploadFiles: async (files) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await axiosInstance.post(`/uploads`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    return response.data;
  },
}; 