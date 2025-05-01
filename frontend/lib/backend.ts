import axiosInstance, { nerBaseUrl } from './axios.config';
import axios from 'axios';

export const nerService = {

  checkHealth: async () => {
    const response = await axiosInstance.get('/health');
    return response.data;
  },

  listModels: async (): Promise<Model[]> => {
    const response = await axiosInstance.get('/models');
    return response.data;
  },

  getModel: async (modelId: string): Promise<Model> => {
    const response = await axiosInstance.get(`/models/${modelId}`);
    return response.data;
  },

  listReports: async (): Promise<Report[]> => {
    const response = await axiosInstance.get('/reports');
    return response.data;
  },

  createReport: async (data: CreateReportRequest): Promise<{ ReportId: string }> => {
    const response = await axiosInstance.post('/reports', data);
    return response.data;
  },

  getReport: async (reportId: string): Promise<Report> => {
    const response = await axiosInstance.get(`/reports/${reportId}`);
    return response.data;
  },

  getReportGroup: async (reportId: string, groupId: string): Promise<Group> => {
    const response = await axiosInstance.get(`/reports/${reportId}/groups/${groupId}`);
    return response.data;
  },

  getReportEntities: async (
    reportId: string,
    params?: {
      offset?: number;
      limit?: number;
      object?: string;
    }
  ): Promise<Entity[]> => {
    const response = await axiosInstance.get(`/reports/${reportId}/entities`, {
      params: {
        offset: params?.offset || 0,
        limit: params?.limit || 100,
        ...(params?.object && { object: params.object }),
      },
    });
    return response.data;
  },

  searchReport: async (reportId: string, query: string): Promise<{ Objects: string[] }> => {
    const response = await axiosInstance.get(`/reports/${reportId}/search`, {
      params: { query }
    });
    return response.data;
  },

  uploadFiles: async (files: File[]): Promise<{ Id: string }> => {
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