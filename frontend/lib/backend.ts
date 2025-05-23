import axiosInstance, { showApiErrorEvent } from './axios.config';
import axios from 'axios';
import qs from 'qs';

// Type definitions for API responses
interface Model {
  Id: string;
  Name: string;
  Status: string;
  BaseModelId?: string;
  Tags?: string[];
}

interface Group {
  Id: string;
  Name: string;
  Query: string;
  Objects?: string[];
}

interface TaskStatusCategory {
  TotalTasks: number;
  TotalSize: number;
}

interface Report {
  Id: string;
  Model: Model;
  SourceS3Bucket: string;
  SourceS3Prefix: string;
  IsUpload?: boolean;
  CreationTime: string;
  Tags?: string[];
  CustomTags?: {
    [key: string]: string;
  };
  FileCount: number;
  SucceededFileCount: number;
  FailedFileCount: number;
  Groups?: Group[];
  ShardDataTaskStatus?: string;
  InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  Errors?: string[];
  ReportName: string;
  TagCounts: { [key: string]: number };
  TotalInferenceTimeSeconds: number;
  ShardDataTimeSeconds: number;
}

interface Entity {
  Object: string;
  Start: number;
  End: number;
  Label: string;
  Text: string;
  LContext?: string;
  RContext?: string;
}

interface ObjectPreview {
  object: string;
  tokens: string[];
  tags: string[];
}

interface CreateReportRequest {
  ModelId: string;
  UploadId?: string;
  S3Endpoint?: string;
  S3Region?: string;
  SourceS3Bucket?: string;
  SourceS3Prefix?: string;
  Tags: string[];
  CustomTags?: { [key: string]: string };
  Groups?: { [key: string]: string };
  report_name: string;
}

export interface InferenceMetrics {
  Completed: number;
  Failed: number;
  InProgress: number;
  DataProcessedMB: number;
  TokensProcessed: number;
}

export interface ThroughputMetrics {
  ModelID: string;
  ReportID?: string;
  ThroughputMBPerHour: number;
}

// Add a utility function to handle API errors with custom messages
const handleApiError = (error: unknown, customMessage?: string): never => {
  console.error('API Error:', error);

  // Extract the error message
  let errorMessage = 'An unexpected error occurred';
  let status: number | undefined = undefined;

  if (axios.isAxiosError(error) && error.response) {
    errorMessage = error.response.data?.message || error.message;
    status = error.response.status;
  } else if (error instanceof Error) {
    errorMessage = error.message;
  }

  // Show the error message (use custom message if provided)
  if (typeof window !== 'undefined') {
    showApiErrorEvent(customMessage || errorMessage, status);
  }

  throw error;
};

export const nerService = {
  checkHealth: async () => {
    try {
      const response = await axiosInstance.get('/health');
      return response;
    } catch (error) {
      return handleApiError(error, 'Failed to connect to the backend service');
    }
  },

  listModels: async (): Promise<Model[]> => {
    try {
      const response = await axiosInstance.get('/models');
      return response.data;
    } catch (error) {
      return handleApiError(error, 'Failed to load models');
    }
  },

  getModel: async (modelId: string): Promise<Model> => {
    try {
      const response = await axiosInstance.get(`/models/${modelId}`);
      return response.data;
    } catch (error) {
      return handleApiError(error, `Failed to load model details for ${modelId}`);
    }
  },

  getTagsFromModel: async (modelId: string): Promise<string[]> => {
    try {
      const model = await nerService.getModel(modelId);
      return model.Tags || [];
    } catch (error) {
      console.error('Error fetching tags from model:', error);
      return [];
    }
  },

  listReports: async (): Promise<Report[]> => {
    try {
      const response = await axiosInstance.get('/reports');
      return response.data;
    } catch (error) {
      return handleApiError(error, 'Failed to load reports');
    }
  },

  createReport: async (data: CreateReportRequest): Promise<{ ReportId: string }> => {
    try {
      const response = await axiosInstance.post('/reports', data);
      return response.data;
    } catch (error) {
      return handleApiError(error, 'Failed to create report');
    }
  },

  getReport: async (reportId: string): Promise<Report> => {
    try {
      const response = await axiosInstance.get(`/reports/${reportId}`);
      return response.data;
    } catch (error) {
      return handleApiError(error, `Failed to load report ${reportId}`);
    }
  },

  deleteReport: async (reportId: string): Promise<void> => {
    try {
      await axiosInstance.delete(`/reports/${reportId}`);
    } catch (error) {
      return handleApiError(error, `Failed to delete report ${reportId}`);
    }
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
      tags?: string[];
    }
  ): Promise<Entity[]> => {
    const response = await axiosInstance.get(`/reports/${reportId}/entities`, {
      params: {
        offset: params?.offset || 0,
        limit: params?.limit || 100,
        tags: params?.tags,
      },
      paramsSerializer: (params) => qs.stringify(params, { arrayFormat: 'repeat' }),
    });
    console.log('Entities response:', response.data);
    return response.data;
  },

  getReportObjects: async (
    reportId: string,
    params?: {
      offset?: number;
      limit?: number;
      tags?: string[];
    }
  ): Promise<ObjectPreview[]> => {
    // Fetch object previews from the backend
    const response = await axiosInstance.get<ObjectPreview[]>(`/reports/${reportId}/objects`, {
      params: {
        offset: params?.offset || 0,
        limit: params?.limit || 100,
      },
    });
    return response.data;
  },

  searchReport: async (reportId: string, query: string): Promise<{ Objects: string[] }> => {
    const response = await axiosInstance.get(`/reports/${reportId}/search`, {
      params: { query },
    });
    return response.data;
  },

  uploadFiles: async (files: File[]): Promise<{ Id: string }> => {
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file);
      });

      const response = await axiosInstance.post(`/uploads`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      return handleApiError(error, 'Failed to upload files');
    }
  },

  getInferenceMetrics: async (modelId?: string, days?: number): Promise<InferenceMetrics> => {
    const params: Record<string, any> = {};
    if (modelId) params.model_id = modelId;
    if (days !== undefined) params.days = days;
    const { data } = await axiosInstance.get<InferenceMetrics>('/metrics', {
      params,
    });
    return data;
  },

  getThroughputMetrics: async (modelId: string, reportId?: string): Promise<ThroughputMetrics> => {
    const params: Record<string, any> = { model_id: modelId };
    if (reportId) params.report_id = reportId;
    const { data } = await axiosInstance.get<ThroughputMetrics>('/metrics/throughput', { params });
    return data;
  },

  validateGroupDefinition: async (groupQuery: string): Promise<string | null> => {
    try {
      await axiosInstance.get('/validate/group', {
        params: { GroupQuery: groupQuery },
      });
      return null;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data) {
        return error.response.data;
      }
      return 'An unexpected error occurred';
    }
  },

  attemptS3Connection: async (
    endpoint: string,
    region: string,
    bucket: string,
    prefix: string
  ): Promise<string | null> => {
    try {
      await axiosInstance.get('/validate/s3', {
        params: {
          S3Endpoint: endpoint,
          S3Region: region,
          SourceS3Bucket: bucket,
          SourceS3Prefix: prefix,
        },
      });

      return null;
    } catch (error) {
      return 'Failed to connect to S3 bucket. Please make sure that it is a valid public bucket.';
    }
  },

  getChatSessions: async (): Promise<{ data: { id: string; title: string }[]; error: string | null }> => {
    try {
      const { data } = await axiosInstance.get('/chat/sessions');
      return { data: data.sessions, error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data 
        ? error.response.data 
        : 'Failed to get chat sessions';
      return { data: [], error: errorMsg };
    }
  },

  startChatSession: async (model: string, title: string): Promise<{ data: { session_id: string } | null; error: string | null }> => {
    try {
      const { data } = await axiosInstance.post('/chat/sessions', { model, title });
      return { data, error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data
        ? error.response.data
        : 'Failed to start chat session';
      return { data: null, error: errorMsg };
    }
  },

  deleteChatSession: async (sessionId: string): Promise<{ error: string | null }> => {
    try {
      await axiosInstance.delete(`/chat/sessions/${sessionId}`);
      return { error: null };
    } catch (error) {
      return { error: 'Failed to delete chat session' };
    }
  },

  getChatSession: async (sessionId: string): Promise<{ data: { id: string; title: string } | null; error: string | null }> => {
    try {
      const { data } = await axiosInstance.get(`/chat/sessions/${sessionId}`);
      return { data, error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data
        ? error.response.data
        : 'Failed to get chat session';
      return { data: null, error: errorMsg };
    }
  },

  renameChatSession: async (sessionId: string, title: string): Promise<{ error: string | null }> => {
    try {
      await axiosInstance.post(`/chat/sessions/${sessionId}/rename`, { title });
      return { error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data
        ? error.response.data
        : 'Failed to rename chat session';
      return { error: errorMsg };
    }
  },

  sendChatMessage: async (
    sessionId: string,
    model: string,
    apiKey: string,
    message: string
  ): Promise<{
    data: { input_text: string; reply: string; tag_map: Record<string, string> } | null;
    error: string | null;
  }> => {
    try {
      const { data } = await axiosInstance.post(`/chat/sessions/${sessionId}/messages`, {
        model,
        api_key: apiKey,
        message,
      });
      return { data, error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data
        ? error.response.data
        : 'Failed to send chat message';
      return { data: null, error: errorMsg };
    }
  },

  getChatHistory: async (
    sessionId: string
  ): Promise<{
    data: Array<{
      message_type: string;
      content: string;
      timestamp: string;
      metadata?: any;
    }>;
    error: string | null;
  }> => {
    try {
      const { data } = await axiosInstance.get(`/chat/sessions/${sessionId}/history`);
      return { data, error: null };
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) && error.response?.data
        ? error.response.data
        : 'Failed to get chat history';
      return { data: [], error: errorMsg };
    }
  },

  getOpenAIApiKey: async (): Promise<{ apiKey: string, error: string | null }> => {
    try {
      const response = await axiosInstance.get('/chat/api-key');
      console.log("API key", response.data);
      return { apiKey: response.data.api_key, error: null };
    } catch (error) {
      return { apiKey: '', error: 'Failed to get OpenAI API key' };
    }
  },

  setOpenAIApiKey: async (apiKey: string): Promise<string | null> => {
    try {
      await axiosInstance.post('/chat/api-key', { api_key: apiKey });
      return null;
    } catch (error) {
      return 'Failed to set OpenAI API key';
    }
  },
};
