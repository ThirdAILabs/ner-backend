import axiosInstance, { showApiErrorEvent } from './axios.config';
import axios from 'axios';
import qs from 'qs';

// Type definitions for API responses
export interface ChatResponse {
  InputText: string;
  Reply: string;
  TagMap: Record<string, string>;
}

export interface Feedback {
  Tokens: string[];
  Labels: string[];
}

export interface SavedFeedback {
  Id: string;
  Tokens?: string[];
  Labels?: string[];
  tokens?: string[];
  labels?: string[];
}

export interface TagInfo {
  name: string;
  description: string;
  examples: string[];
}

export interface FinetuneRequest {
  Name: string;
  TaskPrompt?: string;
  GenerateData?: boolean;
  Tags?: TagInfo[];
  Samples?: Feedback[];
}

export interface FinetuneResponse {
  ModelId: string;
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

  getLicense: async (): Promise<License> => {
    try {
      const response = await axiosInstance.get('/license');
      return response.data;
    } catch (error) {
      return handleApiError(error, 'Failed to load license information');
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
        tags: params?.tags,
      },
      paramsSerializer: (params) => qs.stringify(params, { arrayFormat: 'repeat' }),
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

  getChatSessions: async (): Promise<
    { ID: string; Title: string; TagMap: Record<string, string> }[]
  > => {
    const { data } = await axiosInstance.get('/chat/sessions');
    return data.Sessions;
  },

  startChatSession: async (model: string, title: string): Promise<string> => {
    const { data } = await axiosInstance.post('/chat/sessions', { Model: model, Title: title });
    return data.SessionID;
  },

  deleteChatSession: async (sessionId: string): Promise<void> => {
    await axiosInstance.delete(`/chat/sessions/${sessionId}`);
  },

  getChatSession: async (
    sessionId: string
  ): Promise<{ ID: string; Title: string; TagMap: Record<string, string> }> => {
    const { data } = await axiosInstance.get(`/chat/sessions/${sessionId}`);
    return data;
  },

  renameChatSession: async (sessionId: string, title: string): Promise<void> => {
    await axiosInstance.post(`/chat/sessions/${sessionId}/rename`, { Title: title });
  },

  sendChatMessageStream: async (
    sessionId: string,
    model: string,
    message: string,
    onChunk: (chunk: ChatResponse) => void
  ) => {
    let response;
    try {
      response = await axiosInstance.post(
        `/chat/sessions/${sessionId}/messages`,
        {
          Model: model,
          Message: message,
        },
        {
          responseType: 'stream',
          adapter: 'fetch',
        }
      );
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data) {
        const reader = error.response.data.getReader();
        const decoder = new TextDecoder();
        let text = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          text += decoder.decode(value);
        }
        throw new Error(text);
      }
      throw error;
    }

    const reader = response.data.getReader();
    const decoder = new TextDecoder();
    let chunk: ReadableStreamReadResult<Uint8Array>;

    while (!(chunk = await reader.read()).done) {
      const decodedChunk = decoder.decode(chunk.value, { stream: true });
      const lines = decodedChunk.split('\n').filter(Boolean);

      for (const line of lines) {
        const parsedData = JSON.parse(line);
        if (parsedData.Code !== 200) {
          throw new Error(parsedData.Error);
        }
        onChunk(parsedData.Data);
      }
    }
  },

  getChatHistory: async (
    sessionId: string
  ): Promise<
    {
      MessageType: string;
      Content: string;
      Timestamp: string;
      Metadata?: any;
    }[]
  > => {
    const { data } = await axiosInstance.get(`/chat/sessions/${sessionId}/history`);
    return data || [];
  },

  getOpenAIApiKey: async (): Promise<string> => {
    const response = await axiosInstance.get('/chat/api-key');
    return response.data.ApiKey;
  },

  setOpenAIApiKey: async (apiKey: string): Promise<void> => {
    await axiosInstance.post('/chat/api-key', { ApiKey: apiKey });
  },

  storeFileNameToPath: async (uploadId: string, mapping: { [filename: string]: string }) => {
    try {
      await axiosInstance.post(`/file-name-to-path/${uploadId}`, { Mapping: mapping });
    } catch (error) {
      return handleApiError(error, 'Failed to store upload path mappings');
    }
  },

  getFileNameToPath: async (uploadId: string) => {
    const { data } = await axiosInstance.get(`/file-name-to-path/${uploadId}`);
    console.log('getFileNameToPath', data);
    return data.Mapping as { [filename: string]: string };
  },

  submitFeedback: async (modelId: string, feedback: Feedback) => {
    const { data } = await axiosInstance.post(`/models/${modelId}/feedback`, feedback);
    return data;
  },

  getFeedbackSamples: async (modelId: string): Promise<SavedFeedback[]> => {
    try {
      const { data } = await axiosInstance.get(`/models/${modelId}/feedback`);
      return data;
    } catch (error) {
      return handleApiError(error, `Failed to load feedback samples for model ${modelId}`);
    }
  },

  deleteModelFeedback: async (modelId: string, feedbackId: string): Promise<void> => {
    try {
      await axiosInstance.delete(`/models/${modelId}/feedback/${feedbackId}`);
    } catch (error) {
      return handleApiError(error, `Failed to delete feedback ${feedbackId} for model ${modelId}`);
    }
  },

  finetuneModel: async (modelId: string, request: FinetuneRequest): Promise<FinetuneResponse> => {
    try {
      const { data } = await axiosInstance.post(`/models/${modelId}/finetune`, request);
      return data;
    } catch (error) {
      // Don't use custom message for fine-tuning errors so we can see the actual backend error
      return handleApiError(error);
    }
  },
};
