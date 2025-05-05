import axiosInstance from './axios.config';
import axios from 'axios';

// Type definitions for API responses
interface Model {
  Id: string;
  Name: string;
  Type: string;
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
  CreationTime: string;
  Tags?: string[];
  CustomTags?: { [key: string]: string };
  Groups?: Group[];
  ShardDataTaskStatus?: string;
  InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  Errors?: string[];
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
  SourceS3Bucket?: string;
  SourceS3Prefix?: string;
  Tags: string[];
  CustomTags?: { [key: string]: string };
  Groups?: { [key: string]: string };
}

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

  getTagsFromModel: async (modelId: string): Promise<string[]> => {
    try {
      const model = await nerService.getModel(modelId);
      return model.Tags || [];
    } catch (error) {
      console.error("Error fetching tags from model:", error);
      return [];
    }
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

  getTagsFromReport: async (reportId: string): Promise<{
    regularTags: string[],
    customTags: { [key: string]: string }
  }> => {
    try {
      const report = await nerService.getReport(reportId);
      return {
        regularTags: report.Tags || [],
        customTags: report.CustomTags || {}
      };
    } catch (error) {
      console.error("Error fetching tags from report:", error);
      return { regularTags: [], customTags: {} };
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

  getReportObjects: async (
    reportId: string,
    params?: {
      offset?: number;
      limit?: number;
    }
  ): Promise<ObjectPreview[]> => {
    // Since the /objects endpoint doesn't exist yet, we'll use entities endpoint
    // and transform the data to the format we need
    const entities = await nerService.getReportEntities(reportId, {
      offset: params?.offset || 0,
      limit: params?.limit || 100,
    });
    
    // Group entities by object name
    const objectMap = new Map<string, { tokens: string[], tags: string[] }>();
    
    entities.forEach(entity => {
      if (!objectMap.has(entity.Object)) {
        objectMap.set(entity.Object, { tokens: [], tags: [] });
      }
      
      // For each entity, we add the text and its label
      const obj = objectMap.get(entity.Object)!;
      
      // Add left context as regular text with "O" tag
      if (entity.LContext) {
        obj.tokens.push(entity.LContext);
        obj.tags.push("O");
      }
      
      // Add the entity text with its tag
      obj.tokens.push(entity.Text);
      obj.tags.push(entity.Label);
      
      // Add right context as regular text with "O" tag
      if (entity.RContext) {
        obj.tokens.push(entity.RContext);
        obj.tags.push("O");
      }
    });
    
    // Convert map to array of ObjectPreview objects
    return Array.from(objectMap.entries()).map(([objectName, data]) => ({
      object: objectName,
      tokens: data.tokens,
      tags: data.tags
    }));
  },

  getUniqueTagsFromEntities: async (reportId: string, limit: number = 500): Promise<string[]> => {
    try {
      const entities = await nerService.getReportEntities(reportId, { limit });
      // Extract and deduplicate tag types
      return Array.from(new Set(entities.map(e => e.Label)));
    } catch (error) {
      console.error("Error fetching unique tags from entities:", error);
      return [];
    }
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