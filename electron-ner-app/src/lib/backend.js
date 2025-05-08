import axiosInstance from './axios.config';

/**
 * Entity object structure:
 * {
 *   Object: string,      // Name of the containing object
 *   Start: number,       // Start position in the object
 *   End: number,         // End position in the object
 *   Label: string,       // Entity label (e.g. "PERSON", "ORG")
 *   Text: string,        // The entity text 
 *   LContext: string,    // Left context
 *   RContext: string     // Right context
 * }
 * 
 * ObjectPreview structure:
 * {
 *   object: string,      // Name of the object
 *   tokens: string[],    // Array of tokens
 *   tags: string[]       // Array of tags (parallel to tokens)
 * }
 */

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
    if (!reportId) {
      console.error('getReportEntities called without reportId');
      return [];
    }

    console.log('getReportEntities called with:', { reportId, params });
    
    try {
      const response = await axiosInstance.get(`/reports/${reportId}/entities`, {
        params: {
          offset: params?.offset || 0,
          limit: params?.limit || 100,
          ...(params?.object && { object: params.object }),
        },
      });
      console.log('getReportEntities response:', response.data ? response.data.length : 'empty');
      return response.data;
    } catch (error) {
      console.error('getReportEntities error:', error);
      throw error;
    }
  },

  getReportObjects: async (reportId, params) => {
    if (!reportId) {
      console.error('getReportObjects called without reportId');
      return [];
    }

    console.log('getReportObjects called with:', { reportId, params });
    
    try {
      // Since the /objects endpoint doesn't exist yet, we'll use entities endpoint
      // and transform the data to the format we need
      const entities = await nerService.getReportEntities(reportId, {
        offset: params?.offset || 0,
        limit: params?.limit || 100,
      });

      console.log('getReportObjects received entities:', entities.length);

      // Group entities by object name
      const objectMap = new Map();

      entities.forEach(entity => {
        if (!objectMap.has(entity.Object)) {
          objectMap.set(entity.Object, { tokens: [], tags: [] });
        }

        // For each entity, we add the text and its label
        const obj = objectMap.get(entity.Object);

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
      const result = Array.from(objectMap.entries()).map(([objectName, data]) => ({
        object: objectName,
        tokens: data.tokens,
        tags: data.tags
      }));

      console.log('getReportObjects returning objects:', result.length);
      return result;
    } catch (error) {
      console.error('getReportObjects error:', error);
      throw error;
    }
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

  searchReport: async (reportId, query) => {
    const response = await axiosInstance.get(`/reports/${reportId}/search`, {
      params: { query }
    });
    return response.data;
  },
}; 