import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
  Alert,
} from '@mui/material';
import { Toaster, toast } from 'react-hot-toast';
import { nerService, SavedFeedback } from '@/lib/backend';
import { useHealth } from '@/contexts/HealthProvider';
import { TokenHighlighter } from '@/components/feedback/TokenHighlighter';
import { useConditionalTelemetry } from '@/hooks/useConditionalTelemetry';
import { ChevronRight } from 'lucide-react';
import { ApiKeyDialog } from './ApiKeyDialog';

export interface UserFeedbackSectionProps {
  feedbackData: SavedFeedback[];
  loadingFeedback: boolean;
  handleDeleteFeedback: (id: string) => void;
  handleFinetuneClick: () => void;
  showFinetuneDialog: boolean;
  finetuneModelName: string;
  setFinetuneModelName: (name: string) => void;
  finetuneTaskPrompt: string;
  setFinetuneTaskPrompt: (prompt: string) => void;
  finetuning: boolean;
  handleFinetuneSubmit: () => void;
  handleFinetuneCancel: () => void;
  availableTags: string[];
  generateData: 'yes' | 'no';
  setGenerateData: (value: 'yes' | 'no') => void;
  apiKeyError: string | null;
  showApiKeyDialog: boolean;
  setShowApiKeyDialog: (show: boolean) => void;
  apiKeyInput: string;
  setApiKeyInput: (input: string) => void;
  validatingApiKey: boolean;
  setValidatingApiKey: (validating: boolean) => void;
  setApiKeyError: (error: string | null) => void;
}

export function UserFeedbackSection({
  feedbackData,
  loadingFeedback,
  handleDeleteFeedback,
  handleFinetuneClick,
  showFinetuneDialog,
  finetuneModelName,
  setFinetuneModelName,
  finetuneTaskPrompt,
  setFinetuneTaskPrompt,
  finetuning,
  handleFinetuneSubmit,
  handleFinetuneCancel,
  availableTags,
  generateData,
  setGenerateData,
  apiKeyError,
  showApiKeyDialog,
  setShowApiKeyDialog,
  apiKeyInput,
  setApiKeyInput,
  validatingApiKey,
  setValidatingApiKey,
  setApiKeyError,
}: UserFeedbackSectionProps) {
  const handleDeleteClick = (feedbackId: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    handleDeleteFeedback(feedbackId);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            fontSize: '1.25rem',
            color: '#4a5568',
          }}
        >
          User Feedback
        </Typography>
        {feedbackData.length > 0 && (
          <Button
            variant="contained"
            color="primary"
            onClick={handleFinetuneClick}
            disabled={loadingFeedback}
            sx={{
              textTransform: 'none',
              fontWeight: 600,
              px: 3,
            }}
          >
            Finetune Model
          </Button>
        )}
      </Box>
      <div className="border rounded-md">
        {loadingFeedback ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress size={20} />
          </Box>
        ) : feedbackData.length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
              No feedback data available for this model
            </Typography>
          </Box>
        ) : (
          <div className="divide-y">
            {feedbackData.map((feedback: SavedFeedback, index) => {
              const tokens = (feedback.tokens || feedback.Tokens || []).map(
                (token: string, tokenIndex: number) => {
                  return {
                    text: token,
                    tag: (feedback.labels || feedback.Labels)?.[tokenIndex] || 'O',
                  };
                }
              );
              return (
                <details key={index} className="group text-sm leading-relaxed bg-white">
                  <summary className="p-3 cursor-pointer bg-gray-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
                      <span className="font-medium">Feedback {index + 1}</span>
                    </div>
                    <button
                      className="text-gray-700 hover:text-gray-700 transition-colors"
                      onClick={handleDeleteClick(feedback.Id)}
                      title="Delete feedback"
                    >
                      ✕
                    </button>
                  </summary>
                  <div className="p-4">
                    <TokenHighlighter
                      tokens={tokens}
                      availableTags={availableTags}
                      editable={false}
                    />
                  </div>
                </details>
              );
            })}
          </div>
        )}
      </div>
      {/* Finetune Dialog */}
      <Dialog open={showFinetuneDialog} onClose={handleFinetuneCancel} maxWidth="sm" fullWidth>
        <DialogTitle>Finetune Model</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body2" sx={{ mb: 3, color: 'text.secondary' }}>
              Create a new finetuned model using the feedback data you've reviewed. This will use
              all {feedbackData.length} feedback samples as training data.
            </Typography>
            <TextField
              autoFocus
              margin="dense"
              label="Model Name"
              fullWidth
              variant="outlined"
              value={finetuneModelName}
              onChange={(e) => setFinetuneModelName(e.target.value)}
              helperText="Enter a name for the new finetuned model"
              sx={{ mb: 2 }}
              required
            />
            <TextField
              margin="dense"
              label="Task Prompt (Optional)"
              fullWidth
              multiline
              rows={3}
              variant="outlined"
              value={finetuneTaskPrompt}
              onChange={(e) => setFinetuneTaskPrompt(e.target.value)}
              helperText="Optional custom prompt to guide the finetuning process"
              placeholder="e.g., Focus on improving accuracy for person names and locations..."
              sx={{ mb: 3 }}
            />
            <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>
              Synthetic Data Generation
            </Typography>
            <RadioGroup
              value={generateData}
              onChange={(e) => setGenerateData(e.target.value as 'yes' | 'no')}
              sx={{ mb: 2 }}
            >
              <FormControlLabel
                value="yes"
                control={<Radio />}
                label="Generate Synthetic Data"
                sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
              />
              <FormControlLabel
                value="no"
                control={<Radio />}
                label="Do not generate synthetic data"
                sx={{ '& .MuiFormControlLabel-label': { fontSize: '0.875rem' } }}
              />
            </RadioGroup>
            {generateData === 'yes' && (
              <Typography
                variant="caption"
                sx={{ color: 'text.secondary', display: 'block', mb: 2 }}
              >
                Synthetic data generation will use OpenAI to create additional training examples
                based on your feedback samples.
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={handleFinetuneCancel}
            disabled={finetuning}
            sx={{ textTransform: 'none' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleFinetuneSubmit}
            variant="contained"
            disabled={finetuning || !finetuneModelName.trim()}
            sx={{ textTransform: 'none', ml: 1 }}
          >
            {finetuning ? (
              <>
                <CircularProgress size={16} sx={{ mr: 1 }} />
                Starting Finetuning...
              </>
            ) : (
              'Start Finetuning'
            )}
          </Button>
        </DialogActions>
      </Dialog>

      {/* API Key Dialog */}
      <ApiKeyDialog
        open={showApiKeyDialog}
        onClose={() => setShowApiKeyDialog(false)}
        apiKeyInput={apiKeyInput}
        setApiKeyInput={setApiKeyInput}
        apiKeyError={apiKeyError}
        setApiKeyError={setApiKeyError}
        validatingApiKey={validatingApiKey}
        setValidatingApiKey={setValidatingApiKey}
        onSuccess={handleFinetuneSubmit}
      />
    </Box>
  );
}

const ModelCustomization: React.FC = () => {
  const recordEvent = useConditionalTelemetry();
  useEffect(() => {
    recordEvent({
      UserAction: 'view',
      UIComponent: 'Model Customization Page',
      Page: 'Token Classification Page',
    });
  }, []);

  const { healthStatus } = useHealth();
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [feedbackData, setFeedbackData] = useState<SavedFeedback[]>([]);
  const [loadingFeedback, setLoadingFeedback] = useState(false);

  useEffect(() => {
    if (!healthStatus) return;
    const fetchModels = () => {
      nerService
        .listModels()
        .then((ms: any[]) => setModels(ms))
        .catch((err: any) => {
          // Failed to load models
        });
    };
    fetchModels();
    const intervalId = setInterval(fetchModels, 5000);
    return () => clearInterval(intervalId);
  }, [healthStatus]);

  useEffect(() => {
    if (!selectedModel) return;
    setLoadingFeedback(true);
    nerService
      .getFeedbackSamples(selectedModel.Id)
      .then((feedback: any[]) => {
        // Feedback data received
        setFeedbackData(feedback);
      })
      .catch((e: any) => {
        setFeedbackData([]);
        // Failed to load feedback data
      })
      .finally(() => setLoadingFeedback(false));
  }, [selectedModel]);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0]);
    }
  }, [models, selectedModel]);

  const handleModelChange = (e: SelectChangeEvent<string>) => {
    const model = models.find((m) => m.Id === e.target.value) || null;
    setSelectedModel(model);
  };

  const [showFinetuneDialog, setShowFinetuneDialog] = useState(false);
  const [finetuneModelName, setFinetuneModelName] = useState('');
  const [finetuneTaskPrompt, setFinetuneTaskPrompt] = useState('');
  const [finetuning, setFinetuning] = useState(false);
  const [generateData, setGenerateData] = useState<'yes' | 'no'>('no');
  const [apiKeyError, setApiKeyError] = useState<string | null>(null);
  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [validatingApiKey, setValidatingApiKey] = useState(false);

  const handleDeleteFeedback = async (id: string) => {
    if (!selectedModel) return;
    try {
      await nerService.deleteModelFeedback(selectedModel.Id, id);
      setFeedbackData(feedbackData.filter((feedback) => feedback.Id !== id));
    } catch (error) {
      // Failed to delete feedback
    }
  };

  const handleFinetuneClick = () => {
    if (!selectedModel) return;
    const timestamp = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '-');
    setFinetuneModelName(`finetuned_${timestamp}`);
    setFinetuneTaskPrompt('');
    setGenerateData('no');
    setApiKeyError(null);
    setShowFinetuneDialog(true);
  };

  const handleFinetuneSubmit = async () => {
    if (!selectedModel || !finetuneModelName.trim()) return;

    // Check if synthetic data generation is enabled
    if (generateData === 'yes') {
      // First check if we have an API key stored
      try {
        // Check for stored API key
        const storedKey = await nerService.getOpenAIApiKey();

        if (!storedKey || storedKey.trim() === '') {
          setApiKeyError('OpenAI API key is required for synthetic data generation.');
          setShowApiKeyDialog(true);
          return;
        }

        // Validate the stored API key
        const validation = await nerService.validateOpenAIApiKey(storedKey);

        if (!validation.Valid) {
          setApiKeyError(`Invalid API key: ${validation.Message}`);
          setShowApiKeyDialog(true);
          return;
        }
      } catch (error: any) {
        // Error checking API key
        setApiKeyError('Failed to validate OpenAI API key.');
        setShowApiKeyDialog(true);
        return;
      }
    }

    setFinetuning(true);
    setApiKeyError(null);

    try {
      // Extract unique tags from feedback data
      const uniqueTags = new Set<string>();
      feedbackData.forEach((f) => {
        const labels = f.labels || f.Labels || [];
        labels.forEach((label) => {
          if (label && label !== 'O') {
            uniqueTags.add(label);
          }
        });
      });

      // Create tag info array for synthetic data generation
      const tags = Array.from(uniqueTags).map((tag) => ({
        name: tag,
        description: `Entity of type ${tag}`,
        examples: [], // Backend will extract examples from the samples
      }));

      const request = {
        Name: finetuneModelName.trim(),
        TaskPrompt: finetuneTaskPrompt.trim() || undefined,
        GenerateData: generateData === 'yes',
        Tags: generateData === 'yes' ? tags : undefined,
        Samples:
          feedbackData.length > 0
            ? feedbackData.map((f) => ({
                Tokens: f.tokens || f.Tokens || [],
                Labels: f.labels || f.Labels || [],
              }))
            : undefined,
      };
      await nerService.finetuneModel(selectedModel.Id, request);
      setShowFinetuneDialog(false);
      setFinetuneModelName('');
      setFinetuneTaskPrompt('');
      setGenerateData('no');
      toast.success('Finetuning started successfully!', {
        duration: 3000,
        style: {
          background: '#4CAF50',
          color: '#fff',
          padding: '10px',
          borderRadius: '8px',
        },
        icon: '✓',
      });
    } catch (error: any) {
      // Finetuning failed
      // Check if the error is related to OpenAI API key
      const errorMessage =
        error?.response?.data?.message ||
        error?.response?.data ||
        error?.message ||
        'Finetuning failed';
      const errorString =
        typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage);

      // Check for OpenAI-related errors with various patterns
      const isOpenAIError =
        errorString.toLowerCase().includes('openai') ||
        errorString.toLowerCase().includes('api key') ||
        errorString.toLowerCase().includes('api_key') ||
        errorString.toLowerCase().includes('unauthorized') ||
        errorString.toLowerCase().includes('invalid key');

      if (isOpenAIError && generateData === 'yes') {
        setApiKeyError(errorString);
      } else {
        toast.error(errorString, {
          duration: 5000,
          style: {
            background: '#f44336',
            color: '#fff',
            padding: '10px',
            borderRadius: '8px',
          },
        });
        setShowFinetuneDialog(false);
      }
    } finally {
      setFinetuning(false);
    }
  };

  const handleFinetuneCancel = () => {
    setShowFinetuneDialog(false);
    setFinetuneModelName('');
    setFinetuneTaskPrompt('');
    setGenerateData('no');
    setApiKeyError(null);
  };

  if (!healthStatus) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <Card
        sx={{
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          bgcolor: 'white',
          borderRadius: '12px',
          mx: 'auto',
          maxWidth: '1400px',
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Typography
            variant="h5"
            sx={{
              fontWeight: 600,
              fontSize: '1.5rem',
              color: '#4a5568',
              mb: 4,
            }}
          >
            Model Customization
          </Typography>
          <Box sx={{ maxWidth: 300, mb: 4 }}>
            <Typography
              variant="subtitle2"
              gutterBottom
              sx={{ fontWeight: 600, color: '#475569', mb: 1 }}
            >
              Model
            </Typography>
            <FormControl size="small" fullWidth>
              <Select
                value={selectedModel?.Id || ''}
                displayEmpty
                onChange={handleModelChange}
                renderValue={(val) =>
                  val === ''
                    ? models.length > 0
                      ? models[0].Name
                      : 'Select Model'
                    : models.find((m) => m.Id === val)?.Name
                      ? models
                          .find((m) => m.Id === val)!
                          .Name.charAt(0)
                          .toUpperCase() + models.find((m) => m.Id === val)!.Name.slice(1)
                      : val
                }
                sx={{ bgcolor: '#f8fafc', '&:hover': { bgcolor: '#f1f5f9' } }}
              >
                {models.map((m) => (
                  <MenuItem
                    key={m.Id}
                    value={m.Id}
                    sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        flexGrow: 1,
                      }}
                    >
                      {m.Name.charAt(0).toUpperCase() + m.Name.slice(1)}
                      {m.Status === 'TRAINING' && (
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <CircularProgress size={16} sx={{ ml: 1 }} />
                          <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                            Training...
                          </Typography>
                        </Box>
                      )}
                      {m.Status === 'QUEUED' && (
                        <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                          Queued
                        </Typography>
                      )}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          {selectedModel && (
            <UserFeedbackSection
              feedbackData={feedbackData}
              loadingFeedback={loadingFeedback}
              handleDeleteFeedback={handleDeleteFeedback}
              handleFinetuneClick={handleFinetuneClick}
              showFinetuneDialog={showFinetuneDialog}
              finetuneModelName={finetuneModelName}
              setFinetuneModelName={setFinetuneModelName}
              finetuneTaskPrompt={finetuneTaskPrompt}
              setFinetuneTaskPrompt={setFinetuneTaskPrompt}
              finetuning={finetuning}
              handleFinetuneSubmit={handleFinetuneSubmit}
              handleFinetuneCancel={handleFinetuneCancel}
              availableTags={feedbackData
                .map((f) => f.labels || f.Labels || [])
                .flat()
                .filter((tag): tag is string => tag !== undefined)}
              generateData={generateData}
              setGenerateData={setGenerateData}
              apiKeyError={apiKeyError}
              showApiKeyDialog={showApiKeyDialog}
              setShowApiKeyDialog={setShowApiKeyDialog}
              apiKeyInput={apiKeyInput}
              setApiKeyInput={setApiKeyInput}
              validatingApiKey={validatingApiKey}
              setValidatingApiKey={setValidatingApiKey}
              setApiKeyError={setApiKeyError}
            />
          )}
        </CardContent>
      </Card>
      <Toaster position="top-right" />
    </>
  );
};

export default ModelCustomization;
