import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';
import { nerService } from '@/lib/backend';

interface ApiKeyDialogProps {
  open: boolean;
  onClose: () => void;
  apiKeyInput: string;
  setApiKeyInput: (value: string) => void;
  apiKeyError: string | null;
  setApiKeyError: (error: string | null) => void;
  validatingApiKey: boolean;
  setValidatingApiKey: (validating: boolean) => void;
  onSuccess: () => void;
}

export function ApiKeyDialog({
  open,
  onClose,
  apiKeyInput,
  setApiKeyInput,
  apiKeyError,
  setApiKeyError,
  validatingApiKey,
  setValidatingApiKey,
  onSuccess,
}: ApiKeyDialogProps) {
  const handleCancel = () => {
    onClose();
    setApiKeyInput('');
  };

  const handleSave = async () => {
    if (!apiKeyInput.trim()) {
      setApiKeyError('Please enter an API key');
      return;
    }

    setValidatingApiKey(true);
    setApiKeyError(null);

    try {
      // Validate the API key
      const validation = await nerService.validateOpenAIApiKey(apiKeyInput);

      if (!validation.Valid) {
        setApiKeyError(validation.Message);
        setValidatingApiKey(false);
        return;
      }

      // Save the API key
      await nerService.setOpenAIApiKey(apiKeyInput);

      // Close the dialog
      onClose();
      setApiKeyInput('');
      setValidatingApiKey(false);

      // Wait a bit for the file to be written
      setTimeout(() => {
        // Now retry the finetune submission
        onSuccess();
      }, 100);
    } catch (error: any) {
      // Error saving API key
      setApiKeyError(`Failed to save API key: ${error?.message || 'Unknown error'}`);
      setValidatingApiKey(false);
    }
  };

  return (
    <Dialog open={open} onClose={() => !validatingApiKey && onClose()} maxWidth="sm" fullWidth>
      <DialogTitle>OpenAI API Key Required</DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1 }}>
          <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
            Synthetic data generation requires an OpenAI API key. Please enter your API key below.
          </Typography>
          {apiKeyError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {apiKeyError}
            </Alert>
          )}
          <TextField
            fullWidth
            label="OpenAI API Key"
            type="password"
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            placeholder="sk-..."
            helperText="Your API key will be stored locally and used for generating synthetic training data."
            disabled={validatingApiKey}
          />
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleCancel} disabled={validatingApiKey} sx={{ textTransform: 'none' }}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={validatingApiKey || !apiKeyInput.trim()}
          sx={{ textTransform: 'none', ml: 1 }}
        >
          {validatingApiKey ? (
            <>
              <CircularProgress size={16} sx={{ mr: 1 }} />
              Validating...
            </>
          ) : (
            'Save and Continue'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
