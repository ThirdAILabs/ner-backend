import { ChangeEvent, useEffect, useRef, useState } from 'react';
import { Tooltip } from '@mui/material';
import { Lock, MessageSquare } from 'lucide-react';
import SaveAndCancel from './SaveAndCancel';

interface ModelOptionProps {
  children: React.ReactNode;
  onClick: () => void;
  selected: boolean;
  disabled?: boolean;
}

function ModelOption({ children, onClick, selected, disabled }: ModelOptionProps) {
  const selectedStyle =
    'bg-[rgb(85,152,229)]/10 text-[rgb(85,152,229)] font-semibold border-[rgb(85,152,229)]';
  const unselectedStyle = 'text-gray-700 border-gray-200';
  const disabledStyle = 'cursor-not-allowed';
  const enabledStyle = 'hover:bg-[rgb(85,152,229)]/5';
  return (
    <button
      type="button"
      className={`inline w-full rounded-sm px-4 py-2 text-sm border text-left transition-colors flex items-center gap-2
      ${selected ? selectedStyle : unselectedStyle}
      ${disabled ? disabledStyle : enabledStyle}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

interface OptionsDropdownProps {
  handleBasicMode: () => void;
  handleAdvancedMode: () => void;
  apiKey: string;
  invalidApiKey: boolean;
  onEditApiKey: () => void;
  onSaveApiKey: (key: string) => void;
  onCancelApiKey: () => void;
}

export default function Options({
  handleBasicMode,
  handleAdvancedMode,
  apiKey,
  invalidApiKey,
  onEditApiKey,
  onSaveApiKey,
  onCancelApiKey,
}: OptionsDropdownProps) {
  const apiKeyRef = useRef<HTMLInputElement>(null);
  const [editingApiKey, setEditingApiKey] = useState<boolean>(false);
  const [intermediateApiKey, setIntermediateApiKey] = useState<string>(apiKey);
  const [showInvalidKeyError, setShowInvalidKeyError] = useState<boolean>(invalidApiKey);

  const promptInvalidKey = () => {
    setShowInvalidKeyError(true);
    setEditingApiKey(true);
    apiKeyRef.current?.focus();
  };

  const doneEditing = () => {
    setShowInvalidKeyError(false);
    setEditingApiKey(false);
    apiKeyRef.current?.blur();
  };

  useEffect(() => {
    setIntermediateApiKey(apiKey);
  }, [apiKey]);

  useEffect(() => {
    setShowInvalidKeyError(invalidApiKey);
    if (invalidApiKey) {
      promptInvalidKey();
    }
  }, [invalidApiKey]);

  const handleSaveApiKey = () => {
    if (intermediateApiKey === '' || (intermediateApiKey === apiKey && invalidApiKey)) {
      promptInvalidKey();
      return;
    }
    doneEditing();
    onSaveApiKey(intermediateApiKey);
  };

  const handleCancelApiKey = () => {
    setIntermediateApiKey(apiKey);
    if (apiKey === '' || invalidApiKey) {
      promptInvalidKey();
      return;
    }
    doneEditing();
    onCancelApiKey();
  };

  const handleApiKeyChange = (e: ChangeEvent<HTMLInputElement>) => {
    setEditingApiKey(true);
    setShowInvalidKeyError(false);
    setIntermediateApiKey(e.target.value);
    onEditApiKey();
  };

  const handleBlur = () => {
    if (invalidApiKey || editingApiKey) {
      apiKeyRef.current?.focus();
    }
  };

  return (
    <div className="w-full bg-white rounded-md shadow-lg border border-gray-200">
      <div className="p-2 flex flex-col gap-1">
        <div className="inline font-semibold p-1">Redaction Model</div>

        <div className="flex flex-row w-full gap-2">
          <div className="w-full">
            <ModelOption onClick={handleBasicMode} selected={true}>
              <MessageSquare className="w-4 h-4" />
              <span>Basic</span>
            </ModelOption>
          </div>
          <div className="w-full">
            <Tooltip title="Requires pro subscription. Email us at contact@thirdai.com">
              {/* Need span because tooltip child cannot be a custom react component. */}
              <span className="w-full">
                <ModelOption onClick={handleAdvancedMode} selected={false} disabled>
                  <Lock className="w-4 h-4" />
                  <span>Advanced</span>
                </ModelOption>
              </span>
            </Tooltip>
          </div>
        </div>

        <div className="inline font-semibold p-1">OpenAI Key</div>
        <div className="flex flex-row w-full gap-2">
          <input
            type="text"
            ref={apiKeyRef}
            value={intermediateApiKey}
            onChange={handleApiKeyChange}
            onBlur={handleBlur}
            placeholder="Your API key here..."
            className="flex-1 px-4 py-2 min-w-[100px] border-[1px] rounded-md"
          />
          {editingApiKey && (
            <SaveAndCancel onSave={handleSaveApiKey} onCancel={handleCancelApiKey} />
          )}
        </div>
        <div className="block h-5 text-red-500 text-sm p-1 pb-5">
          {showInvalidKeyError ? 'The OpenAI key is invalid.' : ''}
        </div>
      </div>
    </div>
  );
}
