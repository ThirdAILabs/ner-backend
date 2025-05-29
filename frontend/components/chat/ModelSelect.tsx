import { HiChip } from 'react-icons/hi';

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
      className={`inline w-full rounded-md px-4 py-2 text-sm border text-left transition-colors flex items-center gap-2
      ${selected ? selectedStyle : unselectedStyle}
      ${disabled ? disabledStyle : enabledStyle}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

interface ModelSelectProps {
  model: string;
  onSelect: (model: string) => void;
}

export default function ModelSelect({ model, onSelect }: ModelSelectProps) {
  return (
    <div className="p-4 flex flex-col gap-1">
      <div className="inline font-semibold p-1">Model Selection</div>
      <div className="flex flex-row w-full gap-2">
        <div className="w-full">
          <ModelOption
            onClick={() => onSelect('gpt-4-mini')}
            selected={model === 'gpt-4-mini'}
          >
            <HiChip className="w-4 h-4" />
            <span>GPT-4o Mini</span>
          </ModelOption>
        </div>
        <div className="w-full">
          <ModelOption
            onClick={() => onSelect('gpt-4')}
            selected={model === 'gpt-4'}
          >
            <HiChip className="w-4 h-4" />
            <span>GPT-4o</span>
          </ModelOption>
        </div>
      </div>
    </div>
  );
}
