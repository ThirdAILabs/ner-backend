interface ModelCardProps {
  title: string;
  description: React.ReactNode;
  isSelected?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

const ModelCard: React.FC<ModelCardProps> = ({
  title,
  description,
  isSelected = false,
  disabled = false,
  onClick,
}) => (
  <div
    className={`relative p-6 rounded-2xl border transition-colors
            ${isSelected ? 'border-[#5598E5]' : 'border-gray-200'} 
            ${
              disabled
                ? 'opacity-85 cursor-not-allowed bg-gray-50'
                : 'cursor-pointer hover:border-gray-300'
            }
        `}
    onClick={() => !disabled && onClick()}
  >
    <h3 className="text-md font-medium text-gray-800 mb-2">{title}</h3>
    <div className="text-sm text-gray-500">{description}</div>
  </div>
);

export default ModelCard;
