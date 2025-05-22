function SaveButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="p-1.5 rounded-full border border-black bg-black hover:border-[rgb(85,152,229)] hover:bg-[rgb(85,152,229)] text-white transition-colors"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20 6L9 17l-5-5" />
      </svg>
    </button>
  );
}

function CancelButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="p-1.5 rounded-full border border-black text-black hover:border-red-500 hover:text-red-500 transition-colors"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M18 6L6 18M6 6l12 12" />
      </svg>
    </button>
  );
}

interface SaveAndCancelProps {
  onSave: () => void;
  onCancel: () => void;
}

export default function SaveAndCancel({
  onSave,
  onCancel,
}: SaveAndCancelProps) {
  return <div className="block w-[68px]">
      <div className="flex gap-2 h-full items-center">
        <SaveButton onClick={onSave} />
        <CancelButton onClick={onCancel} />
      </div>
    </div>
}