function ToggleButton({ checked, onChange }: { checked?: boolean; onChange?: () => void }) {
  return (
    <button
      onClick={onChange}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-[rgb(85,152,229)]' : 'bg-gray-200'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  );
}

export default function Toggle({
  checked,
  onChange,
}: {
  checked?: boolean;
  onChange?: () => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-semibold text-gray-600">What GPT sees</span>
      <ToggleButton checked={checked} onChange={onChange} />
    </div>
  );
}
