interface TooltipProps {
  text: string | null;
  x: number | null;
  y: number | null;
}

export const Tooltip: React.FC<TooltipProps> = ({ text, x, y }) => {
  if (!text || x === null || y === null) return null;

  return (
    <div
      className="fixed z-[1400] px-2 py-1 text-xs bg-white rounded shadow-lg pointer-events-none"
      style={{
        left: x + 10,
        top: y - 30,
      }}
    >
      {text}
    </div>
  );
};
