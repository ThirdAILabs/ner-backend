import { useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface SidePanelProps {
    width?: number;
    showPanel: boolean;
    onTogglePanel: () => void;
    children: React.ReactNode;
    className?: string;
    borderPosition?: 'left' | 'right';
    expandedOffset: string;
    collapsedOffset: string;
}

export function SidePanel({
    width = 280,
    showPanel,
    onTogglePanel,
    children,
    className = '',
    borderPosition = 'right',
    expandedOffset,
    collapsedOffset,
}: SidePanelProps) {
    const [showShadow, setShowShadow] = useState(false);
    const contentRef = useRef<HTMLDivElement>(null);

    const handleScroll = () => {
        if (contentRef.current) {
            setShowShadow(contentRef.current.scrollTop > 0);
        }
    };

    return (
        <div
            className={`flex flex-col relative ${showPanel ? `w-[${width}px]` : 'w-0'
                } ${showPanel && borderPosition === 'right' ? 'border-r' : ''} ${showPanel && borderPosition === 'left' ? 'border-l' : ''
                } ${className}`}
        >
            {/* Toggle Button */}
            <button
                onClick={onTogglePanel}
                className={`absolute ${borderPosition === 'right'
                    ? `${showPanel ? `-right-[${expandedOffset}]` : `-right-[${collapsedOffset}]`}`
                    : `-left-[${expandedOffset}]`
                    } top-9 transform -translate-y-1/2 rounded-full border border-gray-200 bg-white p-1 hover:bg-gray-50 transition-colors z-20`}
                aria-label={showPanel ? 'Collapse panel' : 'Expand panel'}
            >
                {showPanel ? (
                    borderPosition === 'right' ? (
                        <ChevronLeft className="h-4 w-4 text-gray-600" />
                    ) : (
                        <ChevronRight className="h-4 w-4 text-gray-600" />
                    )
                ) : (
                    borderPosition === 'right' ? (
                        <ChevronRight className="h-4 w-4 text-gray-600" />
                    ) : (
                        <ChevronLeft className="h-4 w-4 text-gray-600" />
                    )
                )}
            </button>

            {/* Content Area */}
            {showPanel && (
                <div
                    ref={contentRef}
                    className="flex-1 overflow-y-auto"
                    onScroll={handleScroll}
                    style={{
                        boxShadow: showShadow ? 'inset 0 4px 6px -4px rgba(0, 0, 0, 0.1)' : 'none',
                    }}
                >
                    {children}
                </div>
            )}
        </div>
    );
}