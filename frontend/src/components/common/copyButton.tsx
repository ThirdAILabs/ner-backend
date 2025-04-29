import React from 'react';
import { IoCopyOutline } from 'react-icons/io5';
import '../../styles/components/_copybutton.scss';

const CopyButton: React.FC<{ code: string; iconSize?: number, tooltipText?: string; className?: string }> = ({ 
    code, 
    iconSize = 20,
    tooltipText = "",
}) => {
    const [showCopyTooltip, setShowCopyTooltip] = React.useState(false);
    const [isTooltipFading, setIsTooltipFading] = React.useState(false);
    
    const handleCopy = () => {
        navigator.clipboard.writeText(code);
        setShowCopyTooltip(true);
        setIsTooltipFading(false);
    
        setTimeout(() => {
            setIsTooltipFading(true);
            setTimeout(() => {
                setShowCopyTooltip(false);
                setIsTooltipFading(false);
            }, 300);
        }, 700);
    };
    
    return (
        <div className={`copy-button-container`}>
            <button className="copy-button" onClick={handleCopy}>
                <IoCopyOutline size={iconSize} />
            </button>
            {showCopyTooltip && tooltipText && (
                <div className={`tooltip ${isTooltipFading ? 'fade-out' : ''}`}>
                    {tooltipText}
                </div>
            )}
        </div>
    );
}

export default CopyButton;
