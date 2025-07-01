import React from 'react';

interface CustomisableCardProps {
    children: React.ReactNode;
    backgroundImage?: string;
    width?: string;
    height?: string;
    className?: string;
}

const CustomisableCard: React.FC<CustomisableCardProps> = ({
    children,
    backgroundImage,
    width = '100%',
    height = 'auto',
    className = '',
}) => {
    return (
        <div
            className={`
        relative
        border-[1px]
        border-gray-200
        rounded-xl
        overflow-hidden
        shadow-sm
        transition-all
        duration-300
        ${className}
      `}
            style={{
                width,
                height,
                backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'none',
                backgroundSize: 'cover',
                backgroundPosition: 'center',
            }}
        >
            {children}
        </div>
    );
};

export default CustomisableCard;