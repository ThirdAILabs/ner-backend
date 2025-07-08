import React from 'react';

interface SourceCardProps {
    logo: React.ReactNode;
    title: string;
    subtitle: string;
    info: string;
    selected?: boolean;
}

const SourceCard: React.FC<SourceCardProps> = ({ logo, title, subtitle, info, selected = false }) => {
    return (
        <div
            className={`flex flex-col items-center p-8 rounded-2xl font-['Plus_Jakarta_Sans'] border ${selected ? 'border-black' : 'border-gray-200'} hover:border-gray-300 transition-colors cursor-pointer`}
            style={{ width: '408px', height: '264px' }}>
            {/* Icon Container */}
            <div className="w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6">
                {logo}
            </div>

            {/* Text Content */}
            <div className="text-center">
                <h3 className="text-xl font-medium text-gray-500 mb-2">
                    {title}
                </h3>
                <p className="text-gray-500 mb-2">
                    {subtitle}
                </p>
                <p className="text-sm text-gray-400">
                    {info}
                </p>
            </div>
        </div>
    );
};

export default SourceCard;


