import React from 'react';
import '../../styles/components/_badge.scss';

const Badge: React.FC<{
  text: string;
  type: 'success' | 'error' | 'warning' | 'info';
  className?: string;
}> = ({ text, type, className }) => {
  return <span className={`badge ${type} ${className}`}>{text}</span>;
};
export default Badge;
