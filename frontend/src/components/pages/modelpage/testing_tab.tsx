import React, { useState, useRef, DragEvent } from 'react';
import { Divider } from '@mui/material';
import { FiUpload } from 'react-icons/fi';
import CopyButton from '../../common/copyButton';
import '../../../styles/pages/modelpage/testing_tab.scss';

const TestingTab: React.FC = () => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const file = files[0];
    // TODO: Handle file upload logic here
    console.log('Selected file:', file);
  };

  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const api_command = `curl -X POST \\
      https://platform.thirdai.com/Xxfa233zK/query \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer YOUR_AUTHORIZATION_TOKEN" \\
      -d {"text": "Your query text here", "model": "HIPAA-25", "options": {"format": "json"}}`;

  return (
    <div className="testing-container">
      <div className="testing-input-section">
        <input type="text" placeholder="Enter text here to test model..." />

        <Divider className="seperator">or</Divider>

        <div className="upload-area">
          <div
            className={`upload-box ${isDragging ? 'dragging' : ''}`}
            onClick={handleUploadClick}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={(e) => handleFileSelect(e.target.files)}
            />
            <p>
              <FiUpload size={30} />
              {isDragging ? 'Drop file here' : 'Upload Document here'}
            </p>
          </div>
        </div>
      </div>

      <div className="testing-api-section">
        <h3>API Reference</h3>
        <div className="api-command">
          <CopyButton code={api_command} tooltipText='copied!'/>
          <pre>
            {api_command}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default TestingTab;
