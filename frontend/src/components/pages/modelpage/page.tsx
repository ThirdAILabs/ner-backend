import React, { useState, useRef, DragEvent } from 'react';
import '../../../styles/pages/_modelpage.scss';
import { Divider } from '@mui/material';
import { FiUpload } from "react-icons/fi";
import { IoCopyOutline } from "react-icons/io5";

interface DocumentItem {
  name: string;
  source: string;
  initiated: string;
  progress: {
    status: string;
    completed?: string;
    current?: number;
    total?: number;
  };
}

const ModelPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('Testing');
  const [modelName, setModelName] = useState<string>('HIPAA 25');
  const [showCopyTooltip, setShowCopyTooltip] = useState(false);
  const [isTooltipFading, setIsTooltipFading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documents: DocumentItem[] = [
    {
      name: 'Daniel Docs 1',
      source: 'S3 Bucket - Daniel Docs 1',
      initiated: '2 months ago',
      progress: {
        status: 'Completed',
        completed: '2 months ago'
      }
    },
    {
      name: 'Daniel Docs 2',
      source: 'S3 Bucket - Daniel Docs 2',
      initiated: '3 days ago',
      progress: {
        status: 'InProgress',
        current: 52,
        total: 92
      }
    }
  ];

  const renderProgress = (progress: DocumentItem['progress']) => {
    if (progress.status === 'Completed') {
      return <span>Completed {progress.completed}</span>;
    } else {
      return (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${(progress.current! / progress.total!) * 100}%` }}
            ></div>
          </div>
          <span className="progress-text">{progress.current}M / {progress.total}M</span>
        </div>
      );
    }
  };

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

  const handleCopy = () => {
    const code = `curl -X POST \\
  https://platform.thirdai.com/Xxfa233zK/query \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d {"text": "Your query text here", "model": "HIPAA-25", "options": {"format": "json"}}`;
    
    navigator.clipboard.writeText(code);
    setShowCopyTooltip(true);
    setIsTooltipFading(false);
    
    setTimeout(() => {
      setIsTooltipFading(true);
      setTimeout(() => {
        setShowCopyTooltip(false);
        setIsTooltipFading(false);
      }, 300); // matches animation duration
    }, 700);
  };

  const handleTabClick = (tab: string) => {
    setActiveTab(tab);
  };
  
  return (
    <div className="main-container">
        <div className="breadcrumb">
          <a href="#" className="home-link">Home</a>
          <span className="breadcrumb-separator"> &gt; </span>
          <span>{modelName}</span>
        </div>
        
        <h1 className="title">{modelName}</h1>
        
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'Monitoring' ? 'active' : ''}`}
            onClick={() => handleTabClick('Monitoring')}
          >
            Monitoring
          </button>
          <button 
            className={`tab ${activeTab === 'Testing' ? 'active' : ''}`}
            onClick={() => handleTabClick('Testing')}
          >
            Testing
          </button>
          <button 
            className={`tab ${activeTab === 'Jobs' ? 'active' : ''}`}
            onClick={() => handleTabClick('Jobs')}
          >
            Jobs
          </button>
        </div>
        
        <div className="content-area">
          
          {activeTab === 'Testing' && (
            <>
              <div className="testing-input-section">
                <input type="text" placeholder="Enter text here to test model..." />
            
                <Divider className='seperator'>or</Divider>
            
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
                      <FiUpload size={30}/>
                      {isDragging ? 'Drop file here' : 'Upload Document here'}
                    </p>
                  </div>
                </div>
              </div>
          
              <div className="testing-api-section">
                <h3>API Reference</h3>
                <div className="api-command">
                  <div className="copy-icon" onClick={handleCopy}>
                    <IoCopyOutline size={20}/>
                    {showCopyTooltip && (
                      <div className={`tooltip ${isTooltipFading ? 'fade-out' : ''}`}>
                        Copied!
                      </div>
                    )}
                  </div>
                  <pre>
                    {`curl -X POST \\
      https://platform.thirdai.com/Xxfa233zK/query \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer YOUR_API_KEY" \\
      -d {"text": "Your query text here", "model": "HIPAA-25", "options": {"format": "json"}}`}
                  </pre>
                </div>
              </div>
            </>
          )}

          { activeTab === 'Monitoring' && (
            <div className="monitoring-section">
              <h3>Monitoring</h3>
              <p>Monitoring content goes here...</p>
            </div>
          )}
          { activeTab === 'Jobs' && (
            <>
                <button className="new-button">New</button>
              <table className="documents-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Source</th>
                    <th>Initiated</th>
                    <th>Progress</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc, index) => (
                    <tr key={index}>
                      <td>{doc.name}</td>
                      <td>{doc.source}</td>
                      <td>{doc.initiated}</td>
                      <td>{renderProgress(doc.progress)}</td>
                      <td><a href="#" className="view-link">View</a></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
    </div>
  );
};

export default ModelPage;