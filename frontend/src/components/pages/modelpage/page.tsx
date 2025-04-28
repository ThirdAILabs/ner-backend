import React, { useState, useRef } from 'react';
import TestingTab from './testing_tab';
import '../../../styles/pages/modelpage/_modelpage.scss';
import JobTab from './job_tab';

const ModelPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('Testing');
  const [modelName, setModelName] = useState<string>('HIPAA 25');
  const handleTabClick = (tab: string) => {
    setActiveTab(tab);
  };

  return (
    <div className="main-container">
      <div className="breadcrumb">
        <a href="#" className="home-link">
          Home
        </a>
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
        {activeTab === 'Testing' && <TestingTab />}
        {activeTab === 'Monitoring' && (
          <div className="monitoring-section">
            <h3>Monitoring</h3>
            <p>Monitoring content goes here...</p>
          </div>
        )}
        {activeTab === 'Jobs' && <JobTab />}
      </div>
    </div>
  );
};

export default ModelPage;
