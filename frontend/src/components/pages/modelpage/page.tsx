import React, { useState, useRef } from 'react';
import TestingTab from './testing_tab';
import JobTab from './job_tab';
import MonitoringTab from './monitoring_tab';
import CustomizedBreadcrumbs from '../../commons/breadCrumbs';
import '../../../styles/pages/modelpage/_modelpage.scss';
import HomeIcon from '@mui/icons-material/Home';
import ExampleSegmentedControls from '../../commons/tabs';

const ModelPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('Testing');
  const [modelName, setModelName] = useState<string>('HIPAA 25');
  const handleTabClick = (tab: string) => {
    setActiveTab(tab);
  };

  const jobEntries = [
    {
      name: 'Daniel Docs 1',
      source: 'S3 Bucket - Daniel Docs 1',
      initiated: '2 months ago',
      progress: {
        status: 'Completed',
        completed: '2 months ago',
      },
    },
    {
      name: 'Daniel Docs 2',
      source: 'S3 Bucket - Daniel Docs 2',
      initiated: '1 month ago',
      progress: {
        status: 'In Progress',
        current: 50,
        total: 100,
      },
    },
  ];

  //repeating this jobEntries 30 times
  const repeatedJobEntries = Array.from({ length: 30 }, (_, i) => ({
    ...jobEntries[i % jobEntries.length],
    name: `${jobEntries[i % jobEntries.length].name} ${i + 1}`,
  }));

  return (
    <div className="main-container">
      <CustomizedBreadcrumbs
        breadcrumbs={[
          { label: 'Home', href: '/', icon: HomeIcon },
          { label: modelName, href: `/model/${modelName}` },
        ]}
      />

      <h1 className="title">{modelName}</h1>

      <div className="tabs">
        <ExampleSegmentedControls
          tabs={['Monitoring', 'Testing', 'Jobs']}
          value={activeTab}
          onChange={handleTabClick}
        />
      </div>

      <div className="content-area">
        <div className={`tab-content ${activeTab === 'Testing' ? 'active' : ''}`}>
          <TestingTab />
        </div>
        <div className={`tab-content ${activeTab === 'Monitoring' ? 'active' : ''}`}>
          <MonitoringTab />
        </div>
        <div className={`tab-content ${activeTab === 'Jobs' ? 'active' : ''}`}>
          <JobTab jobEntries={repeatedJobEntries} />
        </div>
      </div>
    </div>
  );
};

export default ModelPage;
