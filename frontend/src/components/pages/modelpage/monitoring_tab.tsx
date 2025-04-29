import React from 'react';
import '../../../styles/pages/modelpage/monitoring_tab.scss';
import CopyButton from '../../common/copyButton';

const MonitoringTab: React.FC = () => {
  return (
    <div className="monitoring-tab">
      <div className="monitoring-header">
        <h2>Dashboard</h2>
      </div>
      
      <div className="card-row">
        <div className="dashboard-card status-card">
          <div className="card-header">
            <h3 className="card-title">MODEL STATUS</h3>
            <span className="badge badge-success">Active</span>
          </div>
          
          <div className="model-info">
            <div className="model-info-item">
              <div className="model-info-label">Deployment URL</div>
              <div className="model-info-value with-copy">
                <span>https://api.thirdai.com/token-classification</span>
                <CopyButton code='https://api.thirdai.com/token-classification' iconSize={15}/>
              </div>
            </div>
            <div className="model-info-item">
              <div className="model-info-label">Last Updated</div>
              <div className="model-info-value">April 28, 2024</div>
            </div>
            <div className="model-info-item">
              <div className="model-info-label">Model Version</div>
              <div className="model-info-value">v2.1.3</div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="dashboard-grid">
        <div className="dashboard-card">
          <div className="card-header">
            <h3 className="card-title">PRECISION</h3>
          </div>
          <div className="metric-value">0.97</div>
        </div>
        <div className="dashboard-card">
          <div className="card-header">
            <h3 className="card-title">RECALL</h3>
          </div>
          <div className="metric-value">0.92</div>
        </div>
        
        <div className="dashboard-card">
          <div className="card-header">
            <h3 className="card-title">F1 SCORE</h3>
          </div>
          <div className="metric-value">0.94</div>
        </div>
      </div>
    </div>
  );
};

export default MonitoringTab;