import React from 'react';
import '../../../styles/pages/modelpage/monitoring_tab.scss';
import Badge from '../../commons/badge';

const MonitoringTab: React.FC = () => {
  return (
    <div className="monitoring-tab">
      <div className="monitoring-header">Dashboard</div>

      <div className="card-row">
        <div className="dashboard-card">
          <div className="card-header">
            MODEL STATUS
            <Badge text="Active" type="success" />
          </div>

          <div className="model-info">
            <div className="model-info-item">
              <div className="model-info-item-label">Deployment URL</div>
              <div className="model-info-item-value">
                https://api.thirdai.com/token-classification
              </div>
            </div>
            <div className="model-info-item">
              <div className="model-info-item-label">Last Updated</div>
              <div className="model-info-item-value">April 28, 2024</div>
            </div>
            <div className="model-info-item">
              <div className="model-info-item-label">Model Version</div>
              <div className="model-info-item-value">v2.1.3</div>
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
