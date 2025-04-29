import { useState } from 'react';
import '../../../styles/pages/_home.scss';

function AppCatalog() {
    const [selectedRow, setSelectedRow] = useState<number | null>(null);
    const detectionTools: AppInfo[] = [
        {
            name: 'HIPAA 15',
            description: 'Detects 15 types of info...',
            created: '2 months ago',
            modified: '1 week ago',
            owner: 'Daniel Hapsburg',
        },
        {
            name: 'HIPAA 25',
            description: 'Detects 25 types of info...',
            created: '1 month ago',
            modified: 'Yesterday',
            owner: 'Daniel Hapsburg',
        },
        {
            name: 'PII 15',
            description: 'Detects 15 types of info...',
            created: '2 months ago',
            modified: '1 week ago',
            owner: 'Daniel Hapsburg',
        },
        {
            name: 'PII 25',
            description: 'Detects 25 types of info...',
            created: '1 month ago',
            modified: 'Yesterday',
            owner: 'Daniel Hapsburg',
        },
    ];
    return (
        <div className="container">
            <div className="main-content">
                <div className="header">
                    <button className="new-button">New</button>
                </div>
                <div className="table-container">
                    <table className="table">
                        <thead>
                            <tr>
                                <th className="table-header">Name</th>
                                <th className="table-header">Description</th>
                                <th className="table-header">Created</th>
                                <th className="table-header">Modified</th>
                                <th className="table-header">Owner</th>
                                <th className="table-header text-center">Go</th>
                            </tr>
                        </thead>
                        <tbody>
                            {detectionTools.map((tool, index) => (
                                <tr
                                    key={index}
                                    className={`row ${index % 2 === 0 ? 'alternate' : ''}`}
                                    onClick={() => setSelectedRow(index)}
                                >
                                    <td className="name-cell">{tool.name}</td>
                                    <td className="cell">{tool.description}</td>
                                    <td className="cell">{tool.created}</td>
                                    <td className="cell">{tool.modified}</td>
                                    <td className="cell">{tool.owner}</td>
                                    <td className="center-cell">
                                        <div className="triangle"></div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th className="table-header">Name</th>
                <th className="table-header">Description</th>
                <th className="table-header">Created</th>
                <th className="table-header">Modified</th>
                <th className="table-header">Owner</th>
                <th className="table-header text-center">Go</th>
              </tr>
            </thead>
            <tbody>
              {detectionTools.map((tool, index) => (
                <tr
                  key={index}
                  className={`row ${selectedRow === index ? 'selected' : index % 2 === 0 ? 'alternate' : ''}`}
                  onClick={() => setSelectedRow(index)}
                >
                  <td className="name-cell">{tool.name}</td>
                  <td className="cell">{tool.description}</td>
                  <td className="cell">{tool.created}</td>
                  <td className="cell">{tool.modified}</td>
                  <td className="cell">{tool.owner}</td>
                  <td className="center-cell">
                    <div className="triangle"></div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
export default AppCatalog;
