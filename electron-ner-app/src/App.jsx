import React, { useState } from 'react';
import './styles/globals.css';
import TokenClassification from './pages/TokenClassification';

// We'll use a simple routing approach for now
// Later we can integrate a proper router like react-router-dom if needed
function App() {
  // Simple state-based routing
  const [currentPage, setCurrentPage] = useState('home');

  // Function to render the current page
  const renderPage = () => {
    switch (currentPage) {
      case 'token-classification':
        return <TokenClassification />;
      case 'semantic-search':
        return (
          <div>
            <h2 className="text-xl font-semibold mb-4">Semantic Search</h2>
            <p>This is the Semantic Search page (migrated from Next.js)</p>
            {/* Semantic search components would be imported and rendered here */}
          </div>
        );
      default:
        return (
          <div>
            <h2 className="text-xl font-semibold mb-4">Home</h2>
            <p>Welcome to the NER Electron App!</p>
            <p className="mt-2">Select a feature from the sidebar to get started.</p>
          </div>
        );
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 text-white p-4">
        <h1 className="text-2xl font-bold mb-6">NER Electron App</h1>
        <nav>
          <ul className="space-y-2">
            <li>
              <button
                onClick={() => setCurrentPage('home')}
                className={`w-full text-left px-4 py-2 rounded ${currentPage === 'home' ? 'bg-blue-600' : 'hover:bg-gray-700'}`}
              >
                Home
              </button>
            </li>
            <li>
              <button
                onClick={() => setCurrentPage('token-classification')}
                className={`w-full text-left px-4 py-2 rounded ${currentPage === 'token-classification' ? 'bg-blue-600' : 'hover:bg-gray-700'}`}
              >
                Token Classification
              </button>
            </li>
            <li>
              <button
                onClick={() => setCurrentPage('semantic-search')}
                className={`w-full text-left px-4 py-2 rounded ${currentPage === 'semantic-search' ? 'bg-blue-600' : 'hover:bg-gray-700'}`}
              >
                Semantic Search
              </button>
            </li>
          </ul>
        </nav>
      </div>

      {/* Main content */}
      <div className="flex-grow bg-gray-100 p-0 overflow-auto">
        {renderPage()}
      </div>
    </div>
  );
}

export default App; 