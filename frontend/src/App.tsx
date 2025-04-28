import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import AppCatalog from './components/pages/home/page';
import ModelPage from './components/pages/modelpage/page';

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

function AppContent() {
  return (
    <Routes>
      <Route path="/" element={<AppCatalog />} />
      <Route path="/model-page" element={<ModelPage />} />
      <Route path="/testing" />
      <Route path="/error" />
    </Routes>
  );
}
