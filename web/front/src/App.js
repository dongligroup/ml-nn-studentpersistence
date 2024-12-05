import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import PredictionForm from './components/PredictionForm'; 
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <h1></h1>
        
        {/* Rsetup */}
        <Routes>
          {/* route to render the form component */}
          <Route path="/" element={<PredictionForm />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
