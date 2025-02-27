import React, { useState, useRef } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [models, setModels] = useState('');
  const [numResponses, setNumResponses] = useState(1);
  const [reference, setReference] = useState('');
  const [status, setStatus] = useState([]);       // List of status messages
  const [reportUrl, setReportUrl] = useState(null); // URL of the generated report to display
  const wsRef = useRef(null);

  const handleStart = () => {
    // Clear previous status and report
    setStatus([]);
    setReportUrl(null);

    // Open WebSocket connection to backend
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      // Send generation parameters to backend when connection opens
      const payload = {
        query: query,
        models: models,         // e.g. "model1,model2"
        n: parseInt(numResponses, 10),
        answer: reference
      };
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.event === 'status' || msg.event === 'progress') {
        // Update status messages list with both status and progress updates
        setStatus(prev => [...prev, msg.message]);
      } else if (msg.event === 'done') {
        // Report generation finished; set the report URL (served by backend)
        setStatus(prev => [...prev, 'Report generation completed.']);
        setReportUrl(`http://${window.location.hostname}:8000/reports/${msg.filename}`);
        ws.close();
      } else if (msg.event === 'error') {
        setStatus(prev => [...prev, 'Error: ' + msg.message]);
        ws.close();
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setStatus(prev => [...prev, 'Error during report generation.']);
    };
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>LLM Comparison Tool</h1>
      <div style={{ marginBottom: '20px', maxWidth: '600px' }}>
        <label><strong>Query:</strong></label><br/>
        <textarea 
          value={query} 
          onChange={e => setQuery(e.target.value)} 
          rows={3} 
          style={{ width: '100%' }} 
        />
        <br/><br/>
        <label><strong>Models (comma-separated):</strong></label><br/>
        <input 
          type="text" 
          value={models} 
          onChange={e => setModels(e.target.value)} 
          style={{ width: '100%' }} 
          placeholder='e.g. model1:latest, model2:latest' 
        />
        <br/><br/>
        <label><strong>Responses per model (n):</strong></label><br/>
        <input 
          type="number" 
          min="1" 
          value={numResponses} 
          onChange={e => setNumResponses(e.target.value)} 
        />
        <br/><br/>
        <label><strong>Reference Answer (optional):</strong></label><br/>
        <textarea 
          value={reference} 
          onChange={e => setReference(e.target.value)} 
          rows={2} 
          style={{ width: '100%' }} 
          placeholder='Provide an expected answer for metrics (optional)' 
        />
        <br/><br/>
        <button onClick={handleStart} style={{ padding: '10px 20px' }}>
          Generate Report
        </button>
      </div>

      {/* Display real-time status updates */}
      {status.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3>Progress:</h3>
          <ul>
            {status.map((msg, idx) => <li key={idx}>{msg}</li>)}
          </ul>
        </div>
      )}

      {/* Display the final report in an iframe once ready */}
      {reportUrl && (
        <div>
          <h2>Comparative Report:</h2>
          <iframe 
            src={reportUrl} 
            title="LLM Comparison Report" 
            style={{ width: '100%', height: '600px', border: '1px solid #ccc' }} 
          />
        </div>
      )}
    </div>
  );
}

export default App;
