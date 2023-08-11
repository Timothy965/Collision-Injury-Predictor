import React, { useState } from 'react';

function App() {
  const [vehicleType, setVehicleType] = useState('');
  const [speed, setSpeed] = useState('');
  const [roadCondition, setRoadCondition] = useState('');
  const [predictedInjury, setPredictedInjury] = useState(null);

  const handlePrediction = async () => {
    // Here, you would make an API call to your backend to get the predicted injury
    // based on the input data (vehicleType, speed, roadCondition).
    // For demonstration purposes, let's assume the backend returns a predicted injury probability.

    // Replace this with your actual API call logic.
    const predictedInjuryProbability = 0.75;

    setPredictedInjury(predictedInjuryProbability);
  };

  return (
    <div className="App">
      <h1>Collision Injury Predictor</h1>
      <div>
        <label>Vehicle Type:</label>
        <input type="text" value={vehicleType} onChange={(e) => setVehicleType(e.target.value)} />
      </div>
      <div>
        <label>Speed (mph):</label>
        <input type="number" value={speed} onChange={(e) => setSpeed(e.target.value)} />
      </div>
      <div>
        <label>Road Condition:</label>
        <select value={roadCondition} onChange={(e) => setRoadCondition(e.target.value)}>
          <option value="">Select</option>
          <option value="dry">Dry</option>
          <option value="wet">Wet</option>
          <option value="icy">Icy</option>
        </select>
      </div>
      <button onClick={handlePrediction}>Predict Injury</button>
      {predictedInjury !== null && (
        <div>
          <p>Predicted Injury Probability: {predictedInjury}</p>
        </div>
      )}
    </div>
  );
}

export default App;
