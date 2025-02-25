import { useState, useEffect } from 'react';
import VideoCall from './components/VideoCall';
import './components/VideoCall.css';
import './App.css'; // We'll need to create this file for styling

function App() {
    const [predictedAge, setPredictedAge] = useState(null);

    // Function to handle age prediction from websocket
    const handleAgePrediction = (age) => {
        setPredictedAge(age);
    };

    return (
        <div className="App">
            <h1 className="app-title">Realtime age detector</h1>
            <div className="video-container">
                <VideoCall onAgePrediction={handleAgePrediction} />
            </div>
            <div className="prediction-result">
                <h2>Predicted age: {predictedAge !== null ? predictedAge : "Waiting..."}</h2>
            </div>
        </div>
    );
}

export default App; 