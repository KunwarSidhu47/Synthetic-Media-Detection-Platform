import React, { useState } from 'react';
import Navbar from './components/Navbar';
import VideoUpload from './components/VideoUpload';
import PredictionHero from './components/PredictionHero';
import SignalBreakdownChart from './components/SignalBreakdownChart';
import SuspiciousTimeline from './components/SuspiciousTimeline';
import EvidenceReport from './components/EvidenceReport';

export default function App() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', paddingBottom: '40px' }}>
      <Navbar />

      <main style={{ padding: '0 24px' }}>
        <VideoUpload
          onAnalysisComplete={(result) => setAnalysisResult(result)}
          loading={loading}
          setLoading={setLoading}
        />

        {analysisResult && (
          <>
            <PredictionHero result={analysisResult} />
            <SignalBreakdownChart result={analysisResult} />
            <SuspiciousTimeline result={analysisResult} />
            <EvidenceReport result={analysisResult} />
          </>
        )}
      </main>

      <footer style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', padding: '20px' }}>
        Synthetic Media Detection & Analysis Platform &copy; 2026. Built with PyTorch, ViT, 2D FFT, Bi-LSTM, FastAPI & React.
      </footer>
    </div>
  );
}
