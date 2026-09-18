import React, { useState } from 'react';
import { Upload, FileVideo, Play, Sparkles, AlertCircle } from 'lucide-react';

import { API_BASE } from '../config';

export default function VideoUpload({ onAnalysisComplete, loading, setLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleRate, setSampleRate] = useState(10);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setErrorMsg(null);
    setUploadProgress(20);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        const err = await uploadRes.json();
        throw new Error(err.detail || 'Video upload failed');
      }

      setUploadProgress(60);
      const uploadData = await uploadRes.json();

      const analyzeRes = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_id: uploadData.file_id,
          sample_rate: sampleRate,
        }),
      });

      if (!analyzeRes.ok) {
        const err = await analyzeRes.json();
        throw new Error(err.detail || 'Detection analysis failed');
      }

      setUploadProgress(100);
      const resultData = await analyzeRes.json();
      onAnalysisComplete(resultData);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const handleDemoSample = async (videoPath) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const analyzeRes = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoPath,
          sample_rate: sampleRate,
        }),
      });

      if (!analyzeRes.ok) {
        const err = await analyzeRes.json();
        throw new Error(err.detail || 'Demo sample analysis failed');
      }

      const resultData = await analyzeRes.json();
      onAnalysisComplete(resultData);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Video Source & Processing</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Upload custom video file or test demo dataset samples</p>
        </div>

        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => handleDemoSample('data/raw/real_human_face.mp4')}
            disabled={loading}
            className="btn-secondary"
            style={{ fontSize: '0.85rem', borderColor: 'rgba(16, 185, 129, 0.4)' }}
          >
            <Sparkles size={16} color="var(--status-real)" />
            Test Real Human Face
          </button>

          <button
            onClick={() => handleDemoSample('data/raw/authentic_sample.mp4')}
            disabled={loading}
            className="btn-secondary"
            style={{ fontSize: '0.85rem', borderColor: 'rgba(16, 185, 129, 0.4)' }}
          >
            <Sparkles size={16} color="var(--status-real)" />
            Test Authentic Sample
          </button>

          <button
            onClick={() => handleDemoSample('data/raw/synthetic_sample.mp4')}
            disabled={loading}
            className="btn-secondary"
            style={{ fontSize: '0.85rem', borderColor: 'rgba(244, 63, 94, 0.4)' }}
          >
            <Sparkles size={16} color="var(--status-synthetic)" />
            Test Deepfake (Synthetic)
          </button>
        </div>
      </div>

      <div
        style={{
          border: '2px dashed rgba(255, 255, 255, 0.15)',
          borderRadius: '16px',
          padding: '36px 20px',
          textAlign: 'center',
          background: 'rgba(0, 0, 0, 0.2)',
          cursor: 'pointer',
          transition: 'border-color 0.2s ease',
        }}
        onClick={() => document.getElementById('video-file-input').click()}
      >
        <input
          id="video-file-input"
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
          <div style={{ background: 'rgba(0, 242, 254, 0.1)', padding: '16px', borderRadius: '50%' }}>
            <Upload size={32} color="var(--accent-cyan)" />
          </div>
        </div>

        {selectedFile ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <FileVideo size={20} color="var(--accent-cyan)" />
            <span style={{ fontWeight: 600 }}>{selectedFile.name}</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
            </span>
          </div>
        ) : (
          <div>
            <p style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '4px' }}>
              Click or drag video file to upload
            </p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Supports MP4, MOV, WEBM, AVI (Max 500MB)
            </p>
          </div>
        )}
      </div>

      {errorMsg && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--status-synthetic)', marginTop: '12px', fontSize: '0.85rem' }}>
          <AlertCircle size={16} />
          <span>{errorMsg}</span>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Frame Sampling Rate:</label>
          <select
            value={sampleRate}
            onChange={(e) => setSampleRate(Number(e.target.value))}
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-glass)',
              color: 'var(--text-primary)',
              borderRadius: '8px',
              padding: '6px 12px',
              outline: 'none',
            }}
          >
            <option value={5} style={{ background: '#121826' }}>1 frame every 5 frames (High Density)</option>
            <option value={10} style={{ background: '#121826' }}>1 frame every 10 frames (Standard)</option>
            <option value={15} style={{ background: '#121826' }}>1 frame every 15 frames (Fast)</option>
            <option value={30} style={{ background: '#121826' }}>1 frame every 30 frames (Ultra Fast)</option>
          </select>
        </div>

        <button
          onClick={handleUploadAndAnalyze}
          disabled={!selectedFile || loading}
          className="btn-primary"
        >
          {loading ? (
            <>
              <div className="animate-pulse-slow">Processing Deepfake Pipeline...</div>
            </>
          ) : (
            <>
              <Play size={18} fill="#000" />
              Start Multi-Signal Analysis
            </>
          )}
        </button>
      </div>
    </div>
  );
}
