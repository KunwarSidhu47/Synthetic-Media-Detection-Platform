import React from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Layers, Waves, Clock } from 'lucide-react';

export default function PredictionHero({ result }) {
  if (!result) return null;

  const { prediction, confidence, final_score, spatial_score, frequency_score, temporal_score, video_metadata } = result;

  const getBadge = () => {
    if (prediction.includes('Synthetic') && !prediction.includes('Potentially')) {
      return (
        <span className="badge-synthetic">
          <ShieldAlert size={18} />
          {prediction}
        </span>
      );
    }
    if (prediction.includes('Potentially')) {
      return (
        <span className="badge-potential">
          <AlertTriangle size={18} />
          {prediction}
        </span>
      );
    }
    return (
      <span className="badge-real">
        <ShieldCheck size={18} />
        {prediction}
      </span>
    );
  };

  const confidencePct = Math.round(confidence * 100);
  const scorePct = Math.round(final_score * 100);

  return (
    <div className="glass-card glass-card-glow" style={{ padding: '28px', marginBottom: '24px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', alignItems: 'center' }}>
        
        {/* Prediction Status & Score */}
        <div>
          <div style={{ marginBottom: '12px' }}>{getBadge()}</div>
          
          <h2 style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '6px' }}>
            {scorePct}% <span style={{ fontSize: '1rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Synthetic Score</span>
          </h2>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '20px' }}>
            Confidence Assessment: <strong style={{ color: '#fff' }}>{confidencePct}%</strong> based on fused multimodal signals.
          </p>

          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 16px', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Spatial (ViT)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {Math.round(spatial_score * 100)}%
              </div>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 16px', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Frequency (FFT)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                {Math.round(frequency_score * 100)}%
              </div>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px 16px', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Temporal (Bi-LSTM)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-blue)' }}>
                {Math.round(temporal_score * 100)}%
              </div>
            </div>
          </div>
        </div>

        {/* Metadata Specs & Video Player */}
        <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '20px', borderRadius: '16px', border: '1px solid var(--border-glass)' }}>
          {result.video_url && (
            <div style={{ marginBottom: '16px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-glass)' }}>
              <video
                src={result.video_url}
                controls
                autoPlay
                loop
                muted
                style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', background: '#000', display: 'block' }}
              />
            </div>
          )}

          <h3 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '16px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Video Specification
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', fontSize: '0.85rem' }}>
            <div>
              <div style={{ color: 'var(--text-muted)' }}>Duration</div>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Clock size={14} color="var(--accent-cyan)" />
                {video_metadata.duration_seconds.toFixed(1)}s
              </div>
            </div>

            <div>
              <div style={{ color: 'var(--text-muted)' }}>Frame Rate</div>
              <div style={{ fontWeight: 600 }}>{video_metadata.fps} FPS</div>
            </div>

            <div>
              <div style={{ color: 'var(--text-muted)' }}>Resolution</div>
              <div style={{ fontWeight: 600 }}>{video_metadata.width} x {video_metadata.height}</div>
            </div>

            <div>
              <div style={{ color: 'var(--text-muted)' }}>Total Frames</div>
              <div style={{ fontWeight: 600 }}>{video_metadata.total_frames}</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
