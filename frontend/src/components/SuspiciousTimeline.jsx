import React from 'react';
import { AlertTriangle, Film, CheckCircle } from 'lucide-react';

export default function SuspiciousTimeline({ result }) {
  if (!result) return null;

  const { suspicious_frames, frame_details } = result;

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Film size={22} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Frame Timeline & Anomalies</h3>
        </div>

        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          {suspicious_frames.length > 0 ? (
            <span style={{ color: 'var(--status-synthetic)', fontWeight: 600 }}>
              {suspicious_frames.length} Frame(s) Flagged
            </span>
          ) : (
            <span style={{ color: 'var(--status-real)', fontWeight: 600 }}>
              No Suspicious Anomaly Spikes
            </span>
          )}
        </div>
      </div>

      {/* Frame Details Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '12px' }}>
        {frame_details.map((fd) => {
          const isSuspicious = suspicious_frames.includes(fd.frame_number);
          const scorePct = fd.combined_score ? Math.round(fd.combined_score * 100) : 0;

          return (
            <div
              key={fd.frame_number}
              style={{
                background: isSuspicious ? 'rgba(244, 63, 94, 0.12)' : 'rgba(255, 255, 255, 0.03)',
                border: isSuspicious ? '1px solid rgba(244, 63, 94, 0.4)' : '1px solid var(--border-glass)',
                borderRadius: '12px',
                padding: '12px',
                position: 'relative'
              }}
            >
              {isSuspicious && (
                <div style={{ position: 'absolute', top: '8px', right: '8px' }}>
                  <AlertTriangle size={16} color="var(--status-synthetic)" />
                </div>
              )}

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Frame #{fd.frame_number}</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                {fd.timestamp_seconds.toFixed(2)}s
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Score</span>
                <span style={{
                  fontWeight: 700,
                  fontSize: '0.9rem',
                  color: scorePct > 65 ? 'var(--status-synthetic)' : scorePct > 35 ? 'var(--status-potential)' : 'var(--status-real)'
                }}>
                  {scorePct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
