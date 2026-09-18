import React, { useState, useEffect } from 'react';
import { ShieldAlert, Activity, CheckCircle2 } from 'lucide-react';

import { API_BASE } from '../config';

export default function Navbar() {
  const [health, setHealth] = useState({ status: 'checking', torch_device: 'cpu' });

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'offline', torch_device: 'unknown' }));
  }, []);

  return (
    <header className="glass-card" style={{ margin: '16px 24px', padding: '16px 28px', borderRadius: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo))',
            padding: '10px',
            borderRadius: '12px',
            display: 'flex'
          }}>
            <ShieldAlert size={26} color="#000" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, background: 'linear-gradient(90deg, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Synthetic Media Detection & Analysis
            </h1>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Multimodal Deepfake Forensic Platform
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(255, 255, 255, 0.04)',
            padding: '6px 14px',
            borderRadius: '20px',
            border: '1px solid var(--border-glass)',
            fontSize: '0.85rem'
          }}>
            <Activity size={16} color="var(--accent-cyan)" />
            <span style={{ color: 'var(--text-secondary)' }}>Device:</span>
            <span style={{ fontWeight: 600, color: 'var(--accent-cyan)', textTransform: 'uppercase' }}>
              {health.torch_device}
            </span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.85rem',
            color: health.status === 'healthy' ? 'var(--status-real)' : 'var(--status-synthetic)'
          }}>
            <CheckCircle2 size={16} />
            <span style={{ textTransform: 'capitalize', fontWeight: 500 }}>{health.status}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
