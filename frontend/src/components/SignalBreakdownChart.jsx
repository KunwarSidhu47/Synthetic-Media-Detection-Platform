import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

export default function SignalBreakdownChart({ result }) {
  if (!result) return null;

  const data = [
    { name: 'Spatial (ViT)', score: Math.round(result.spatial_score * 100), color: '#00f2fe' },
    { name: 'Frequency (FFT)', score: Math.round(result.frequency_score * 100), color: '#a855f7' },
    { name: 'Temporal (LSTM)', score: Math.round(result.temporal_score * 100), color: '#3b82f6' },
  ];

  const radarData = [
    { subject: 'Spatial Artifacts', A: Math.round(result.spatial_score * 100) },
    { subject: 'Spectral Anomaly', A: Math.round(result.frequency_score * 100) },
    { subject: 'Inter-frame Flicker', A: Math.round(result.temporal_score * 100) },
    { subject: 'Confidence', A: Math.round(result.confidence * 100) },
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '24px', marginBottom: '24px' }}>
      
      {/* Bar Chart */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '16px' }}>
          Modal Signal Confidence Comparison
        </h3>
        
        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 40, bottom: 10 }}>
              <XAxis type="number" domain={[0, 100]} stroke="#64748b" tickFormatter={(v) => `${v}%`} />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" tick={{ fontSize: 12 }} width={110} />
              <Tooltip
                contentStyle={{ background: '#090d16', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                formatter={(val) => [`${val}%`, 'Synthetic Score']}
              />
              <Bar dataKey="score" radius={[0, 8, 8, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar Chart */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '16px' }}>
          Forensic Signal Profile
        </h3>

        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart outerRadius={80} data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis dataKey="subject" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#64748b" />
              <Radar name="Detection Signals" dataKey="A" stroke="#00f2fe" fill="#00f2fe" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
