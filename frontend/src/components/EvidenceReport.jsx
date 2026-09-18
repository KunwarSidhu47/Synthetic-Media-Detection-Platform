import React, { useState, useEffect } from 'react';
import { FileText, Download, ShieldCheck, Info, Cpu, Waves, Clock, Sparkles, AlertTriangle } from 'lucide-react';

import { API_BASE } from '../config';

export default function EvidenceReport({ result }) {
  const [activeTab, setActiveTab] = useState('llm');
  const [llmExplanation, setLlmExplanation] = useState(null);
  const [llmLoading, setLlmLoading] = useState(false);

  useEffect(() => {
    if (result && result.analysis_id) {
      setLlmLoading(true);
      fetch(`${API_BASE}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_id: result.analysis_id }),
      })
        .then((res) => res.json())
        .then((data) => setLlmExplanation(data))
        .catch((err) => console.error('Error fetching LLM explanation:', err))
        .finally(() => setLlmLoading(false));
    }
  }, [result]);

  if (!result) return null;

  const { evidence_summary } = result;

  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `deepfake_analysis_${result.analysis_id.slice(0, 8)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleExportMarkdownReport = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_id: result.analysis_id }),
      });
      const data = await res.json();
      const dataStr = "data:text/markdown;charset=utf-8," + encodeURIComponent(data.report_markdown);
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `forensic_report_${result.analysis_id.slice(0, 8)}.md`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    } catch (err) {
      console.error('Failed to export markdown report:', err);
    }
  };

  return (
    <div className="glass-card" style={{ padding: '24px', marginBottom: '36px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <FileText size={22} color="var(--accent-purple)" />
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>AI Forensic Analysis & Responsible AI Report</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Interpretable natural language explanation layer & evidence metadata</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={handleExportMarkdownReport} className="btn-secondary" style={{ fontSize: '0.85rem' }}>
            <FileText size={16} color="var(--accent-cyan)" />
            Export Markdown Report
          </button>
          <button onClick={handleExportJSON} className="btn-primary" style={{ fontSize: '0.85rem' }}>
            <Download size={16} />
            Export JSON Data
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-glass)', paddingBottom: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
        <button
          onClick={() => setActiveTab('llm')}
          style={{
            background: activeTab === 'llm' ? 'rgba(0, 242, 254, 0.15)' : 'transparent',
            border: activeTab === 'llm' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
            color: activeTab === 'llm' ? '#fff' : 'var(--text-secondary)',
            padding: '8px 16px',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Sparkles size={16} color="var(--accent-cyan)" /> AI Natural Language Explanation
        </button>

        <button
          onClick={() => setActiveTab('spatial')}
          style={{
            background: activeTab === 'spatial' ? 'rgba(0, 242, 254, 0.15)' : 'transparent',
            border: activeTab === 'spatial' ? '1px solid var(--accent-cyan)' : '1px solid transparent',
            color: activeTab === 'spatial' ? '#fff' : 'var(--text-secondary)',
            padding: '8px 16px',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Cpu size={16} /> Spatial ViT Signals
        </button>

        <button
          onClick={() => setActiveTab('frequency')}
          style={{
            background: activeTab === 'frequency' ? 'rgba(168, 85, 247, 0.15)' : 'transparent',
            border: activeTab === 'frequency' ? '1px solid var(--accent-purple)' : '1px solid transparent',
            color: activeTab === 'frequency' ? '#fff' : 'var(--text-secondary)',
            padding: '8px 16px',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Waves size={16} /> Frequency FFT Signals
        </button>

        <button
          onClick={() => setActiveTab('temporal')}
          style={{
            background: activeTab === 'temporal' ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
            border: activeTab === 'temporal' ? '1px solid var(--accent-blue)' : '1px solid transparent',
            color: activeTab === 'temporal' ? '#fff' : 'var(--text-secondary)',
            padding: '8px 16px',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Clock size={16} /> Temporal Bi-LSTM Signals
        </button>

        <button
          onClick={() => setActiveTab('limitations')}
          style={{
            background: activeTab === 'limitations' ? 'rgba(245, 158, 11, 0.15)' : 'transparent',
            border: activeTab === 'limitations' ? '1px solid var(--status-potential)' : '1px solid transparent',
            color: activeTab === 'limitations' ? '#fff' : 'var(--text-secondary)',
            padding: '8px 16px',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Info size={16} /> Responsible AI Safeguards
        </button>
      </div>

      {/* Tab Content */}
      <div style={{ background: 'rgba(0,0,0,0.25)', padding: '24px', borderRadius: '12px', border: '1px solid var(--border-glass)' }}>
        {activeTab === 'llm' && (
          <div>
            {llmLoading ? (
              <div className="animate-pulse-slow" style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Synthesizing natural language explanation...
              </div>
            ) : llmExplanation ? (
              <div>
                <div style={{ marginBottom: '16px' }}>
                  <h4 style={{ fontSize: '0.95rem', color: 'var(--accent-cyan)', marginBottom: '6px' }}>Executive Summary</h4>
                  <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: '#f8fafc' }}>{llmExplanation.executive_summary}</p>
                </div>

                <div style={{ marginBottom: '16px' }}>
                  <h4 style={{ fontSize: '0.95rem', color: 'var(--accent-purple)', marginBottom: '6px' }}>Suspicious Frame Findings</h4>
                  <p style={{ fontSize: '0.9rem', lineHeight: 1.5, color: 'var(--text-secondary)' }}>{llmExplanation.suspicious_frame_notes}</p>
                </div>

                <div style={{ background: 'rgba(245, 158, 11, 0.1)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--status-potential)', fontWeight: 600, fontSize: '0.85rem', marginBottom: '4px' }}>
                    <AlertTriangle size={16} /> Responsible AI & Human Review Safeguard
                  </div>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                    {llmExplanation.responsible_ai_disclaimer}
                  </p>
                </div>
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>Explanation unavailable.</p>
            )}
          </div>
        )}

        {activeTab === 'spatial' && (
          <ul style={{ listStyle: 'none' }}>
            {evidence_summary.spatial_evidence.map((obs, idx) => (
              <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px', fontSize: '0.9rem' }}>
                <ShieldCheck size={18} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{obs}</span>
              </li>
            ))}
          </ul>
        )}

        {activeTab === 'frequency' && (
          <ul style={{ listStyle: 'none' }}>
            {evidence_summary.frequency_evidence.map((obs, idx) => (
              <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px', fontSize: '0.9rem' }}>
                <Waves size={18} color="var(--accent-purple)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{obs}</span>
              </li>
            ))}
          </ul>
        )}

        {activeTab === 'temporal' && (
          <ul style={{ listStyle: 'none' }}>
            {evidence_summary.temporal_evidence.map((obs, idx) => (
              <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px', fontSize: '0.9rem' }}>
                <Clock size={18} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{obs}</span>
              </li>
            ))}
          </ul>
        )}

        {activeTab === 'limitations' && (
          <ul style={{ listStyle: 'none' }}>
            {evidence_summary.limitations.map((lim, idx) => (
              <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                <Info size={18} color="var(--status-potential)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{lim}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
