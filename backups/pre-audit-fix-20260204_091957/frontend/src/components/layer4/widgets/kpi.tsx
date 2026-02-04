// @ts-nocheck

// Typography Components - shared design system primitives
const SectionLabel = ({ children, className = '' }) => (
  React.createElement('span', {
    className: `text-[10px] uppercase font-bold tracking-widest text-neutral-400 ${className}`
  }, children)
);

const ValueDisplay = ({ children, className = '' }) => (
  React.createElement('span', {
    className: `text-3xl font-bold tracking-tight text-neutral-900 ${className}`
  }, children)
);

const MonoText = ({ children, className = '' }) => (
  React.createElement('span', {
    className: `font-mono text-xs text-neutral-700 ${className}`
  }, children)
);

const Label = ({ children, className = '' }) => (
  React.createElement('span', {
    className: `text-[10px] uppercase font-bold tracking-widest text-neutral-400 select-none ${className}`
  }, children)
);

const SectionHeader = ({ children, className = '' }) => (
  React.createElement('h3', {
    className: `text-xs uppercase font-bold tracking-widest text-neutral-500 mb-4 ${className}`
  }, children)
);

const HeroValue = ({ children, className = '' }) => (
  React.createElement('div', {
    className: `text-[5rem] font-bold leading-none tracking-tighter text-neutral-900 ${className}`
  }, children)
);

const Value = ({ children, className = '' }) => (
  React.createElement('div', {
    className: `text-3xl font-bold tracking-tight text-neutral-900 ${className}`
  }, children)
);

const Mono = ({ children, className = '' }) => (
  React.createElement('span', {
    className: `font-mono text-xs text-neutral-700 ${className}`
  }, children)
);

// Design Tokens - color and typography constants
const COLORS = {
  neutral: {
    50: '#fafafa', 100: '#f5f5f5', 200: '#e5e5e5', 300: '#d4d4d4',
    400: '#a3a3a3', 500: '#737373', 600: '#525252', 700: '#404040',
    800: '#262626', 900: '#171717',
  },
  primary: '#2563eb',
  secondary: '#a3a3a3',
  grid: '#f5f5f5',
  success: '#16a34a',
  warning: '#d97706',
  critical: '#ef4444',
  danger: '#ef4444',
  accent: '#2563eb',
  sources: {
    'Grid': '#2563eb',
    'Solar': '#16a34a',
    'DG': '#d97706',
    'Wind': '#06b6d4',
  },
  loads: {
    'HVAC': '#525252',
    'Lighting': '#737373',
    'Machinery': '#171717',
    'Loss': '#ef4444',
  }
};

const TYPO = {
  label: "text-[10px] uppercase font-bold tracking-widest text-neutral-400",
  sectionHeader: "text-xs uppercase font-bold tracking-widest text-neutral-500",
  h1: "text-2xl font-bold tracking-tight text-neutral-900",
  heroValue: "text-[3rem] font-bold leading-none tracking-tighter text-neutral-900",
  cardValue: "text-3xl font-bold tracking-tight text-neutral-900",
  mono: "font-mono text-xs text-neutral-700",
};


import React from 'react';


import { AlertTriangle, Info, Zap, Activity } from 'lucide-react';

interface KpiRendererProps {
  spec: WidgetSpec;
}

const KpiRenderer: React.FC<KpiRendererProps> = ({ spec }) => {
  const { layout, visual, demoData, variant } = spec;
  const isDark = visual.theme === 'dark';

  const baseClasses = `
    ${layout.padding} 
    ${layout.radius}
    ${visual.background}
    ${visual.border}
    border shadow-sm
    flex flex-col justify-between
    transition-all duration-200
    hover:shadow-md
    h-full
  `;

  // Helper to determine text colors based on theme and state
  const labelColor = isDark ? 'text-neutral-400' : 'text-neutral-500';
  const valueColor = isDark ? 'text-white' : 'text-neutral-900';
  const unitColor = isDark ? 'text-neutral-500' : 'text-neutral-400';

  const renderIcon = () => {
    if (demoData.state === 'warning' || demoData.state === 'critical') {
      return <AlertTriangle className={`w-5 h-5 ${demoData.state === 'critical' ? 'text-red-500' : 'text-amber-500'}`} />;
    }
    if (variant === 'KPI_LIVE') return <Zap className="w-5 h-5 text-blue-500" />;
    return null;
  };

  return (
    <div className={baseClasses}>
      {/* Header */}
      <div className="flex justify-between items-start mb-1">
        <SectionLabel className={labelColor}>{demoData.label}</SectionLabel>
        {renderIcon()}
      </div>

      {/* Body */}
      <div className="flex-1 flex flex-col justify-end min-h-0">
        {/* Value Rendering */}
        {variant !== 'KPI_STATUS' && (
          <div className="flex items-baseline gap-1.5">
            <span className={`text-xl font-bold tracking-tight ${
              demoData.state === 'critical' ? 'text-red-600' :
              demoData.state === 'warning' ? 'text-amber-600' :
              valueColor
            }`}>
              {demoData.value}
            </span>
            {demoData.unit && <span className={`text-xs font-medium ${unitColor}`}>{demoData.unit}</span>}
          </div>
        )}

        {/* Status Rendering */}
        {variant === 'KPI_STATUS' && (
          <div className="flex items-center gap-1.5 mt-1">
             <span className={`w-2 h-2 rounded-full ${demoData.statusColor || 'bg-gray-400'}`}></span>
             <span className={`text-sm font-semibold ${valueColor}`}>{demoData.status}</span>
          </div>
        )}

        {/* Lifecycle Bar */}
        {variant === 'KPI_LIFECYCLE' && demoData.max && (
          <div className="mt-1.5 w-full bg-neutral-100 dark:bg-neutral-800 rounded-full h-1 overflow-hidden">
            <div
              className={`h-full rounded-full ${isDark ? 'bg-green-500' : 'bg-blue-600'}`}
              style={{ width: `${(Number(demoData.value) / demoData.max) * 100}%` }}
            ></div>
          </div>
        )}

        {/* Accumulated Period */}
        {variant === 'KPI_ACCUMULATED' && demoData.period && (
           <div className={`mt-1 text-[9px] uppercase tracking-wider font-medium ${isDark ? 'text-neutral-600' : 'text-neutral-400'}`}>
             {demoData.period}
           </div>
        )}
      </div>

      {/* Spec Metadata Overlay (Hover only) */}
      <div className="hidden group-hover:block absolute top-2 right-2">
        <div className="bg-black/80 text-white text-[9px] px-2 py-1 rounded font-mono">
           {spec.variant}
        </div>
      </div>
    </div>
  );
};


export default function ScenarioComponent({ data }) {
  if (!data) {
    return (
      <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#991b1b' }}>
        <strong>Missing spec data</strong>
      </div>
    );
  }
  return <KpiRenderer spec={data} />;
}
