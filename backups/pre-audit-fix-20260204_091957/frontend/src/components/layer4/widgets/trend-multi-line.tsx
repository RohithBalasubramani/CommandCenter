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
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend
} from 'recharts';

import { MoreHorizontal, Download, Activity, AlertTriangle } from 'lucide-react';

interface TrendWidgetProps {
  preset: DemoDataPreset;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-neutral-200 shadow-lg p-3 rounded-lg z-50">
        <p className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 mb-2">
          {new Date(label).toLocaleTimeString()}
        </p>
        <div className="space-y-1">
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between gap-4 text-xs font-mono">
              <span className="flex items-center gap-2 text-neutral-600">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }}></span>
                {entry.name}:
              </span>
              <span className="font-bold text-neutral-900">
                {entry.value.toFixed(1)} <span className="text-neutral-400 font-normal">{entry.unit}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

const TrendWidget: React.FC<TrendWidgetProps> = ({ preset }) => {
  // Determine Y-Axis configurations
  const hasRightAxis = preset.series.some(s => s.yAxis === 'right');
  const leftUnit = preset.series.find(s => s.yAxis === 'left')?.unit || '';
  const rightUnit = preset.series.find(s => s.yAxis === 'right')?.unit || '';

  return (
    <div className="flex flex-col h-full bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden hover:shadow-md transition-all duration-300">
      {/* Header */}
      <div className="h-16 px-6 border-b border-gray-100 flex items-center justify-between bg-white shrink-0">
        <div className="flex items-center gap-3">
            <div className="bg-neutral-100 p-2 rounded-lg">
                <Activity className="w-4 h-4 text-neutral-600" />
            </div>
            <div>
                <h3 className="text-sm font-bold text-neutral-900 tracking-tight">{preset.name}</h3>
                <div className="flex items-center gap-2">
                    <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{preset.description}</span>
                </div>
            </div>
        </div>
        <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-green-50 text-green-700 border border-green-100">
                Live
            </span>
            <button className="p-2 text-neutral-400 hover:text-neutral-900 hover:bg-neutral-50 rounded-lg transition-colors">
                <Download className="w-4 h-4" />
            </button>
            <button className="p-2 text-neutral-400 hover:text-neutral-900 hover:bg-neutral-50 rounded-lg transition-colors">
                <MoreHorizontal className="w-4 h-4" />
            </button>
        </div>
      </div>

      {/* Chart Area */}
      <div className="flex-1 min-h-0 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={preset.data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f5f5f5" vertical={false} />
            <XAxis 
              dataKey="timestamp" 
              tick={{ fontSize: 10, fill: '#a3a3a3' }} 
              tickLine={false}
              axisLine={{ stroke: '#e5e5e5' }}
              tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              minTickGap={40}
            />
            
            {/* Left Axis */}
            <YAxis 
              yAxisId="left"
              tick={{ fontSize: 10, fill: '#a3a3a3' }} 
              tickLine={false}
              axisLine={false}
              width={40}
              label={{ value: leftUnit, angle: -90, position: 'insideLeft', style: { fill: '#d4d4d4', fontSize: 10, textAnchor: 'middle' } }}
            />

            {/* Right Axis (Conditional) */}
            {hasRightAxis && (
              <YAxis 
                yAxisId="right"
                orientation="right"
                tick={{ fontSize: 10, fill: '#a3a3a3' }} 
                tickLine={false}
                axisLine={false}
                width={40}
                label={{ value: rightUnit, angle: 90, position: 'insideRight', style: { fill: '#d4d4d4', fontSize: 10, textAnchor: 'middle' } }}
              />
            )}

            <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#2563eb', strokeWidth: 1, strokeDasharray: '4 4' }} />
            
            <Legend 
                wrapperStyle={{ paddingTop: '10px', fontSize: '11px', fontFamily: 'Inter' }}
                iconType="circle" 
                iconSize={8}
            />

            {/* Threshold Lines */}
            {preset.thresholds?.map((t, i) => (
                <ReferenceLine 
                    key={i} 
                    y={t.value} 
                    yAxisId="left" 
                    stroke={t.color} 
                    strokeDasharray="3 3"
                    label={{ 
                        position: 'insideTopRight', 
                        value: t.label, 
                        fill: t.color, 
                        fontSize: 9, 
                        fontWeight: 'bold' 
                    }} 
                />
            ))}

            {/* Series Rendering */}
            {preset.series.map((s) => (
              <Line
                key={s.id}
                yAxisId={s.yAxis}
                type="monotone"
                dataKey={s.id}
                stroke={s.colorToken}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: s.colorToken }}
                name={s.label}
                strokeDasharray={s.lineStyle === 'dashed' ? '5 5' : s.lineStyle === 'dotted' ? '2 2' : undefined}
                unit={s.unit}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer / Controls */}
      <div className="h-12 border-t border-gray-100 flex items-center justify-between px-6 bg-neutral-50/50 text-xs shrink-0">
         <div className="flex items-center gap-4 text-neutral-400">
            <span className="font-medium text-neutral-600">Range: {preset.timeRange}</span>
            <span className="w-px h-3 bg-neutral-300"></span>
            <span>Step: {preset.granularity}</span>
         </div>
         <div className="flex items-center gap-2">
            {preset.thresholds && (
                <div className="flex items-center gap-1 text-amber-600 bg-amber-50 px-2 py-0.5 rounded border border-amber-100">
                    <AlertTriangle className="w-3 h-3" />
                    <span className="font-bold text-[10px] uppercase">Alerts Active</span>
                </div>
            )}
         </div>
      </div>
    </div>
  );
};



export default function ScenarioComponent({ data }) {
  return <TrendWidget preset={data} />;
}
