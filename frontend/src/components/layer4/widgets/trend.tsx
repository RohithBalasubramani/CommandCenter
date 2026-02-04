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
  LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, 
  ResponsiveContainer, ReferenceLine, CartesianGrid, Legend 
} from 'recharts';
import { Activity, Maximize2, MoreHorizontal } from 'lucide-react';

interface TrendRendererProps {
  spec: WidgetSpec;
}

const TrendRenderer: React.FC<TrendRendererProps> = ({ spec }) => {
  const { layout, visual, demoData, variant, representation } = spec;
  const isDark = visual.theme === 'dark';
  
  // Base Container Styles
  const containerClasses = `
    ${layout.padding} 
    ${layout.radius} 
    ${visual.background} 
    ${visual.border} 
    border shadow-sm 
    flex flex-col 
    transition-all duration-200 
    hover:shadow-md
    relative
    overflow-hidden
  `;

  // Fill parent container — the grid cell controls actual height
  const heightClass = 'h-full';

  // Colors
  const mainColor = visual.colors?.[0] || '#2563eb';
  const gridColor = isDark ? '#333' : '#f3f4f6';
  const textColor = isDark ? '#737373' : '#a3a3a3';
  const tooltipBg = isDark ? '#171717' : '#ffffff';
  const tooltipBorder = isDark ? '#262626' : '#e5e5e5';

  // --- Render Chart Content ---
  const renderChart = () => {
    if (representation === 'Heatmap') {
      return (
        <div className="w-full h-full flex flex-col">
           <div className="flex-1 grid grid-cols-7 gap-1 mt-2">
              {demoData.buckets?.map((bucket, i) => {
                 // Intensity mapping
                 let bg = isDark ? 'bg-neutral-800' : 'bg-gray-100';
                 if (bucket.intensity > 0.25) bg = 'bg-blue-200';
                 if (bucket.intensity > 0.50) bg = 'bg-blue-400';
                 if (bucket.intensity > 0.75) bg = 'bg-blue-600';
                 
                 return (
                   <div key={i} className={`rounded-sm relative group ${bg} transition-colors hover:ring-2 ring-blue-500/50`}>
                      {/* Tooltip hint */}
                      <div className="opacity-0 group-hover:opacity-100 absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-black text-white text-[9px] px-1 py-0.5 rounded pointer-events-none whitespace-nowrap z-10">
                        {Math.round(bucket.intensity * 100)}%
                      </div>
                   </div>
                 );
              })}
           </div>
           <div className="flex justify-between text-[9px] text-neutral-400 mt-2 font-mono uppercase">
              <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span><span>Sat</span><span>Sun</span>
           </div>
        </div>
      );
    }

    const CommonAxis = (
      <>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={gridColor} />
        <XAxis 
          dataKey="time" 
          tick={{ fontSize: 10, fill: textColor }} 
          tickLine={false} 
          axisLine={false}
          dy={10}
        />
        <YAxis 
          tick={{ fontSize: 10, fill: textColor }} 
          tickLine={false} 
          axisLine={false}
          width={30}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: tooltipBg, 
            borderColor: tooltipBorder, 
            fontSize: '11px',
            borderRadius: '6px'
          }}
          itemStyle={{ color: isDark ? '#fff' : '#000' }}
        />
      </>
    );

    if (representation === 'Area') {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={demoData.timeSeries}>
            <defs>
              <linearGradient id={`grad${spec.variant}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={mainColor} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={mainColor} stopOpacity={0}/>
              </linearGradient>
            </defs>
            {CommonAxis}
            <Area type="monotone" dataKey="value" stroke={mainColor} fill={`url(#grad${spec.variant})`} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      );
    }

    if (representation === 'RGB Phase Line') {
      return (
         <ResponsiveContainer width="100%" height="100%">
           <LineChart data={demoData.timeSeries}>
             {CommonAxis}
             <Line type="monotone" dataKey="r" stroke={visual.colors?.[0]} strokeWidth={1.5} dot={false} />
             <Line type="monotone" dataKey="y" stroke={visual.colors?.[1]} strokeWidth={1.5} dot={false} />
             <Line type="monotone" dataKey="b" stroke={visual.colors?.[2]} strokeWidth={1.5} dot={false} />
             <Legend iconType="circle" wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }}/>
           </LineChart>
         </ResponsiveContainer>
      );
    }

    // Default to Line / Step Line
    const stepType = representation === 'Step Line' ? 'stepAfter' : 'monotone';
    
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={demoData.timeSeries}>
          {CommonAxis}
          {variant === 'TREND_ALERT_CONTEXT' && (
             <ReferenceLine y={85} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Crit', fill: '#ef4444', fontSize: 9 }} />
          )}
          <Line 
            type={stepType} 
            dataKey="value" 
            stroke={mainColor} 
            strokeWidth={2} 
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className={`${containerClasses} ${heightClass}`}>
      {/* Header Zone */}
      <div className="flex justify-between items-start mb-4 shrink-0">
        <div>
           <SectionLabel className={isDark ? 'text-neutral-500' : 'text-neutral-400'}>{demoData.label}</SectionLabel>
           <div className={`text-[10px] font-mono mt-1 ${isDark ? 'text-neutral-600' : 'text-neutral-400'}`}>
              {demoData.timeRange}
           </div>
        </div>
        <div className="flex gap-2">
           <button className={`p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors ${isDark ? 'text-neutral-500' : 'text-neutral-400'}`}>
              <Maximize2 className="w-3.5 h-3.5" />
           </button>
           <button className={`p-1 rounded hover:bg-black/5 dark:hover:bg-white/10 transition-colors ${isDark ? 'text-neutral-500' : 'text-neutral-400'}`}>
              <MoreHorizontal className="w-3.5 h-3.5" />
           </button>
        </div>
      </div>

      {/* Chart Zone */}
      <div className="flex-1 min-h-0 relative">
         {renderChart()}
         
         {/* Live Indicator Overlay */}
         {variant === 'TREND_LIVE' && (
           <div className="absolute top-0 right-0 flex items-center gap-1.5 bg-red-500/10 px-2 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
              <span className="text-[9px] font-bold text-red-500 tracking-wide">LIVE</span>
           </div>
         )}
      </div>

      {/* Footer Zone (if step line or pattern) */}
      {variant === 'TREND_STANDARD' && (
         <div className="mt-2 pt-2 border-t border-gray-100 dark:border-neutral-800 flex justify-between items-center text-[10px] text-neutral-400">
            <span>Agg: 1h Avg</span>
            <span>Export CSV</span>
         </div>
      )}
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
  return <TrendRenderer spec={data} />;
}
