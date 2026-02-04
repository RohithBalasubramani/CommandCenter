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


import React, { useState, useMemo } from 'react';
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
  ReferenceLine,
  ReferenceArea
} from 'recharts';
import { format, parseISO } from 'date-fns';
import { Maximize2, MoreHorizontal, Calendar, AlertTriangle, TrendingUp, Clock } from 'lucide-react';

interface TrendWidgetProps {
  config: WidgetConfig;
  data: DataPoint[];
  isLoading?: boolean;
  timeRange?: TimeRange;
  onTimeRangeChange?: (range: TimeRange) => void;
  selectedDate?: Date;
  onDateChange?: (date: Date) => void;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-3 border border-neutral-200 shadow-lg rounded-lg text-sm z-50">
        <div className="mb-2 font-mono text-xs text-neutral-500 pb-2 border-b border-neutral-100">
          {format(parseISO(label), 'MMM dd, HH:mm')}
        </div>
        <div className="space-y-1">
          {payload.map((entry: any, idx: number) => (
            <div key={idx} className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-neutral-600 font-medium">{entry.name}</span>
              </div>
              <span className="font-mono font-bold text-neutral-900">
                {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
                <span className="text-neutral-400 text-xs ml-1">{entry.unit}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

const TrendWidget: React.FC<TrendWidgetProps> = ({ 
  config, 
  data, 
  isLoading,
  timeRange = '1D',
  onTimeRangeChange,
  selectedDate = new Date(),
  onDateChange
}) => {
  const [hoverData, setHoverData] = useState<DataPoint | null>(null);

  // Calculate totals for header summary
  const totals = useMemo(() => {
    if (!data.length || isLoading) return {};
    const lastPoint = data[data.length - 1];
    const result: Record<string, number> = {};
    
    config.series.forEach(s => {
      const key = config.mode === 'cumulative' 
        ? `${s.id}_cumulative` 
        : `${s.id}_raw`;
      
      if (config.mode === 'cumulative') {
         result[s.id] = Number(lastPoint[key]);
      } else {
         result[s.id] = data.reduce((acc, curr) => acc + (Number(curr[key]) || 0), 0);
      }
    });
    return result;
  }, [data, config, isLoading]);

  const primarySeries = config.series[0];
  const primaryTotal = totals[primarySeries.id] || 0;
  
  // Determine if we are in a warning state
  const isWarning = !isLoading && config.targetLine && primaryTotal > config.targetLine;

  return (
    <div className="w-full h-full bg-white rounded-xl border border-neutral-100 shadow-sm hover:shadow-md transition-shadow duration-300 flex flex-col overflow-hidden relative group">
      {/* 2.1 Header Row */}
      <div className="shrink-0 border-b border-neutral-100 px-3 py-2 flex flex-wrap items-center justify-between bg-white sticky top-0 z-20 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className={`p-1.5 rounded-lg border border-neutral-100 transition-colors shrink-0 ${isLoading ? 'bg-neutral-50 text-neutral-300' : 'bg-neutral-50 text-neutral-400'}`}>
            <TrendingUp size={16} />
          </div>
          <div className={isLoading ? "opacity-80" : ""}>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold text-neutral-900 leading-tight">
                {config.title}
              </h2>
              {isWarning && (
                <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-amber-50 text-amber-700 border border-amber-100 flex items-center gap-1">
                  <AlertTriangle size={10} /> Over Target
                </span>
              )}
            </div>
            <p className="text-xs text-neutral-400 mt-0.5 font-medium">
              {config.subtitle}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0 flex-wrap">
           {/* Controls Section */}
           <div className="flex items-center gap-1.5">
             {/* Time Range Selector */}
             <div className="flex bg-neutral-100 p-0.5 rounded-md">
               {(['1D', '1W', '1M'] as TimeRange[]).map((r) => (
                 <button
                   key={r}
                   disabled={isLoading}
                   onClick={() => onTimeRangeChange?.(r)}
                   className={`px-2 py-0.5 text-[10px] font-bold rounded transition-all ${
                     timeRange === r
                       ? 'bg-white text-neutral-900 shadow-sm'
                       : 'text-neutral-500 hover:text-neutral-700 hover:bg-neutral-200/50'
                   } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                 >
                   {r}
                 </button>
               ))}
             </div>

             {/* Date Picker - hidden at small sizes */}
             <div className="relative group hidden lg:block">
               <div className="absolute inset-y-0 left-0 pl-2 flex items-center pointer-events-none">
                 <Calendar size={12} className="text-neutral-400" />
               </div>
               <input
                 type="date"
                 disabled={isLoading}
                 value={selectedDate ? format(selectedDate, 'yyyy-MM-dd') : ''}
                 onChange={(e) => e.target.valueAsDate && onDateChange?.(e.target.valueAsDate)}
                 className={`pl-7 pr-2 py-1 bg-white border border-neutral-200 rounded-md text-[10px] font-medium text-neutral-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 w-[120px] transition-shadow ${isLoading ? 'opacity-50 cursor-not-allowed bg-neutral-50' : 'hover:border-neutral-300'}`}
               />
             </div>

             <button disabled={isLoading} className={`p-1 rounded-md transition-colors ${isLoading ? 'text-neutral-200 cursor-not-allowed' : 'text-neutral-400 hover:text-neutral-900 hover:bg-neutral-50'}`}>
               <MoreHorizontal size={14} />
             </button>
           </div>

           {/* Mini Summary - only on large screens */}
           <div className="text-right hidden xl:flex items-center gap-2 ml-2">
              {!isLoading && (
                <div className={`text-sm font-bold tracking-tight leading-none ${isWarning ? 'text-amber-600' : 'text-neutral-900'}`}>
                  {new Intl.NumberFormat('en-US').format(Math.round(primaryTotal))}
                  <span className="text-[10px] text-neutral-400 font-medium ml-0.5">{primarySeries.unit}</span>
                </div>
              )}
           </div>
        </div>
      </div>

      {/* Body: Chart */}
      <div className="p-4 flex-1 min-h-0 w-full relative">
        {isLoading ? (
          <div className="w-full h-full flex flex-col gap-4 animate-pulse">
            {/* Fake Grid lines & Chart Area */}
            <div className="flex-1 flex flex-col justify-between py-2">
                {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="w-full h-[1px] bg-neutral-100" />
                ))}
                
                {/* Simulated Chart Shape */}
                <div className="absolute inset-0 flex items-end px-4 pb-8 opacity-50">
                    <div className="w-full h-2/3 bg-neutral-50 rounded-t-xl" />
                </div>
            </div>
            
            {/* Axis labels skeleton */}
            <div className="flex justify-between mt-2 px-2">
               {[1, 2, 3, 4, 5].map((i) => (
                   <div key={i} className="h-2 w-8 bg-neutral-100 rounded" />
               ))}
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
              onMouseMove={(e) => {
                if (e.activePayload) {
                  setHoverData(e.activePayload[0].payload);
                }
              }}
              onMouseLeave={() => setHoverData(null)}
            >
              <defs>
                <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.1}/>
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid 
                vertical={false} 
                stroke="#f5f5f5" 
                strokeDasharray="3 3" 
              />
              <XAxis 
                dataKey="x" 
                tickFormatter={(str) => format(parseISO(str), timeRange === '1D' ? 'HH:mm' : 'MMM dd')}
                tick={{ fontSize: 10, fill: '#a3a3a3', fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                minTickGap={40}
                dy={10}
              />
              <YAxis 
                tick={{ fontSize: 10, fill: '#a3a3a3', fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                dx={-10}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#a3a3a3', strokeWidth: 1, strokeDasharray: '3 3' }} />

              {/* Threshold Bands */}
              {config.thresholdBands?.map((band, idx) => (
                <ReferenceArea 
                  key={idx}
                  y1={band.min} 
                  y2={band.max} 
                  fill={
                    band.severity === 'critical' ? '#fee2e2' : 
                    band.severity === 'warning' ? '#fef3c7' : '#f0fdf4'
                  } 
                  fillOpacity={0.4} 
                />
              ))}

              {/* Target Line */}
              {config.targetLine && (
                 <ReferenceLine 
                   y={config.targetLine} 
                   stroke="#f59e0b" 
                   strokeDasharray="4 4" 
                   strokeWidth={1.5}
                   label={{ 
                      position: 'insideTopRight', 
                      value: 'BUDGET LIMIT', 
                      fill: '#d97706', 
                      fontSize: 10, 
                      fontWeight: 700 
                   }}
                 />
              )}

              {/* Series Rendering */}
              {config.series.map((series, idx) => {
                const dataKey = config.mode === 'cumulative' ? `${series.id}_cumulative` : `${series.id}_raw`;
                const isStep = config.variant === 'V6';
                
                if (config.variant === 'V3' || config.variant === 'V2') {
                  // Area Chart types
                  return (
                    <Area
                      key={series.id}
                      type={isStep ? "stepAfter" : "monotone"}
                      dataKey={dataKey}
                      name={series.label}
                      unit={series.unit}
                      stackId={config.stacked ? "1" : undefined}
                      stroke={series.color || '#2563eb'}
                      fill={series.color || 'url(#colorGradient)'}
                      fillOpacity={config.stacked ? 1 : 0.3}
                      strokeWidth={2}
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                  );
                } else {
                   // Line Chart types (V1, V4, V5, V6)
                   return (
                     <Line
                      key={series.id}
                      type={isStep ? "stepAfter" : "monotone"}
                      dataKey={dataKey}
                      name={series.label}
                      unit={series.unit}
                      stroke={series.color || '#171717'}
                      strokeWidth={series.isBaseline ? 2 : 2.5}
                      strokeDasharray={series.isBaseline ? "4 4" : undefined}
                      dot={false}
                      activeDot={{ r: 6, strokeWidth: 2, stroke: '#fff' }}
                     />
                   );
                }
              })}
            </ComposedChart>
          </ResponsiveContainer>
        )}
        
        {/* Hover Info Overlay (Industrial Style) */}
        {!isLoading && hoverData && (
          <div className="absolute top-4 right-4 pointer-events-none bg-white/90 backdrop-blur-sm p-3 rounded-lg border border-neutral-100 shadow-sm z-10 hidden md:block">
            <div className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 mb-1">
              Cursor Value
            </div>
            {config.series.map(s => {
               const key = config.mode === 'cumulative' ? `${s.id}_cumulative` : `${s.id}_raw`;
               return (
                 <div key={s.id} className="flex items-center justify-between gap-4">
                   <span className="text-xs text-neutral-500 font-medium">{s.label}</span>
                   <span className="text-sm font-mono font-bold text-neutral-900">
                     {Number(hoverData[key]).toFixed(1)} <span className="text-[10px] text-neutral-400">{s.unit}</span>
                   </span>
                 </div>
               )
            })}
          </div>
        )}
      </div>

      {/* Footer Legend */}
      <div className="bg-neutral-50 border-t border-neutral-100 px-3 py-2 flex flex-wrap gap-1.5 items-center shrink-0">
        {isLoading ? (
          <div className="flex gap-2 animate-pulse w-full items-center">
            <div className="h-7 w-20 bg-neutral-200 rounded-full" />
            <div className="h-7 w-24 bg-neutral-200 rounded-full" />
            <div className="flex-1" />
            <div className="h-3 w-32 bg-neutral-200 rounded" />
          </div>
        ) : (
          <>
            {config.series.map((s) => (
              <button
                key={s.id}
                className={`
                  flex items-center gap-1.5 px-2 py-1 rounded-full border text-[10px] font-semibold transition-all
                  ${s.isBaseline ? 'border-neutral-200 bg-white text-neutral-400' : 'border-neutral-200 bg-white text-neutral-700 hover:border-blue-300 hover:text-blue-700 hover:shadow-sm'}
                `}
              >
                <div className={`w-2 h-2 rounded-full ${s.isBaseline ? 'bg-neutral-300' : ''}`} style={{ backgroundColor: !s.isBaseline ? s.color || '#2563eb' : undefined }} />
                {s.label}
              </button>
            ))}
            
            <div className="flex-1"></div>
            
            {/* Footer Metadata */}
            <div className="flex items-center gap-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400">
              <span>{config.mode} Mode</span>
              {config.stacked && <span>Stacked</span>}
              <span>{data.length} pts</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};


export default function ScenarioComponent({ data }) {
  if (!data) return null;
  const selectedDate = data.selectedDate ? new Date(data.selectedDate) : undefined;
  return (
    <TrendWidget
      config={data.config}
      data={data.data}
      timeRange={data.timeRange}
      selectedDate={selectedDate}
    />
  );
}
