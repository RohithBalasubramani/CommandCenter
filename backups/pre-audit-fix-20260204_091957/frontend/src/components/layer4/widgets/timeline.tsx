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

// Card component for Timeline
const Card = ({ children, className = '', onClick, variant = 'light' }) => {
  const baseClasses = "rounded-xl p-4 transition-all duration-200 flex flex-col h-full";
  const variants = {
    light: "bg-white border border-gray-100 shadow-sm hover:shadow-md",
    dark: "bg-neutral-900 border border-neutral-800 shadow-md text-white"
  };
  const interactiveClasses = onClick ? "cursor-pointer active:scale-[0.99]" : "";
  return React.createElement('div', {
    className: baseClasses + ' ' + variants[variant] + ' ' + interactiveClasses + ' ' + className,
    onClick: onClick
  }, children);
};


import React, { useState, useEffect } from 'react';



import { AlertTriangle, CheckCircle, Info, XCircle, Clock, Pause, Zap, Flag, FileText, User, Search, BarChart3, ZoomIn, ZoomOut, Filter, Layers, ChevronRight, ChevronLeft, Activity, RotateCcw } from 'lucide-react';

// --- HELPERS ---

const getTimestamp = (timeStr: string): number => {
  // ISO Date
  if (timeStr.includes('T') || timeStr.includes('-')) {
    return new Date(timeStr).getTime();
  }
  // Simple HH:mm (assume today)
  const [h, m] = timeStr.split(':').map(Number);
  const d = new Date();
  d.setHours(h, m, 0, 0);
  return d.getTime();
};

// Updated to accept numeric start/end for dynamic zooming
const getPosition = (time: string, start: number, end: number) => {
  const t = getTimestamp(time);
  const total = end - start;
  if (total === 0) return 0;
  return ((t - start) / total) * 100;
};

// Updated to accept numeric start/end
const getWidth = (startTime: string, endTime: string | undefined, rangeStart: number, rangeEnd: number) => {
  if (!endTime) return 0; // Point event
  const start = getPosition(startTime, rangeStart, rangeEnd);
  const end = getPosition(endTime, rangeStart, rangeEnd);
  return end - start;
};

const formatTimeLabel = (timeVal: string | number) => {
  const date = typeof timeVal === 'number' ? new Date(timeVal) : new Date(timeVal);
  if (isNaN(date.getTime())) return String(timeVal);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const StatusColorMap: Record<TimelineStatus, string> = {
  normal: 'bg-green-500',
  success: 'bg-green-500',
  warning: 'bg-amber-400',
  critical: 'bg-red-500',
  idle: 'bg-neutral-300',
  neutral: 'bg-blue-400',
  maintenance: 'bg-orange-400',
  offline: 'bg-slate-500',
  unknown: 'bg-gray-400'
};

const getIcon = (iconName?: string) => {
  switch (iconName) {
    case 'alert': return AlertTriangle;
    case 'check': return CheckCircle;
    case 'error': return XCircle;
    case 'pause': return Pause;
    case 'zap': return Zap;
    case 'flag': return Flag;
    case 'file': return FileText;
    default: return Info;
  }
};

// --- SUB-COMPONENTS ---

const TimeAxis: React.FC<{ start: number; end: number }> = ({ start, end }) => {
  const duration = end - start;
  
  // Determine tick interval based on duration
  let tickCount = 6;
  const interval = duration / (tickCount - 1);
  const ticks = [];
  
  for (let i = 0; i < tickCount; i++) {
    const timeVal = start + (interval * i);
    const date = new Date(timeVal);
    // Format: HH:mm:ss if zoomed in, HH:mm otherwise
    const showSeconds = duration < 3600000; // < 1 hour
    const label = date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: showSeconds ? '2-digit' : undefined 
    });
    ticks.push({ label, percent: (i / (tickCount - 1)) * 100 });
  }

  return (
    <div className="relative w-full h-6 mt-4 border-t border-gray-100 transition-all duration-300">
      {ticks.map((tick, i) => (
        <div 
          key={i} 
          className="absolute transform -translate-x-1/2 flex flex-col items-center"
          style={{ left: `${tick.percent}%` }}
        >
          <div className="w-px h-1 bg-gray-300 mb-1"></div>
          <span className="text-[9px] text-neutral-400 font-mono whitespace-nowrap">{tick.label}</span>
        </div>
      ))}
    </div>
  );
};

const LinearEventMarker: React.FC<{ event: TimelineEvent, range: { start: number, end: number } }> = ({ event, range }) => {
  const left = getPosition(event.startTime, range.start, range.end);
  // Hide if out of view
  if (left < -5 || left > 105) return null;

  const Icon = getIcon(event.icon);
  
  return (
    <div 
      className="absolute top-1/2 -translate-y-1/2 transform -translate-x-1/2 group cursor-pointer z-10 transition-all duration-300"
      style={{ left: `${left}%` }}
    >
      <div className={`p-1 rounded-full border-2 border-white shadow-sm transition-transform group-hover:scale-125 ${StatusColorMap[event.status]} text-white`}>
        <Icon className="w-3 h-3" />
      </div>
      
      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
        <div className="bg-neutral-900 text-white text-xs rounded px-2 py-1 shadow-lg flex flex-col items-center">
          <span className="font-bold">{formatTimeLabel(event.startTime)}</span>
          <span>{event.label}</span>
        </div>
        <div className="w-2 h-2 bg-neutral-900 transform rotate-45 mx-auto -mt-1"></div>
      </div>
    </div>
  );
};

const StatusBlock: React.FC<{ event: TimelineEvent, range: { start: number, end: number } }> = ({ event, range }) => {
  const left = getPosition(event.startTime, range.start, range.end);
  const width = getWidth(event.startTime, event.endTime, range.start, range.end);
  
  // Optimization: Don't render if completely out of view
  if (left + width < 0 || left > 100) return null;

  return (
    <div 
      className={`absolute h-full top-0 border-r border-white/20 hover:opacity-90 cursor-pointer flex items-center justify-center overflow-hidden group ${StatusColorMap[event.status]} transition-all duration-300`}
      style={{ left: `${left}%`, width: `${width}%` }}
      title={`${event.label} (${formatTimeLabel(event.startTime)} - ${event.endTime ? formatTimeLabel(event.endTime) : ''})`}
    >
      {width > 10 && (
         <span className="text-[10px] font-bold text-white/90 truncate px-1 uppercase tracking-wider group-hover:scale-105 transition-transform">
           {event.label}
         </span>
      )}
    </div>
  );
};

const AnnotationPin: React.FC<{ annotation: TimelineAnnotation, range: { start: number, end: number } }> = ({ annotation, range }) => {
  const left = getPosition(annotation.time, range.start, range.end);
  if (left < -5 || left > 105) return null;

  const typeColors = {
    operator_note: 'bg-blue-500 text-white border-blue-200',
    rca_finding: 'bg-purple-600 text-white border-purple-200',
    action: 'bg-emerald-500 text-white border-emerald-200'
  };

  const TypeIcon = annotation.type === 'operator_note' ? User : annotation.type === 'rca_finding' ? Search : CheckCircle;

  return (
    <div 
      className="absolute top-0 transform -translate-x-1/2 flex flex-col items-center group z-30 transition-all duration-300"
      style={{ left: `${left}%` }}
    >
       {/* Pin Head */}
       <div className={`
         relative flex items-center gap-1.5 px-2 py-1 rounded-md shadow-md border 
         ${typeColors[annotation.type]} 
         transition-transform group-hover:-translate-y-1 cursor-pointer
       `}>
          <TypeIcon className="w-3 h-3" />
          <span className="text-[10px] font-bold whitespace-nowrap max-w-[100px] truncate">{annotation.label}</span>
          
          {/* Detailed Tooltip */}
          <div className="absolute left-1/2 bottom-full -translate-x-1/2 mb-2 w-48 bg-white text-neutral-900 border border-neutral-200 shadow-xl rounded-lg p-3 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
             <div className="flex items-center gap-2 mb-1">
               <span className="text-[9px] uppercase font-bold text-neutral-400">{annotation.type.replace('_', ' ')}</span>
               <span className="ml-auto text-[9px] font-mono text-neutral-400">{formatTimeLabel(annotation.time)}</span>
             </div>
             <p className="text-xs font-medium leading-normal mb-1">{annotation.label}</p>
             {annotation.author && (
               <div className="text-[10px] text-neutral-500">By: {annotation.author}</div>
             )}
          </div>
       </div>
       {/* Pin Stick */}
       <div className="w-0.5 h-8 bg-neutral-300 group-hover:bg-neutral-400"></div>
       {/* Anchor Dot */}
       <div className="w-1.5 h-1.5 rounded-full bg-neutral-400"></div>
    </div>
  );
};

const ClusterBubble: React.FC<{ event: TimelineEvent, range: { start: number, end: number } }> = ({ event, range }) => {
  const left = getPosition(event.startTime, range.start, range.end);
  if (left < -5 || left > 105) return null;

  const count = event.clusterCount || 1;
  const isCritical = event.status === 'critical';
  
  // Calculate severity visual composition (simulated ring segments)
  const criticalPct = event.severityBreakdown ? (event.severityBreakdown.critical / count) * 100 : (isCritical ? 70 : 10);
  const warningPct = event.severityBreakdown ? (event.severityBreakdown.warning / count) * 100 : 20;
  
  // Conic gradient for the ring
  const gradient = `conic-gradient(
    #ef4444 0% ${criticalPct}%, 
    #f59e0b ${criticalPct}% ${criticalPct + warningPct}%, 
    #3b82f6 ${criticalPct + warningPct}% 100%
  )`;

  // Size grows with count, capped at 56px for very large bursts
  const size = Math.min(28 + count * 2, 56);
  
  // Critical bursts get a breathing glow
  const glowStyle = isCritical ? { boxShadow: '0 0 15px rgba(239, 68, 68, 0.4)' } : {};

  return (
    <div 
      className="absolute top-1/2 -translate-y-1/2 transform -translate-x-1/2 group cursor-pointer z-20 transition-all duration-300"
      style={{ left: `${left}%` }}
    >
      {/* Outer Ring (Severity Composition) */}
      <div 
        className={`rounded-full p-[4px] shadow-sm transition-transform duration-300 group-hover:scale-110 ${isCritical ? 'animate-pulse' : ''}`}
        style={{ width: `${size}px`, height: `${size}px`, background: gradient, ...glowStyle }}
      >
        {/* Inner Circle (Count) */}
        <div className="w-full h-full bg-white rounded-full flex items-center justify-center border border-neutral-100">
           <span className="text-[10px] font-bold text-neutral-800 tabular-nums">+{count}</span>
        </div>
      </div>

      {/* Burst Summary Callout */}
      <div className="absolute top-full left-1/2 -translate-x-1/2 mt-4 w-64 bg-white/95 backdrop-blur-md rounded-xl shadow-2xl border border-neutral-200 hidden group-hover:block z-50 overflow-hidden ring-1 ring-black/5 animate-in fade-in slide-in-from-top-2 duration-200">
         {/* Callout Header */}
         <div className="bg-neutral-50/80 px-4 py-2.5 border-b border-neutral-100 flex justify-between items-center">
            <span className="text-[10px] uppercase font-bold text-neutral-500 tracking-wider">Burst Analysis</span>
            {event.burstDetails?.correlationId && (
              <span className="flex items-center gap-1 text-[9px] text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-100 font-medium">
                <Layers className="w-3 h-3" /> {event.burstDetails.correlationId}
              </span>
            )}
         </div>
         
         {/* Callout Body */}
         <div className="p-4 space-y-4">
            <div className="flex justify-between items-end">
               <div>
                 <span className="text-2xl font-bold text-neutral-900 block leading-none">{count}</span>
                 <span className="text-[10px] text-neutral-400 font-medium uppercase mt-1 block">Total Events</span>
               </div>
               <div className="text-right">
                 <span className="text-xs font-mono text-neutral-600 bg-neutral-100 px-1.5 py-0.5 rounded">{event.burstDetails?.span || '< 1s'}</span>
                 <span className="text-[10px] text-neutral-400 font-medium uppercase mt-1 block">Duration</span>
               </div>
            </div>
            
            {/* Severity Bars */}
            <div className="space-y-2">
               {event.severityBreakdown && (
                 <>
                   {event.severityBreakdown.critical > 0 && (
                     <div className="flex items-center text-xs text-neutral-600">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-500 mr-2 shadow-[0_0_5px_rgba(239,68,68,0.5)]"></div>
                        <span className="flex-1 font-medium">Critical Errors</span>
                        <span className="font-mono font-bold text-red-600">{event.severityBreakdown.critical}</span>
                     </div>
                   )}
                   {event.severityBreakdown.warning > 0 && (
                     <div className="flex items-center text-xs text-neutral-600">
                        <div className="w-1.5 h-1.5 rounded-full bg-amber-500 mr-2"></div>
                        <span className="flex-1">Warnings</span>
                        <span className="font-mono text-neutral-700">{event.severityBreakdown.warning}</span>
                     </div>
                   )}
                   {event.severityBreakdown.info > 0 && (
                     <div className="flex items-center text-xs text-neutral-600">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mr-2"></div>
                        <span className="flex-1">System Info</span>
                        <span className="font-mono text-neutral-500">{event.severityBreakdown.info}</span>
                     </div>
                   )}
                 </>
               )}
            </div>

            <div className="pt-3 border-t border-dashed border-neutral-200">
               <span className="text-[9px] text-neutral-400 font-bold uppercase tracking-wider block mb-1.5">DOMINANT SIGNAL</span>
               <div className="text-xs font-semibold text-neutral-800 flex items-center gap-1.5 bg-neutral-50 p-2 rounded border border-neutral-100">
                  <Activity className="w-3.5 h-3.5 text-neutral-400" />
                  {event.burstDetails?.dominantType || event.label}
               </div>
            </div>
         </div>

         {/* Callout Actions */}
         <div className="bg-neutral-50 p-2 flex gap-2">
            <button className="flex-1 py-1.5 bg-white border border-neutral-200 rounded text-[10px] font-bold text-neutral-600 hover:text-blue-600 hover:border-blue-300 hover:shadow-sm flex items-center justify-center gap-1 transition-all group/btn">
               <ZoomIn className="w-3 h-3 group-hover/btn:scale-110 transition-transform" /> Inspect
            </button>
            <button className="flex-1 py-1.5 bg-white border border-neutral-200 rounded text-[10px] font-bold text-neutral-600 hover:text-blue-600 hover:border-blue-300 hover:shadow-sm flex items-center justify-center gap-1 transition-all group/btn">
               <ChevronRight className="w-3 h-3 group-hover/btn:translate-x-0.5 transition-transform" /> Details
            </button>
         </div>
      </div>
      
      {/* Connector Triangle for Tooltip */}
      <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-b-[6px] border-b-white hidden group-hover:block z-50"></div>
    </div>
  );
};

// --- MAIN RENDERER ---

const TimelineRenderer: React.FC<{ spec: TimelineSpec }> = ({ spec }) => {
  // --- VIEW STATE FOR ZOOM/PAN ---
  const [viewRange, setViewRange] = useState<{start: number, end: number}>({
    start: getTimestamp(spec.range.start),
    end: getTimestamp(spec.range.end)
  });

  // Reset view when spec changes
  useEffect(() => {
    setViewRange({
      start: getTimestamp(spec.range.start),
      end: getTimestamp(spec.range.end)
    });
  }, [spec.range]);

  const handleZoom = (factor: number) => {
    const duration = viewRange.end - viewRange.start;
    const center = viewRange.start + duration / 2;
    const newDuration = duration * factor;
    
    // Prevent zooming in too much (e.g., < 10 seconds) or out too much (e.g., > 30 days)
    if (newDuration < 10000 || newDuration > 30 * 24 * 3600 * 1000) return;

    setViewRange({
      start: center - newDuration / 2,
      end: center + newDuration / 2
    });
  };

  const handlePan = (direction: 'left' | 'right') => {
    const duration = viewRange.end - viewRange.start;
    const shift = duration * 0.2; // Shift by 20%
    const delta = direction === 'left' ? -shift : shift;
    
    setViewRange(prev => ({
      start: prev.start + delta,
      end: prev.end + delta
    }));
  };

  const handleReset = () => {
    setViewRange({
      start: getTimestamp(spec.range.start),
      end: getTimestamp(spec.range.end)
    });
  };

  return (
    <Card className="flex flex-col overflow-hidden">
      <div className="flex justify-between items-start mb-3 shrink-0">
        <div>
           <div className="flex items-center gap-2">
             <SectionLabel>{spec.title}</SectionLabel>
             {spec.variant === 'dense' && (
               <span className="px-1.5 py-0.5 rounded bg-blue-50 text-blue-600 text-[9px] font-bold border border-blue-100 uppercase tracking-wider">
                 Density Mode
               </span>
             )}
           </div>
           <p className="text-xs text-neutral-500 mt-1">{spec.description}</p>
        </div>
        
        {/* HEADER CONTROLS */}
        <div className="flex items-center gap-3">
           <div className="hidden sm:flex items-center gap-1 bg-neutral-100 p-1 rounded-lg border border-neutral-200">
             <button 
                onClick={() => handlePan('left')} 
                className="p-1 hover:bg-white hover:shadow-sm rounded transition-all text-neutral-500 hover:text-neutral-900" 
                title="Pan Left"
             >
               <ChevronLeft className="w-3.5 h-3.5" />
             </button>
             <div className="w-px h-3 bg-neutral-300 mx-0.5"></div>
             <button 
                onClick={() => handleZoom(1.2)} 
                className="p-1 hover:bg-white hover:shadow-sm rounded transition-all text-neutral-500 hover:text-neutral-900" 
                title="Zoom Out"
             >
               <ZoomOut className="w-3.5 h-3.5" />
             </button>
             <button 
                onClick={handleReset} 
                className="p-1 hover:bg-white hover:shadow-sm rounded transition-all text-neutral-500 hover:text-neutral-900" 
                title="Reset View"
             >
               <RotateCcw className="w-3.5 h-3.5" />
             </button>
             <button 
                onClick={() => handleZoom(0.8)} 
                className="p-1 hover:bg-white hover:shadow-sm rounded transition-all text-neutral-500 hover:text-neutral-900" 
                title="Zoom In"
             >
               <ZoomIn className="w-3.5 h-3.5" />
             </button>
             <div className="w-px h-3 bg-neutral-300 mx-0.5"></div>
             <button 
                onClick={() => handlePan('right')} 
                className="p-1 hover:bg-white hover:shadow-sm rounded transition-all text-neutral-500 hover:text-neutral-900" 
                title="Pan Right"
             >
               <ChevronRight className="w-3.5 h-3.5" />
             </button>
           </div>
           
           <span className="text-[10px] font-mono text-neutral-400 bg-neutral-100 px-2 py-1 rounded border border-neutral-200 whitespace-nowrap">
             {formatTimeLabel(viewRange.start)} - {formatTimeLabel(viewRange.end)}
           </span>
        </div>
      </div>

      <div className="flex-1 w-full relative min-h-0 flex flex-col justify-end pb-2 overflow-visible">
        
        {/* VARIANT: LINEAR (Events on a line) */}
        {spec.variant === 'linear' && (
          <div className="relative w-full py-4">
            <div className="absolute top-1/2 w-full h-0.5 bg-neutral-200"></div>
            {spec.events.map(ev => (
              <LinearEventMarker key={ev.id} event={ev} range={viewRange} />
            ))}
          </div>
        )}

        {/* VARIANT: STATUS (Solid Bars) */}
        {spec.variant === 'status' && (
          <div className="relative w-full h-12 bg-neutral-100 rounded-md overflow-hidden">
            {spec.events.map(ev => (
              <StatusBlock key={ev.id} event={ev} range={viewRange} />
            ))}
          </div>
        )}

        {/* VARIANT: MULTILANE (Gantt style) */}
        {spec.variant === 'multilane' && spec.lanes && (
          <div className="space-y-2">
             {spec.lanes.map(lane => (
               <div key={lane.id} className="flex items-center gap-3">
                 <div className="w-20 flex-shrink-0 text-right">
                    <span className="text-xs font-bold text-neutral-600 block">{lane.label}</span>
                 </div>
                 <div className="flex-1 relative h-8 bg-neutral-100 rounded overflow-hidden">
                    {spec.events.filter(e => e.laneId === lane.id).map(ev => (
                      <StatusBlock key={ev.id} event={ev} range={viewRange} />
                    ))}
                 </div>
               </div>
             ))}
          </div>
        )}

        {/* VARIANT: FORENSIC (Layered) */}
        {spec.variant === 'forensic' && (
           <div className="relative w-full flex-1 min-h-[80px] mt-2">
              
              {/* Layer 1: Incident Window Highlight */}
              {spec.incidentWindow && (
                <div 
                  className="absolute top-0 bottom-0 bg-red-50/50 border-x border-red-200 border-dashed z-0 flex justify-center pt-2 transition-all duration-300"
                  style={{ 
                    // Incident window should move with zoom/pan. Use viewRange.
                    left: `${getPosition(spec.incidentWindow.start, viewRange.start, viewRange.end)}%`,
                    width: `${getWidth(spec.incidentWindow.start, spec.incidentWindow.end, viewRange.start, viewRange.end)}%`
                  }}
                >
                  <span className="text-[9px] font-bold text-red-400 uppercase tracking-widest bg-white/80 px-1 rounded h-fit whitespace-nowrap">
                    {spec.incidentWindow.label}
                  </span>
                </div>
              )}

              {/* Layer 2: State Segments (Background Track) */}
              <div className="absolute bottom-6 left-0 right-0 h-4 bg-neutral-100 rounded-full overflow-hidden z-10">
                 {spec.stateSegments?.map(seg => (
                   <div 
                     key={seg.id}
                     className={`absolute h-full ${StatusColorMap[seg.status]} opacity-80 transition-all duration-300`}
                     style={{ 
                       left: `${getPosition(seg.start, viewRange.start, viewRange.end)}%`, 
                       width: `${getWidth(seg.start, seg.end, viewRange.start, viewRange.end)}%` 
                     }}
                   />
                 ))}
              </div>

              {/* Layer 3: Events (Markers on track) */}
              <div className="absolute bottom-8 left-0 right-0 h-0 z-20">
                 {spec.events.map(ev => (
                    <LinearEventMarker key={ev.id} event={ev} range={viewRange} />
                 ))}
              </div>

              {/* Layer 4: Annotations (Pins above) */}
              <div className="absolute top-0 left-0 right-0 h-full pointer-events-none">
                 {/* Enable pointer events for children only */}
                 <div className="w-full h-full relative pointer-events-auto">
                    {spec.annotations?.map(ann => (
                       <AnnotationPin key={ann.id} annotation={ann} range={viewRange} />
                    ))}
                 </div>
              </div>

           </div>
        )}

        {/* VARIANT: DENSE (Heatmap + Clusters) */}
        {spec.variant === 'dense' && (
           <div className="relative w-full flex-1 min-h-[40px] mt-1">
              {/* Background Density Heatmap - Multi-layered for organic look */}
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-50 via-emerald-100/50 to-emerald-50 rounded-lg overflow-hidden ring-1 ring-inset ring-black/5">
                 
                 {/* Low Intensity Base Noise */}
                 <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')]"></div>

                 {/* Mid Intensity Bands - Static for demo, could be dynamic */}
                 <div className="absolute top-0 bottom-0 left-[15%] w-[20%] bg-gradient-to-r from-transparent via-amber-200/40 to-transparent blur-xl"></div>

                 {/* High Intensity Hotspots */}
                 <div className="absolute top-0 bottom-0 left-[50%] w-[10%] bg-red-400/20 blur-xl"></div>
                 
                 {/* Extreme Load Vertical Bands */}
                 <div className="absolute top-0 bottom-0 left-[53%] w-[2px] bg-red-500/40 shadow-[0_0_8px_rgba(239,68,68,0.5)]"></div>
                 <div className="absolute top-0 bottom-0 left-[55%] w-[1px] bg-red-500/30"></div>
                 <div className="absolute top-0 bottom-0 left-[57%] w-[0.5px] bg-red-500/20"></div>
              </div>

              {/* Grid Overlay */}
              <div className="absolute inset-0 flex justify-between px-4 pointer-events-none">
                 {[...Array(5)].map((_, i) => (
                    <div key={i} className="w-px h-full border-r border-dashed border-neutral-300/50"></div>
                 ))}
              </div>
              
              {/* Main Track Line */}
              <div className="absolute top-1/2 w-full h-px bg-neutral-300/80"></div>

              {/* Events & Clusters */}
              {spec.events.map(ev => {
                 // Check if this is a cluster representative
                 if (ev.clusterCount && ev.clusterCount > 1) {
                    return <ClusterBubble key={ev.id} event={ev} range={viewRange} />;
                 }
                 // Standard Tick Marker
                 const left = getPosition(ev.startTime, viewRange.start, viewRange.end);
                 // Filter if out of view
                 if (left < -2 || left > 102) return null;

                 const isCritical = ev.status === 'critical';
                 const isWarn = ev.status === 'warning';
                 const color = isCritical ? 'bg-red-500' : isWarn ? 'bg-amber-400' : 'bg-blue-400';
                 
                 return (
                    <div 
                      key={ev.id}
                      className={`absolute top-1/2 -translate-y-1/2 w-1 h-3 rounded-full ${color} hover:h-5 hover:scale-125 transition-all cursor-pointer shadow-sm`}
                      style={{ left: `${left}%` }}
                      title={`${ev.label} - ${formatTimeLabel(ev.startTime)}`}
                    ></div>
                 );
              })}

              {/* Density Legend (Conditional) */}
              <div className="absolute bottom-2 right-2 flex items-center gap-2 bg-white/90 backdrop-blur-md px-2.5 py-1.5 rounded-full border border-neutral-200 shadow-sm z-10 pointer-events-none">
                 <span className="text-[8px] font-bold text-neutral-400 uppercase tracking-widest">Density</span>
                 <div className="w-12 h-1.5 rounded-full bg-gradient-to-r from-emerald-200 via-amber-300 to-red-500"></div>
                 <span className="text-[8px] text-neutral-400 font-mono">Hi</span>
              </div>
           </div>
        )}
        
        {/* Time Axis (Common) */}
        <TimeAxis start={viewRange.start} end={viewRange.end} />

      </div>
    </Card>
  );
};


export default function ScenarioComponent({ data }) {
  if (!data || !data.variant) {
    return (
      <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#991b1b' }}>
        <strong>Missing timeline data</strong>
        <pre style={{ fontSize: 11, marginTop: 8 }}>{JSON.stringify(data, null, 2)}</pre>
      </div>
    );
  }
  return <TimelineRenderer spec={data} />;
}
