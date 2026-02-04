// @ts-nocheck
import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  HelpCircle,
  Info,
  XOctagon
} from 'lucide-react';


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




interface Props {
  data: DemoData;
  state: WidgetState;
}

const ValueHeatmap: React.FC<Props> = ({ data, state }) => {
  const [hoverCell, setHoverCell] = useState<HeatmapCell | null>(null);
  const [selectedCell, setSelectedCell] = useState<HeatmapCell | null>(null);

  if (state.isLoading) return <div className="h-full w-full bg-neutral-100 animate-pulse rounded-lg" />;
  if (state.isEmpty || !data.dataset) return <div className="h-full w-full flex items-center justify-center text-neutral-400">No Data</div>;

  const { rows, cols, cells, min, max, unit } = data.dataset;

  const getCell = (rId: string, cId: string) => cells.find(c => c.rowId === rId && c.colId === cId);

  // Sequential Blue Scale
  const getColor = (val: number) => {
    const ratio = (val - min) / (max - min);
    // Interpolate roughly from neutral-50 (#f9fafb) to blue-600 (#2563eb)
    // Simplified opacity approach for demo
    return `rgba(37, 99, 235, ${Math.max(0.1, ratio)})`; 
  };

  return (
    <div className="h-full flex flex-col relative pt-4">
      <div className="flex justify-between items-start mb-4">
        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{data.label}</span>
        <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-neutral-400">MIN: {min}{unit}</span>
            <div className="w-16 h-2 rounded bg-gradient-to-r from-blue-50 to-blue-600"></div>
            <span className="text-[10px] font-mono text-neutral-400">MAX: {max}{unit}</span>
        </div>
      </div>

      <div className="flex-1 overflow-auto custom-scrollbar relative">
        <div 
            className="grid gap-[2px]"
            style={{ 
                gridTemplateColumns: `auto repeat(${cols.length}, minmax(40px, 1fr))` 
            }}
        >
          {/* Header Row */}
          <div className="sticky top-0 left-0 z-20 bg-white"></div>
          {cols.map(c => (
            <div 
                key={c.id} 
                className={`sticky top-0 z-10 bg-white text-center py-2 border-b-2 transition-colors ${
                    (hoverCell?.colId === c.id || selectedCell?.colId === c.id) ? 'border-blue-500 text-blue-600' : 'border-neutral-100 text-neutral-400'
                }`}
            >
              <span className="text-[10px] font-bold uppercase tracking-wider">{c.label}</span>
            </div>
          ))}

          {/* Rows */}
          {rows.map(r => (
            <React.Fragment key={r.id}>
              <div 
                className={`sticky left-0 z-10 bg-white flex items-center justify-end pr-3 border-r-2 transition-colors ${
                    (hoverCell?.rowId === r.id || selectedCell?.rowId === r.id) ? 'border-blue-500 text-blue-600' : 'border-neutral-100 text-neutral-400'
                }`}
              >
                 <span className="text-[10px] font-bold uppercase tracking-wider">{r.label}</span>
              </div>
              {cols.map(c => {
                const cell = getCell(r.id, c.id);
                const isHovered = hoverCell === cell;
                const isSelected = selectedCell === cell;
                const isActiveRowCol = (hoverCell?.rowId === r.id || hoverCell?.colId === c.id || selectedCell?.rowId === r.id || selectedCell?.colId === c.id);
                
                return (
                  <div
                    key={`${r.id}-${c.id}`}
                    onMouseEnter={() => setHoverCell(cell || { rowId: r.id, colId: c.id, value: 0 })}
                    onMouseLeave={() => setHoverCell(null)}
                    onClick={() => setSelectedCell(cell || null)}
                    className={`
                        h-10 rounded-sm flex items-center justify-center cursor-pointer transition-all duration-200 relative
                        ${!cell ? 'bg-neutral-100' : ''} 
                        ${isSelected ? 'ring-2 ring-blue-600 z-10 scale-105 shadow-sm' : ''}
                        ${!isSelected && isActiveRowCol ? 'opacity-100' : 'opacity-90'}
                        ${!isSelected && !isActiveRowCol && (selectedCell || hoverCell) ? 'opacity-40' : ''}
                    `}
                    style={{ backgroundColor: cell ? getColor(cell.value) : undefined }}
                  >
                    {cell ? (
                        <span className={`text-[10px] font-bold ${cell.value > (min + (max-min)/2) ? 'text-white' : 'text-blue-900'}`}>
                            {cell.value}
                        </span>
                    ) : (
                        state.isError || (state.isLoading === false) ? <span className="text-neutral-300 text-[8px]">N/A</span> : null
                    )}

                    {/* Tooltip */}
                    {isHovered && cell && (
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-neutral-900 text-white p-2 rounded shadow-lg z-50 min-w-[100px] pointer-events-none">
                            <div className="flex flex-col items-center">
                                <span className="text-[10px] uppercase font-bold text-neutral-400 mb-1">{r.label} • {c.label}</span>
                                <span className="text-lg font-bold leading-none">{cell.value}<span className="text-xs font-normal text-neutral-400 ml-0.5">{unit}</span></span>
                            </div>
                            <div className="w-2 h-2 bg-neutral-900 rotate-45 absolute -bottom-1 left-1/2 -translate-x-1/2"></div>
                        </div>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      {selectedCell && (
          <div className="mt-4 p-2 bg-blue-50 border border-blue-100 rounded flex items-center justify-between animate-in slide-in-from-bottom-2 fade-in duration-300">
             <span className="text-xs text-blue-900 font-medium">Selected: <span className="font-bold">{rows.find(r=>r.id===selectedCell.rowId)?.label}</span> / <span className="font-bold">{cols.find(c=>c.id===selectedCell.colId)?.label}</span></span>
             <button onClick={() => setSelectedCell(null)} className="text-[10px] uppercase font-bold text-blue-500 hover:text-blue-700">Clear</button>
          </div>
      )}
    </div>
  );
};


interface Props {
  data: DemoData;
  state: WidgetState;
}

const CorrelationMatrix: React.FC<Props> = ({ data, state }) => {
  const [hoverCell, setHoverCell] = useState<HeatmapCell | null>(null);
  
  if (state.isLoading) return <div className="h-full w-full bg-neutral-100 animate-pulse rounded-lg" />;
  if (state.isEmpty || !data.dataset) return <div className="h-full w-full flex items-center justify-center text-neutral-400">No Data</div>;

  const { rows, cols, cells } = data.dataset;
  const getCell = (rId: string, cId: string) => cells.find(c => c.rowId === rId && c.colId === cId);

  // Diverging Scale: -1 (Red) -> 0 (White) -> 1 (Blue)
  const getColor = (val: number) => {
    if (val === 1) return '#2563eb'; // Blue-600
    if (val === -1) return '#dc2626'; // Red-600
    if (val > 0) {
        // White to Blue
        const intensity = Math.floor(val * 255);
        return `rgba(37, 99, 235, ${val})`;
    }
    if (val < 0) {
        // White to Red
        return `rgba(220, 38, 38, ${Math.abs(val)})`;
    }
    return '#ffffff';
  };

  return (
    <div className="h-full flex flex-col">
       <div className="flex justify-between items-start mb-4">
        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{data.label}</span>
        <div className="flex items-center gap-1">
             <div className="w-3 h-3 bg-red-600 rounded-sm"></div>
             <span className="text-[10px] text-neutral-400 mr-2">-1</span>
             <div className="w-3 h-3 bg-white border border-gray-200 rounded-sm"></div>
             <span className="text-[10px] text-neutral-400 mr-2">0</span>
             <div className="w-3 h-3 bg-blue-600 rounded-sm"></div>
             <span className="text-[10px] text-neutral-400">1</span>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center overflow-hidden">
        <div
            className="grid gap-[2px] w-full h-full max-w-full"
            style={{
                gridTemplateColumns: `minmax(30px, auto) repeat(${cols.length}, 1fr)`,
                gridTemplateRows: `auto repeat(${rows.length}, 1fr)`,
            }}
        >
          <div /> {/* Corner */}
          {cols.map(c => (
             <div key={c.id} className="text-center pb-1 min-w-0 overflow-hidden">
                 <span className="text-[10px] font-bold text-neutral-500 writing-mode-vertical truncate">{c.label}</span>
             </div>
          ))}

          {rows.map(r => (
            <React.Fragment key={r.id}>
              <div className="flex items-center justify-end pr-1 min-w-0">
                 <span className="text-[10px] font-bold text-neutral-500 truncate">{r.label}</span>
              </div>
              {cols.map(c => {
                const cell = getCell(r.id, c.id);
                // Symmetric highlighting
                const isMirror = hoverCell && (
                    (hoverCell.rowId === c.id && hoverCell.colId === r.id) ||
                    (hoverCell.rowId === r.id && hoverCell.colId === c.id)
                );

                return (
                  <div
                    key={`${r.id}-${c.id}`}
                    onMouseEnter={() => setHoverCell(cell || null)}
                    onMouseLeave={() => setHoverCell(null)}
                    className={`
                        aspect-square flex items-center justify-center text-[10px] font-mono cursor-pointer transition-all min-w-0
                        ${!cell ? 'bg-neutral-50' : ''}
                        ${isMirror ? 'ring-2 ring-neutral-900 z-10' : ''}
                        hover:ring-2 hover:ring-neutral-900 hover:z-20
                    `}
                    style={{ backgroundColor: cell ? getColor(cell.value) : undefined }}
                  >
                     {cell && (
                         <span className={Math.abs(cell.value) > 0.5 ? 'text-white font-bold' : 'text-neutral-900'}>
                             {cell.value > 0 ? cell.value.toFixed(2).replace('0.', '.') : cell.value === 1 ? '1.0' : cell.value.toFixed(1)}
                         </span>
                     )}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};


interface Props {
  data: DemoData;
  state: WidgetState;
}

const CalendarHeatmap: React.FC<Props> = ({ data, state }) => {
  if (state.isLoading) return <div className="h-full w-full bg-neutral-100 animate-pulse rounded-lg" />;
  if (state.isEmpty || !data.dataset) return <div className="h-full w-full flex items-center justify-center text-neutral-400">No Data</div>;

  const { rows, cols, cells, min, max } = data.dataset; // Rows = Weeks, Cols = Days
  const getCell = (rId: string, cId: string) => cells.find(c => c.rowId === rId && c.colId === cId);

  const getColor = (val: number) => {
    // Green Scale
    const ratio = (val - min) / (max - min);
    // bg-green-50 to bg-green-600
    if (ratio === 0) return '#f0fdf4'; // green-50
    if (ratio < 0.25) return '#bbf7d0'; // green-200
    if (ratio < 0.5) return '#86efac'; // green-300
    if (ratio < 0.75) return '#4ade80'; // green-400
    return '#16a34a'; // green-600
  };

  return (
    <div className="h-full flex flex-col">
       <div className="flex justify-between items-start mb-6">
        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{data.label}</span>
        <span className="text-[10px] font-mono text-neutral-400">{new Date(data.timestamp).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</span>
      </div>

      <div className="flex-1 overflow-hidden">
        <div
            className="grid gap-[3px] h-full"
            style={{
                gridTemplateColumns: `minmax(24px, auto) repeat(${cols.length}, 1fr)`,
                gridTemplateRows: `auto repeat(${rows.length}, 1fr)`,
            }}
        >
           <div />
           {cols.map(c => (
             <div key={c.id} className="text-center mb-1 min-w-0">
                 <span className="text-[10px] font-bold text-neutral-400 truncate">{c.label}</span>
             </div>
           ))}

           {rows.map(r => (
             <React.Fragment key={r.id}>
               <div className="flex items-center justify-end pr-1 min-w-0">
                   <span className="text-[9px] font-bold text-neutral-400 truncate">{r.label}</span>
               </div>
               {cols.map(c => {
                 const cell = getCell(r.id, c.id);
                 return (
                   <div
                     key={`${r.id}-${c.id}`}
                     className={`
                        aspect-square rounded-sm flex items-center justify-center group relative cursor-pointer min-w-0
                        ${!cell ? 'bg-neutral-100' : ''}
                        hover:ring-2 hover:ring-green-500 hover:z-10 transition-all
                     `}
                     style={{ backgroundColor: cell ? getColor(cell.value) : undefined }}
                   >
                      {cell && (
                          <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-neutral-900 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 whitespace-nowrap pointer-events-none transition-opacity z-20">
                              {cell.value} {data.dataset?.unit}
                          </div>
                      )}

                      {cell && cell.value < 10 && (
                          <div className="w-1.5 h-1.5 bg-red-500 rounded-full shadow-sm"></div>
                      )}
                   </div>
                 );
               })}
             </React.Fragment>
           ))}
        </div>
      </div>
    </div>
  );
};



interface Props {
  data: DemoData;
  state: WidgetState;
}

const StatusMatrix: React.FC<Props> = ({ data, state }) => {
  if (state.isLoading) return <div className="h-full w-full bg-neutral-100 animate-pulse rounded-lg" />;
  if (state.isEmpty || !data.dataset) return <div className="h-full w-full flex items-center justify-center text-neutral-400">No Data</div>;

  const { rows, cols, cells } = data.dataset;
  const getCell = (rId: string, cId: string) => cells.find(c => c.rowId === rId && c.colId === cId);

  const getStatusStyles = (severity?: Severity) => {
    switch(severity) {
        case 'normal': return 'bg-green-100 text-green-600 border-green-200';
        case 'warning': return 'bg-amber-100 text-amber-600 border-amber-200';
        case 'critical': return 'bg-red-100 text-red-600 border-red-200';
        case 'unknown': return 'bg-neutral-100 text-neutral-400 border-neutral-200';
        default: return 'bg-neutral-50 border-neutral-100';
    }
  };

  const getIcon = (severity?: Severity) => {
    switch(severity) {
        case 'normal': return <Check className="w-3 h-3" />;
        case 'warning': return <AlertTriangle className="w-3 h-3" />;
        case 'critical': return <XOctagon className="w-3 h-3" />;
        default: return <HelpCircle className="w-3 h-3" />;
    }
  };

  return (
    <div className="h-full flex flex-col p-4">
       <div className="flex justify-between items-start mb-4">
        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{data.label}</span>
      </div>

      <div className="flex-1 overflow-auto custom-scrollbar">
         <div 
            className="grid gap-1"
            style={{ 
                gridTemplateColumns: `minmax(60px, auto) repeat(${cols.length}, 1fr)` 
            }}
        >
            <div />
            {cols.map(c => (
                <div key={c.id} className="text-center pb-1 border-b border-neutral-100">
                    <span className="text-[10px] font-bold text-neutral-400">{c.label}</span>
                </div>
            ))}

            {rows.map(r => (
                <React.Fragment key={r.id}>
                    <div className="flex items-center justify-start border-r border-neutral-100 pr-2">
                        <span className="text-[10px] font-bold text-neutral-500">{r.label}</span>
                    </div>
                    {cols.map(c => {
                        const cell = getCell(r.id, c.id);
                        const styles = getStatusStyles(cell?.severity);
                        return (
                            <div 
                                key={`${r.id}-${c.id}`}
                                className={`
                                    h-8 rounded flex items-center justify-center border transition-all cursor-pointer hover:brightness-95
                                    ${styles}
                                `}
                            >
                                {cell ? getIcon(cell.severity) : <span className="text-neutral-300">-</span>}
                            </div>
                        );
                    })}
                </React.Fragment>
            ))}
        </div>
      </div>
      
      <div className="mt-4 flex gap-4 border-t border-neutral-100 pt-2">
          {['normal', 'warning', 'critical', 'unknown'].map(s => (
              <div key={s} className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${s === 'normal' ? 'bg-green-500' : s === 'warning' ? 'bg-amber-500' : s === 'critical' ? 'bg-red-500' : 'bg-neutral-400'}`}></div>
                  <span className="text-[10px] uppercase text-neutral-400">{s}</span>
              </div>
          ))}
      </div>
    </div>
  );
};


interface Props {
  data: DemoData;
  state: WidgetState;
}

const DensityMatrix: React.FC<Props> = ({ data, state }) => {
  if (state.isLoading) return <div className="h-full w-full bg-neutral-100 animate-pulse rounded-lg" />;
  if (state.isEmpty || !data.dataset) return <div className="h-full w-full flex items-center justify-center text-neutral-400">No Data</div>;

  const { rows, cols, cells, min, max } = data.dataset;
  const getCell = (rId: string, cId: string) => cells.find(c => c.rowId === rId && c.colId === cId);

  const getColor = (val: number) => {
    // Mono Dark Sequential (Neutral-900 base)
    const ratio = (val - min) / (max - min);
    // Opacity based
    if (val === 0) return '#f5f5f5';
    return `rgba(23, 23, 23, ${Math.max(0.1, ratio)})`; 
  };

  return (
    <div className="h-full flex flex-col">
       <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">{data.label}</span>
         <span className="text-[10px] font-mono text-neutral-400">Total: {cells.reduce((a, b) => a + b.value, 0)}</span>
      </div>

      <div className="flex-1 flex flex-col justify-center">
        <div className="flex h-full">
            {/* Y Axis */}
            <div className="flex flex-col justify-between pr-2 py-2 h-full">
                {rows.map(r => (
                    <span key={r.id} className="text-[9px] font-mono text-neutral-400 h-full flex items-center">{r.label}</span>
                ))}
            </div>

            {/* Grid */}
            <div className="flex-1 flex flex-col">
                 <div className="flex-1 grid gap-[1px] bg-neutral-200 border border-neutral-200" style={{ gridTemplateColumns: `repeat(${cols.length}, 1fr)` }}>
                    {rows.map(r => (
                        cols.map(c => {
                             const cell = getCell(r.id, c.id);
                             return (
                                 <div 
                                    key={`${r.id}-${c.id}`}
                                    className="w-full h-full relative group hover:ring-2 hover:ring-blue-500 hover:z-10 transition-all"
                                    style={{ backgroundColor: cell ? getColor(cell.value) : '#fff' }}
                                 >
                                     {cell && cell.value > 0 && (
                                         <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100">
                                             <span className={`text-xs font-bold ${cell.value > max/2 ? 'text-white' : 'text-neutral-900'}`}>{cell.value}</span>
                                         </div>
                                     )}
                                 </div>
                             );
                        })
                    ))}
                 </div>
                 
                 {/* X Axis */}
                 <div className="flex justify-between pt-1">
                     {cols.map((c, i) => (
                         // Show every other label to save space if needed
                         <span key={c.id} className="text-[9px] font-mono text-neutral-400 w-full text-center">{c.label}</span>
                     ))}
                 </div>
            </div>
        </div>
      </div>
    </div>
  );
};








interface Props {
  spec: WidgetSpec;
  state: WidgetState;
}

const WidgetPreview: React.FC<Props> = ({ spec, state }) => {
  const renderWidget = () => {
    switch (spec.variant) {
      case 'VALUE_HEATMAP': return <ValueHeatmap data={spec.demoData} state={state} />;
      case 'CORRELATION_MATRIX': return <CorrelationMatrix data={spec.demoData} state={state} />;
      case 'CALENDAR_HEATMAP': return <CalendarHeatmap data={spec.demoData} state={state} />;
      case 'STATUS_MATRIX': return <StatusMatrix data={spec.demoData} state={state} />;
      case 'DENSITY_MATRIX': return <DensityMatrix data={spec.demoData} state={state} />;
      default: return <div>Unknown Variant</div>;
    }
  };

  return (
    <div className="h-full w-full">
      <div className="h-full w-full relative overflow-hidden">
        {renderWidget()}
      </div>
    </div>
  );
};


export default function ScenarioComponent({ data }) {
  const spec = data && data.spec ? data.spec : data || {};
  const state = data && data.state ? data.state : { isLoading: false, isEmpty: false, isError: false };
  return <WidgetPreview spec={spec} state={state} />;
}
