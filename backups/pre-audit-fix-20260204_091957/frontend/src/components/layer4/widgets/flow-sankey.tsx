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


import React, { useState, useEffect } from 'react';


// --- SHARED UTILS ---

const FlowPath = ({ d, color, width, opacity = 0.2, animated = false }: any) => (
  <>
    <path 
      d={d} 
      fill="none" 
      stroke={color} 
      strokeWidth={width} 
      opacity={opacity} 
      className={`transition-all duration-300 hover:opacity-60 cursor-pointer ${animated ? 'animate-pulse' : ''}`}
    />
  </>
);

const FlowNode = ({ x, y, width, height, color, label, value, subLabel }: any) => (
  <g className="group cursor-pointer">
    <rect 
      x={x} y={y} 
      width={width} height={height} 
      fill={color} 
      rx="4" 
      className="transition-all duration-300 group-hover:filter group-hover:brightness-110"
    />
    <text 
      x={x + width / 2} 
      y={y - 10} 
      textAnchor="middle" 
      className="text-[10px] font-bold fill-neutral-400 uppercase tracking-wider"
    >
      {label}
    </text>
    {value && (
      <text 
        x={x + width / 2} 
        y={y + height + 15} 
        textAnchor="middle" 
        className="text-[10px] font-mono fill-neutral-600 font-bold"
      >
        {value} kW
      </text>
    )}
    {subLabel && (
      <text 
        x={x + width / 2} 
        y={y + height + 26} 
        textAnchor="middle" 
        className="text-[9px] font-mono fill-neutral-400"
      >
        {subLabel}
      </text>
    )}
  </g>
);

// Helper to check connection for Standard Sankey
const isConnected = (link: any, nodeId: string | null) => {
  if (!nodeId) return false;
  return link.source === nodeId || link.target === nodeId;
};

// --- VARIANTS ---

// 1. STANDARD SANKEY - INTERACTIVE
const SankeyStandard = ({ data }: { data: any }) => {
  const [hoveredElement, setHoveredElement] = useState<string | null>(null); // 'node-id' or 'link-id'
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

  const svgW = 600;
  const svgH = 300;

  // Helper logic for opacity
  const getLinkOpacity = (sourceId: string, targetId: string) => {
    // If nothing interacting, default low opacity
    if (!hoveredElement && !selectedNode) return 0.2;

    const activeId = hoveredElement || selectedNode?.id;
    
    // Check if this link connects to the active node
    if (sourceId === activeId || targetId === activeId) return 0.6; // Highlight
    
    // Check secondary connections (if node A is active, and Link connects A->B, highlight B)
    // For simple demo, we just highlight direct connections.
    
    return 0.05; // Dim unconnected
  };

  const getNodeOpacity = (nodeId: string) => {
     if (!hoveredElement && !selectedNode) return 1;
     const activeId = hoveredElement || selectedNode?.id;
     
     if (nodeId === activeId) return 1;
     
     // Check if connected via any link in data
     const isLinked = data.links.some((l: any) => 
        (l.source === activeId && l.target === nodeId) || 
        (l.target === activeId && l.source === nodeId)
     );
     
     return isLinked ? 1 : 0.3; // Dim unconnected nodes
  }

  return (
    <div className="w-full h-full flex flex-col relative bg-neutral-50 rounded-lg border border-neutral-100 p-4">
      {/* Detail Overlay */}
      {selectedNode && (
        <div className="absolute top-4 left-4 z-20 bg-white/90 backdrop-blur border border-neutral-200 shadow-lg p-4 rounded-xl w-64 animate-in fade-in slide-in-from-top-4">
             <div className="flex justify-between items-start mb-2">
                 <span className="text-[10px] uppercase font-bold text-neutral-400">Node Details</span>
                 <button onClick={() => setSelectedNode(null)} className="text-neutral-400 hover:text-neutral-900">&times;</button>
             </div>
             <div className="text-lg font-bold text-neutral-900 mb-1">{selectedNode.label}</div>
             <div className="flex items-baseline gap-2 mb-3">
                 <span className="text-3xl font-bold tracking-tighter">{selectedNode.value}</span>
                 <span className="text-xs font-mono text-neutral-500">kW</span>
             </div>
             
             {/* Mock specific breakdown */}
             <div className="space-y-1">
                 <div className="flex justify-between text-xs">
                    <span className="text-neutral-500">Efficiency</span>
                    <span className="font-bold text-green-600">98.2%</span>
                 </div>
                 <div className="flex justify-between text-xs">
                    <span className="text-neutral-500">Status</span>
                    <span className="font-bold text-neutral-900">Normal</span>
                 </div>
             </div>
        </div>
      )}

      <svg width="100%" height="100%" viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="xMidYMid meet" onClick={() => setSelectedNode(null)}>
        {/* Links - We map these manually to data for the demo structure, but apply dynamic opacity */}
        <FlowPath 
            d="M 120 80 C 250 80, 250 150, 400 150" 
            color={COLORS.sources.Grid} 
            width={40} 
            opacity={getLinkOpacity('src1', 'bus')}
        />
        <FlowPath 
            d="M 120 220 C 250 220, 250 150, 400 150" 
            color={COLORS.sources.Solar} 
            width={15} 
            opacity={getLinkOpacity('src2', 'bus')}
        />
        
        <FlowPath d="M 400 150 C 500 150, 500 60, 550 60" color={COLORS.loads.Machinery} width={30} opacity={getLinkOpacity('bus', 'load1')} />
        <FlowPath d="M 400 150 C 500 150, 500 150, 550 150" color={COLORS.loads.HVAC} width={20} opacity={getLinkOpacity('bus', 'load2')} />
        <FlowPath d="M 400 150 C 500 150, 500 240, 550 240" color={COLORS.loads.Lighting} width={10} opacity={getLinkOpacity('bus', 'load3')} />

        {/* Nodes Wrapper for easier interaction mapping */}
        {/* Helper to render interactive group */}
        {[
            { id: 'src1', x: 80, y: 60, w: 40, h: 40, c: COLORS.sources.Grid, l: 'Grid', v: 450 },
            { id: 'src2', x: 80, y: 212, w: 40, h: 16, c: COLORS.sources.Solar, l: 'Solar', v: 120 },
            { id: 'bus', x: 380, y: 110, w: 40, h: 80, c: '#404040', l: 'Main Bus', v: 570 },
            { id: 'load1', x: 550, y: 45, w: 40, h: 30, c: COLORS.loads.Machinery, l: 'Mach', v: 300 },
            { id: 'load2', x: 550, y: 140, w: 40, h: 20, c: COLORS.loads.HVAC, l: 'HVAC', v: 200 },
            { id: 'load3', x: 550, y: 235, w: 40, h: 10, c: COLORS.loads.Lighting, l: 'Light', v: 70 },
        ].map(node => (
            <g 
                key={node.id}
                onMouseEnter={() => setHoveredElement(node.id)} 
                onMouseLeave={() => setHoveredElement(null)}
                onClick={(e) => { e.stopPropagation(); setSelectedNode(data.nodes.find((n:any) => n.id === node.id)); }}
                opacity={getNodeOpacity(node.id)}
                className="transition-opacity duration-300"
            >
                <FlowNode x={node.x} y={node.y} width={node.w} height={node.h} color={node.c} label={node.l} value={node.v} />
            </g>
        ))}
      </svg>
      
      {!selectedNode && (
          <div className="absolute bottom-4 left-4 text-[10px] text-neutral-400 pointer-events-none">
              Hover nodes to trace flow • Click for details
          </div>
      )}
    </div>
  );
};

// 2. ENERGY BALANCE (WITH LOSSES) - INTERACTIVE
const SankeyBalance = ({ data }: { data: any }) => {
  const [focusedPart, setFocusedPart] = useState<'inflow' | 'useful' | 'loss' | null>(null);

  const total = data.inflow;
  const useful = data.outflow;
  const losses = data.losses;
  const efficiency = Math.round((useful / total) * 100);

  // Helper to determine opacity based on focus
  const getOpacity = (part: string) => {
    if (!focusedPart) return 0.15; // Default lower opacity for better contrast
    return focusedPart === part ? 0.8 : 0.05; // Highlight active, dim inactive
  };

  const getStrokeWidth = (part: string, base: number) => {
      if (focusedPart === part) return base + 10;
      return base;
  }

  return (
    <div className="w-full h-full flex flex-col relative bg-white rounded-lg p-6">
        {/* Header Stats - Dynamic */}
        <div className="flex justify-between items-end mb-3 border-b border-gray-100 pb-3 shrink-0">
             <div>
                 <div className="text-[10px] uppercase font-bold text-neutral-400 tracking-widest mb-1">Efficiency</div>
                 <div className="flex items-baseline gap-2">
                     <div className="text-4xl font-bold text-neutral-900 tracking-tighter">{efficiency}%</div>
                     <div className={`text-xs font-bold px-2 py-0.5 rounded ${efficiency > 80 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                        {efficiency > 80 ? 'GOOD' : 'ATTENTION'}
                     </div>
                 </div>
             </div>
             
             {/* Dynamic description based on hover */}
             <div className="text-right">
                <div className="text-[10px] uppercase font-bold text-neutral-400 tracking-widest mb-1">
                    {focusedPart === 'loss' ? 'Losses Detected' : focusedPart === 'useful' ? 'Useful Output' : 'Total Input'}
                </div>
                <div className={`text-2xl font-bold transition-colors font-mono ${
                    focusedPart === 'loss' ? 'text-red-500' : 
                    focusedPart === 'useful' ? 'text-green-600' : 'text-neutral-900'
                }`}>
                    {focusedPart === 'loss' ? `${losses} kW` : 
                     focusedPart === 'useful' ? `${useful} kW` : `${total} kW`}
                </div>
             </div>
        </div>

       {/* Interactive Diagram */}
       <div className="flex-1 relative flex items-center justify-center">
           <svg width="100%" height="100%" viewBox="0 0 800 300" className="max-w-4xl overflow-visible">
              <defs>
                  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#a3a3a3" />
                  </marker>
              </defs>

              {/* Input Segment */}
              <g 
                onMouseEnter={() => setFocusedPart('inflow')} 
                onMouseLeave={() => setFocusedPart(null)}
                className="cursor-pointer"
              >
                  <path 
                    d="M 50 150 L 300 150" 
                    stroke={COLORS.primary} 
                    strokeWidth={getStrokeWidth('inflow', 80)} 
                    fill="none" 
                    opacity={focusedPart === 'inflow' ? 0.2 : getOpacity('inflow')} 
                    className="transition-all duration-300 ease-out"
                  />
                   {/* Flow Arrow Overlay */}
                   {focusedPart === 'inflow' && (
                        <path d="M 100 150 L 250 150" stroke={COLORS.primary} strokeWidth="2" markerEnd="url(#arrow)" />
                   )}
                   <text x="100" y="100" className={`text-xs font-bold transition-colors duration-300 ${focusedPart === 'inflow' ? 'fill-neutral-900' : 'fill-neutral-400'}`}>INPUT SOURCE</text>
              </g>

              {/* Useful Output Segment */}
              <g 
                onMouseEnter={() => setFocusedPart('useful')} 
                onMouseLeave={() => setFocusedPart(null)}
                className="cursor-pointer"
              >
                  <path 
                    d="M 300 150 L 750 150" 
                    stroke={COLORS.success} 
                    strokeWidth={getStrokeWidth('useful', 60)} 
                    fill="none" 
                    opacity={focusedPart === 'useful' ? 0.2 : getOpacity('useful')} 
                    className="transition-all duration-300 ease-out"
                  />
                  {focusedPart === 'useful' && (
                        <path d="M 350 150 L 700 150" stroke={COLORS.success} strokeWidth="2" markerEnd="url(#arrow)" />
                   )}
                  <text x="600" y="100" className={`text-xs font-bold transition-colors duration-300 ${focusedPart === 'useful' ? 'fill-green-700' : 'fill-neutral-400'}`}>USEFUL LOAD</text>
              </g>
              
              {/* Loss Segment */}
              <g 
                onMouseEnter={() => setFocusedPart('loss')} 
                onMouseLeave={() => setFocusedPart(null)}
                className="cursor-pointer"
              >
                  <path 
                    d="M 300 150 Q 400 150 450 250" 
                    stroke={COLORS.critical} 
                    strokeWidth={getStrokeWidth('loss', 20)} 
                    fill="none" 
                    opacity={focusedPart === 'loss' ? 0.3 : getOpacity('loss')} 
                    strokeDasharray={focusedPart === 'loss' ? "0" : "4 4"}
                    className="transition-all duration-300 ease-out"
                  />
                   {focusedPart === 'loss' && (
                        <circle cx="450" cy="250" r="10" fill={COLORS.critical} className="animate-pulse" />
                   )}
                  <text x="460" y="270" className={`text-xs font-bold transition-colors duration-300 ${focusedPart === 'loss' ? 'fill-red-600' : 'fill-neutral-400'}`}>LOSSES</text>
              </g>
              
              {/* Central Node */}
              <rect 
                x="280" y="100" width="40" height="100" 
                fill={COLORS.neutral['200']} 
                rx="4" 
                className="transition-all duration-300 hover:fill-neutral-300"
              />
           </svg>

           {/* Overlay Loss Breakdown (Only visible when loss is focused) */}
           <div className={`
                absolute bottom-4 right-4 bg-white/95 backdrop-blur border border-red-100 p-4 rounded-xl shadow-xl z-20
                transition-all duration-300 transform origin-bottom-right
                ${focusedPart === 'loss' ? 'opacity-100 scale-100 translate-y-0' : 'opacity-0 scale-95 translate-y-4 pointer-events-none'}
           `}>
                <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-red-500 mb-3 border-b border-red-100 pb-2">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                    Loss Breakdown
                </div>
                <div className="flex flex-col gap-3">
                    {data.lossBreakdown.map((l: any) => (
                        <div key={l.label} className="flex items-center justify-between w-56 text-xs group">
                            <span className="text-neutral-500 group-hover:text-neutral-900 transition-colors">{l.label}</span>
                            <div className="flex items-center gap-2">
                                <div className="w-16 h-1 bg-red-50 rounded-full overflow-hidden">
                                    <div 
                                        className="h-full bg-red-400 rounded-full" 
                                        style={{ width: `${(l.value / losses) * 100}%` }}
                                    ></div>
                                </div>
                                <span className="font-mono font-bold text-neutral-900 w-12 text-right">{l.value} <span className="text-[9px] text-neutral-400 font-normal">kW</span></span>
                            </div>
                        </div>
                    ))}
                    <div className="border-t border-red-100 pt-2 mt-1 flex justify-between text-xs font-bold text-red-700">
                        <span>TOTAL LOSS</span>
                        <span>{losses} kW</span>
                    </div>
                </div>
           </div>
       </div>
    </div>
  );
};

// 3. MULTI SOURCE (INTERACTIVE)
const SankeyMultiSource = ({ data }: { data: any }) => {
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);

  const handleSourceClick = (label: string) => {
      if (selectedLabel === label) {
          setSelectedLabel(null);
      } else {
          setSelectedLabel(label);
      }
  };

  const activeData = selectedLabel ? data.sources.find((s:any) => s.label === selectedLabel) : null;
  const displayValue = activeData ? activeData.value : data.totalLoad;
  const displayLabel = activeData ? "CONTRIBUTION" : "TOTAL LOAD";
  const displaySub = activeData ? `${Math.round((activeData.value / data.totalLoad) * 100)}% of System` : "kW";

  return (
    <div className="w-full h-full p-6 flex gap-8">
      {/* Sources Column */}
      <div className="flex flex-col justify-center gap-4 w-1/3 z-10">
         {data.sources.map((src: any) => {
            const isSelected = selectedLabel === src.label;
            const isDimmed = selectedLabel && !isSelected;
            
            return (
            <div 
                key={src.label} 
                onClick={() => handleSourceClick(src.label)}
                className={`
                    relative p-4 rounded-xl border cursor-pointer transition-all duration-300
                    ${isSelected 
                        ? 'bg-neutral-900 border-neutral-900 shadow-lg scale-105' 
                        : isDimmed 
                            ? 'bg-gray-50 border-gray-100 opacity-40 grayscale' 
                            : 'bg-white border-gray-200 hover:border-blue-400 hover:shadow-md'
                    }
                `}
            >
                <div className={`text-[10px] uppercase font-bold mb-1 transition-colors ${isSelected ? 'text-neutral-400' : 'text-neutral-400'}`}>
                    {src.label}
                </div>
                <div className={`text-xl font-bold transition-colors ${isSelected ? 'text-white' : 'text-neutral-900'}`}>
                    {src.value} <span className="text-xs font-normal opacity-60">kW</span>
                </div>
                
                {/* Status Indicator */}
                <div className="absolute right-4 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-white shadow-sm" style={{ backgroundColor: src.color }}></div>
            </div>
         )})}
      </div>

      {/* Aggregation Visual */}
      <div className="flex-1 flex items-center justify-center relative">
         {/* Dynamic Connections Layer */}
         <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
             <svg width="100%" height="100%" viewBox="0 0 400 400" preserveAspectRatio="none">
                {data.sources.map((src: any, index: number) => {
                    const count = data.sources.length;
                    const spacing = 90;
                    const startY = (index - (count - 1) / 2) * spacing + 200;
                    const endY = 200;
                    
                    const isSelected = selectedLabel === src.label;
                    const isDimmed = selectedLabel && !isSelected;
                    
                    const strokeOpacity = isSelected ? 0.8 : (isDimmed ? 0.05 : 0.15);
                    const strokeWidth = isSelected ? 24 : (src.value / data.totalLoad) * 40; 
                    
                    return (
                        <path 
                            key={src.label}
                            d={`M 0 ${startY} C 150 ${startY}, 150 ${endY}, 200 ${endY}`} 
                            stroke={src.color} 
                            strokeWidth={Math.max(2, strokeWidth)} 
                            fill="none" 
                            opacity={strokeOpacity}
                            className="transition-all duration-500"
                        />
                    );
                })}
             </svg>
         </div>
         
         {/* Center Node */}
         <div className="relative z-20">
             <div className={`
                w-56 h-56 rounded-full border-8 bg-white shadow-2xl flex flex-col items-center justify-center
                transition-all duration-500
                ${selectedLabel ? 'border-neutral-900 scale-110' : 'border-neutral-100'}
             `}>
                 <span className="text-[10px] uppercase font-bold text-neutral-400 tracking-widest mb-2">
                    {displayLabel}
                 </span>
                 <span className={`text-5xl font-bold tracking-tighter transition-colors ${selectedLabel ? 'text-blue-600' : 'text-neutral-900'}`}>
                    {displayValue}
                 </span>
                 <span className="text-xs font-mono text-neutral-500 mt-2 font-medium">
                    {displaySub}
                 </span>
                 
                 {selectedLabel && (
                     <button 
                        onClick={(e) => { e.stopPropagation(); setSelectedLabel(null); }}
                        className="absolute -bottom-8 text-[10px] text-neutral-400 hover:text-neutral-900 underline underline-offset-4"
                     >
                        RESET VIEW
                     </button>
                 )}
             </div>
         </div>
      </div>
    </div>
  );
};

// 4. LAYERED (HIERARCHY) - INTERACTIVE
const SankeyLayered = ({ data }: { data: any }) => {
  const [activeLevel, setActiveLevel] = useState<number>(0);
  
  // Generate dummy children for the active level
  // In a real app, this would come from the data source, here we simulate drill down.
  const activeChildren = Array.from({ length: data.levels[activeLevel].count || 4 }).map((_, i) => ({
      id: i,
      label: `${data.levels[activeLevel].name} ${String.fromCharCode(65 + i)}`,
      value: Math.floor(Math.random() * 500) + 100,
      status: Math.random() > 0.8 ? 'warning' : 'normal'
  }));

  return (
    <div className="w-full h-full flex flex-col p-6 bg-neutral-50 rounded-xl">
       {/* Level Selectors */}
       <div className="flex justify-between items-center relative z-10 mb-8">
          {data.levels.map((lvl: any, idx: number) => {
             const isActive = idx === activeLevel;
             return (
             <div key={lvl.name} className="flex flex-col items-center gap-4 w-32 relative group">
                 <div 
                    onClick={() => setActiveLevel(idx)}
                    className={`
                        w-full h-24 rounded-lg border flex flex-col items-center justify-center transition-all cursor-pointer relative z-10
                        ${isActive 
                            ? 'bg-neutral-900 border-neutral-900 shadow-lg text-white scale-105' 
                            : 'bg-white border-neutral-200 text-neutral-900 hover:border-blue-300 hover:shadow-md'
                        }
                    `}
                 >
                    <span className="text-xl font-bold">{lvl.value ? lvl.value : lvl.count}</span>
                    <span className={`text-[10px] uppercase font-bold ${isActive ? 'text-neutral-400' : 'text-neutral-400'}`}>{lvl.value ? 'kW' : 'Items'}</span>
                    
                    {/* Active Indicator Arrow */}
                    {isActive && (
                        <div className="absolute -bottom-2 w-4 h-4 bg-neutral-900 rotate-45"></div>
                    )}
                 </div>
                 <span className={`text-xs font-bold uppercase tracking-widest transition-colors ${isActive ? 'text-neutral-900' : 'text-neutral-400'}`}>{lvl.name}</span>
                 
                 {/* Connection Line to next level */}
                 {idx < data.levels.length - 1 && (
                     <div className="absolute top-12 -right-16 w-16 h-0.5 bg-neutral-200 -z-10"></div>
                 )}
             </div>
          )})}
       </div>
       
       {/* Drill Down View (Children) */}
       <div className="flex-1 bg-white rounded-xl border border-neutral-200 p-6 animate-in fade-in slide-in-from-bottom-2 duration-500 shadow-sm overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4 border-b border-neutral-100 pb-2">
                <h4 className="text-xs font-bold uppercase tracking-widest text-neutral-500">
                    Breakdown: {data.levels[activeLevel].name} ({activeChildren.length})
                </h4>
                <div className="text-[10px] font-mono text-neutral-400">SORT BY: CONSUMPTION</div>
            </div>
            
            <div className="flex-1 overflow-auto pr-2 grid grid-cols-2 gap-4">
                {activeChildren.map(child => (
                    <div key={child.id} className="flex items-center justify-between p-3 rounded-lg border border-neutral-100 bg-neutral-50 hover:bg-white hover:border-blue-200 transition-colors group">
                        <div className="flex items-center gap-3">
                            <div className={`w-2 h-2 rounded-full ${child.status === 'warning' ? 'bg-amber-500' : 'bg-green-500'}`}></div>
                            <span className="text-xs font-bold text-neutral-700">{child.label}</span>
                        </div>
                        <div className="flex flex-col items-end">
                            <span className="text-sm font-bold font-mono text-neutral-900 group-hover:text-blue-600">{child.value} kW</span>
                            <div className="w-24 h-1 bg-neutral-200 rounded-full mt-1 overflow-hidden">
                                <div className="h-full bg-neutral-400" style={{ width: `${Math.random() * 100}%` }}></div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
       </div>
    </div>
  );
};

// 5. TIME SLICED (ANIMATED)
const SankeyTimeSliced = ({ data }: { data: any }) => {
  const [frameIdx, setFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(() => {
        setFrameIdx(prev => (prev + 1) % data.frames.length);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isPlaying, data.frames.length]);

  const frame = data.frames[frameIdx];

  // Dynamic calculations for demo
  const gridHeight = frame.grid / 5;
  const solarHeight = frame.solar / 5;
  const total = frame.grid + frame.solar;

  return (
    <div className="w-full h-full flex flex-col">
       {/* Viz Area */}
       <div className="flex-1 relative flex items-center justify-center bg-white rounded-lg border border-gray-100 m-4 overflow-hidden">
           <svg width="600" height="200" className="transition-all duration-500">
              {/* Grid Flow */}
              <path 
                d="M 50 80 C 200 80, 200 120, 500 120" 
                fill="none" 
                stroke={COLORS.sources.Grid} 
                strokeWidth={gridHeight} 
                opacity="0.3"
                className="transition-all duration-500 ease-in-out"
              />
              {/* Solar Flow */}
              <path 
                d="M 50 160 C 200 160, 200 120, 500 120" 
                fill="none" 
                stroke={COLORS.sources.Solar} 
                strokeWidth={solarHeight} 
                opacity="0.3"
                className="transition-all duration-500 ease-in-out"
              />
              
              {/* Output */}
              <path 
                d="M 500 120 L 580 120" 
                stroke="#171717" 
                strokeWidth={(gridHeight + solarHeight)} 
                opacity="0.1" 
                className="transition-all duration-500 ease-in-out"
              />
           </svg>
           
           <div className="absolute top-4 right-4 text-right">
               <div className="text-3xl font-bold font-mono text-neutral-900 tabular-nums">{total} kW</div>
               <div className="text-[10px] uppercase font-bold text-neutral-400">Total Consumption</div>
           </div>
       </div>

       {/* Controls */}
       <div className="h-16 bg-neutral-900 text-white flex items-center px-6 justify-between rounded-b-xl">
           <div className="flex items-center gap-4">
               <button 
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center hover:bg-neutral-200"
               >
                   {isPlaying ? (
                       <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                   ) : (
                       <svg className="w-3 h-3 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                   )}
               </button>
               <span className="text-xs font-mono text-neutral-400">
                  {frame.time}
               </span>
           </div>
           
           {/* Timeline dots */}
           <div className="flex gap-2">
               {data.frames.map((f: any, i: number) => (
                   <button 
                     key={f.time}
                     onClick={() => setFrameIdx(i)}
                     className={`w-2 h-2 rounded-full transition-all ${i === frameIdx ? 'bg-blue-500 scale-125' : 'bg-neutral-700'}`}
                   />
               ))}
           </div>
       </div>
    </div>
  );
};


const WidgetRenderer = ({ spec }: { spec: DesignSpec }) => {
    return (
        <div className="bg-white rounded-xl border border-gray-100 shadow-sm flex flex-col overflow-hidden h-full">
            {/* Standard Widget Header */}
            <div className="flex justify-between items-center px-4 py-2 border-b border-gray-50 shrink-0">
                <div>
                    <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                        <h3 className="text-[10px] uppercase font-bold tracking-widest text-neutral-500">
                            {spec.coreWidget}
                        </h3>
                    </div>
                    <h2 className="text-xl font-bold tracking-tight text-neutral-900 mt-1">
                        {spec.variant.replace('FLOW_SANKEY_', '').replace('_', ' ')}
                    </h2>
                </div>
                <div className="flex gap-2">
                     <div className="px-2 py-1 bg-neutral-100 rounded text-[10px] font-mono text-neutral-500">
                        {spec.purpose.substring(0, 30)}...
                     </div>
                </div>
            </div>

            {/* Visualizer Container */}
            <div className="flex-1 bg-neutral-50/50 relative">
                {spec.variant === 'FLOW_SANKEY_STANDARD' && <SankeyStandard data={spec.demoData} />}
                {spec.variant === 'FLOW_SANKEY_ENERGY_BALANCE' && <SankeyBalance data={spec.demoData} />}
                {spec.variant === 'FLOW_SANKEY_MULTI_SOURCE' && <SankeyMultiSource data={spec.demoData} />}
                {spec.variant === 'FLOW_SANKEY_LAYERED' && <SankeyLayered data={spec.demoData} />}
                {spec.variant === 'FLOW_SANKEY_TIME_SLICED' && <SankeyTimeSliced data={spec.demoData} />}
            </div>

            {/* Standard Footer */}
            <div className="flex justify-between items-center px-4 py-2 bg-white border-t border-gray-100 text-[10px] shrink-0">
                <div className="flex gap-4 text-neutral-400">
                    <span className="flex items-center gap-1.5">
                       <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                       LIVE FEED
                    </span>
                    <span className="font-mono text-neutral-300">|</span>
                    <span>UPDATED 2s AGO</span>
                </div>
                <div className="flex gap-2">
                    {spec.interactions.map(interaction => (
                        <span key={interaction} className="px-2 py-0.5 rounded-full bg-neutral-100 text-neutral-500 border border-neutral-200">
                            {interaction}
                        </span>
                    ))}
                </div>
            </div>
        </div>
    );
};


export default function ScenarioComponent({ data }) {
  if (!data) {
    return (
      <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#991b1b' }}>
        <strong>Missing flow data</strong>
      </div>
    );
  }
  return <WidgetRenderer spec={data} />;
}
