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

import { ArrowUpRight, ArrowDownRight, MoveRight, TrendingUp, TrendingDown, GripHorizontal, Target, Activity, Clock, Info, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface ComparisonRendererProps {
  spec: WidgetSpec;
}

const ComparisonRenderer: React.FC<ComparisonRendererProps> = ({ spec }) => {
  const { layout, visual, demoData, variant } = spec;
  const colors = visual.colors || {};

  // -- RENDERERS --

  const renderSideBySideVisual = () => {
    const isPositive = (demoData.delta || 0) > 0;
    const DeltaIcon = isPositive ? ArrowUpRight : ArrowDownRight;
    const deltaColor = isPositive ? 'text-emerald-600' : 'text-amber-600';
    const bgDelta = isPositive ? 'bg-emerald-50' : 'bg-amber-50';
    const borderDelta = isPositive ? 'border-emerald-100' : 'border-amber-100';

    return (
      <div className="flex items-center justify-between gap-6 h-full w-full py-4 relative">
        {/* Connector Line */}
        <div className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neutral-200 to-transparent -z-10"></div>

        {/* Side A (Primary) */}
        <div className="flex flex-col bg-white/50 backdrop-blur-sm p-4 rounded-xl border border-neutral-100 shadow-sm flex-1 group hover:border-neutral-200 transition-colors">
           <span className="text-[10px] uppercase tracking-widest font-bold text-neutral-400 mb-1 flex items-center gap-2">
             <div className="w-1.5 h-1.5 rounded-full bg-neutral-900"></div>
             {demoData.labelA}
           </span>
           <div className="flex items-baseline gap-1 mt-1">
             <span className="text-4xl font-black tracking-tighter text-neutral-900">{demoData.valueA}</span>
             <span className="text-xs font-bold text-neutral-400">{demoData.unit}</span>
           </div>
        </div>

        {/* Visual Connector / Delta */}
        <div className="flex flex-col items-center justify-center px-2 z-10">
           <div className={`flex items-center justify-center w-12 h-12 rounded-full ${bgDelta} ${deltaColor} border ${borderDelta} shadow-sm mb-1 backdrop-blur-md`}>
             <DeltaIcon className="w-5 h-5" strokeWidth={3} />
           </div>
           <div className={`text-xs font-black ${deltaColor} bg-white px-2 py-0.5 rounded-full shadow-sm border border-neutral-100`}>
             {isPositive ? '+' : ''}{demoData.deltaPct}%
           </div>
        </div>

        {/* Side B (Secondary) */}
        <div className="flex flex-col items-end text-right bg-white/50 backdrop-blur-sm p-4 rounded-xl border border-neutral-100 shadow-sm flex-1 group hover:border-neutral-200 transition-colors">
           <span className="text-[10px] uppercase tracking-widest font-bold text-neutral-400 mb-1 flex items-center gap-2 flex-row-reverse">
             <div className="w-1.5 h-1.5 rounded-full bg-neutral-300"></div>
             {demoData.labelB}
           </span>
           <div className="flex items-baseline gap-1 mt-1 flex-row-reverse">
             <span className="text-4xl font-black tracking-tighter text-neutral-400 group-hover:text-neutral-500 transition-colors">{demoData.valueB}</span>
             <span className="text-xs font-bold text-neutral-400">{demoData.unit}</span>
           </div>
        </div>
      </div>
    );
  };

  const renderDeltaBarVisual = () => {
    const maxVal = Math.max(Number(demoData.valueA), Number(demoData.valueB)) * 1.3; 
    const aPct = Math.min((Number(demoData.valueA) / maxVal) * 100, 100);
    const bPct = Math.min((Number(demoData.valueB) / maxVal) * 100, 100);
    
    const delta = Number(demoData.valueA) - Number(demoData.valueB);
    const isOver = delta > 0;
    // Striped background class
    const striped = "bg-[linear-gradient(45deg,rgba(0,0,0,0.03)_25%,transparent_25%,transparent_50%,rgba(0,0,0,0.03)_50%,rgba(0,0,0,0.03)_75%,transparent_75%,transparent)] bg-[length:10px_10px]";

    return (
      <div className="flex flex-col justify-center h-full w-full gap-5">
         {/* Metrics Header */}
         <div className="flex justify-between items-end">
            <div className="flex flex-col">
              <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-widest mb-1 flex items-center gap-1">
                <Target className="w-3 h-3" /> Target: {demoData.valueB}
              </span>
              <span className="text-4xl font-black text-neutral-100 tabular-nums tracking-tight">
                {demoData.valueA} <span className="text-lg text-neutral-400 font-medium">{demoData.unit}</span>
              </span>
            </div>
            <div className={`flex flex-col items-end`}>
               <div className={`px-2 py-1 rounded bg-neutral-800 border border-neutral-700 flex items-center gap-2`}>
                  <span className="text-[10px] font-bold text-neutral-400 uppercase">Delta</span>
                  <span className={`text-lg font-bold tabular-nums leading-none ${isOver ? 'text-amber-600' : 'text-emerald-600'}`}>
                    {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                  </span>
               </div>
            </div>
         </div>

         {/* The Bar Track */}
         <div className="relative h-14 bg-neutral-800 rounded-lg w-full overflow-hidden shadow-inner ring-1 ring-neutral-700/50">
            {/* Striped Background */}
            <div className={`absolute inset-0 ${striped}`}></div>
            
            {/* Grid Lines */}
            <div className="absolute inset-0 flex justify-between px-4">
              {[0, 20, 40, 60, 80, 100].map(i => <div key={i} className="h-full w-px bg-neutral-600/30 dashed"></div>)}
            </div>

            {/* Target Marker (Triangle + Line) */}
            <div className="absolute top-0 bottom-0 w-0.5 bg-neutral-300 z-30 shadow-[0_0_10px_rgba(255,255,255,0.15)]" style={{ left: `${bPct}%` }}>
               <div className="absolute -top-0 -translate-x-1/2 w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[8px] border-t-neutral-300"></div>
            </div>
            
            {/* Actual Value Bar */}
            <div 
              className={`absolute top-3 bottom-3 roundedr-md shadow-md transition-all duration-700 ease-out flex items-center justify-end pr-2 overflow-hidden border-r border-black/5
                ${isOver ? 'bg-amber-500' : 'bg-emerald-500'}
              `}
              style={{ width: `${aPct}%`, borderRadius: '0 4px 4px 0' }}
            >
              <div className="w-full h-full opacity-20 bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.5),transparent)]"></div>
            </div>
         </div>
      </div>
    );
  };

  const renderGroupedBarVisual = () => {
    const items = demoData.items || [];
    const values = items.map(i => i.valueB || 0);
    const total = values.reduce((a, b) => a + b, 0);
    const average = total / (values.length || 1);
    
    // Imbalance calculation (NEMA definition: max deviation from average)
    let maxDeviation = 0;
    values.forEach(v => {
      const dev = Math.abs(v - average);
      if (dev > maxDeviation) maxDeviation = dev;
    });
    
    const imbalancePct = average > 0 ? (maxDeviation / average) * 100 : 0;
    const isImbalanced = imbalancePct > 3; // Strict industrial standard often 1-5%
    
    // Visual scaling
    const maxVal = Math.max(...values, average * 1.1);
    const minVal = 0; 
    
    // Band calculations (e.g. 5% tolerance)
    const tolerance = 0.05;
    const upperLimit = average * (1 + tolerance);
    const lowerLimit = average * (1 - tolerance);
    
    const getPct = (val: number) => Math.min((val / (maxVal * 1.1)) * 100, 100);

    return (
        <div className="flex flex-col h-full w-full">
            {/* Header section with Stats */}
            <div className="flex justify-between items-end mb-6">
                <div>
                    <span className="text-[10px] uppercase tracking-widest text-neutral-400 font-bold">Total Load</span>
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-black text-neutral-900">{total}</span>
                        <span className="text-sm font-bold text-neutral-400">Amps</span>
                    </div>
                </div>
                
                <div className="text-right">
                    <span className="text-[10px] uppercase tracking-widest text-neutral-400 font-bold">Max Imbalance</span>
                    <div className={`flex items-center justify-end gap-2 ${isImbalanced ? 'text-amber-600' : 'text-emerald-600'}`}>
                        {isImbalanced ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                        <span className="text-xl font-black">{imbalancePct.toFixed(1)}%</span>
                    </div>
                    <span className="text-[9px] text-neutral-400">Limit: 3.0%</span>
                </div>
            </div>

            {/* Chart Area */}
            <div className="relative flex-1 w-full border-b border-neutral-200 min-h-[160px]">
                {/* Safe Zone Band */}
                <div 
                    className="absolute w-full bg-emerald-500/5 border-y border-emerald-500/20 transition-all duration-500"
                    style={{ 
                        bottom: `${getPct(lowerLimit)}%`, 
                        height: `${getPct(upperLimit) - getPct(lowerLimit)}%` 
                    }}
                >
                     <div className="absolute right-0 top-0 -mt-2.5 px-1 bg-white/50 backdrop-blur-sm text-[9px] text-emerald-600 font-bold rounded">
                        Target Avg
                     </div>
                </div>
                
                {/* Average Line */}
                <div 
                    className="absolute w-full border-t border-dashed border-emerald-500/40"
                    style={{ bottom: `${getPct(average)}%` }}
                />

                {/* Bars */}
                <div className="absolute inset-0 flex items-end justify-around px-4">
                    {items.map((item, i) => {
                        const val = item.valueB || 0;
                        const pct = getPct(val);
                        const dev = Math.abs(val - average);
                        const devPct = average > 0 ? (dev / average) * 100 : 0;
                        const isWarning = devPct > 3; // Individual phase warning
                        
                        return (
                            <div key={i} className="flex flex-col items-center justify-end h-full w-16 group relative">
                                {/* Value Label */}
                                <div className={`mb-2 text-sm font-bold transition-all ${isWarning ? 'text-amber-600' : 'text-neutral-700'} group-hover:-translate-y-1`}>
                                    {val}
                                </div>
                                
                                {/* Bar */}
                                <div 
                                    className={`w-full rounded-t-md transition-all duration-500 relative overflow-hidden
                                        ${isWarning ? 'bg-amber-500 shadow-[0_0_15px_rgba(245,158,11,0.3)]' : 'bg-neutral-800'}
                                    `}
                                    style={{ height: `${pct}%` }}
                                >
                                    {/* Glass reflection effect */}
                                    <div className="absolute top-0 left-0 right-0 h-full bg-gradient-to-tr from-white/10 to-transparent pointer-events-none" />
                                </div>
                                
                                {/* Axis Label */}
                                <div className="mt-3">
                                    <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-neutral-100 text-[10px] font-bold text-neutral-600 border border-neutral-200">
                                        {item.label}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
            
             {/* Legend/Context */}
             <div className="mt-4 flex items-center justify-center gap-6 opacity-80">
                <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-neutral-800"></div>
                    <span className="text-[10px] text-neutral-500 font-medium">Nominal</span>
                </div>
                 <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-amber-500"></div>
                    <span className="text-[10px] text-neutral-500 font-medium">Deviation {'>'} 3%</span>
                </div>
             </div>
        </div>
    );
  };

  const renderWaterfallVisual = () => {
    const items = demoData.items || [];
    // Start value (Rated)
    const startVal = items.find(i => i.valueA !== undefined)?.valueA || 100;
    
    // Calculate running totals for precise rendering to determine chart scale
    let runningTotal = startVal;
    let minCalcVal = startVal;
    let maxCalcVal = startVal;

    const chartItems = items.map(item => {
        if (item.valueA !== undefined && !item.diff) {
            // It's a total bar (Start or End)
            const val = item.valueA || 0;
            minCalcVal = Math.min(minCalcVal, val);
            maxCalcVal = Math.max(maxCalcVal, val);
            return { ...item, type: 'total', val };
        }
        
        // It's a diff bar (loss or gain)
        const diff = item.diff || 0;
        const absDiff = Math.abs(diff);
        
        // Waterfall logic for loss (negative diff): Top aligned with previous total.
        const currentTop = runningTotal;
        runningTotal += diff; 
        
        minCalcVal = Math.min(minCalcVal, runningTotal);
        maxCalcVal = Math.max(maxCalcVal, runningTotal); // Update max if gain

        return { 
            ...item, 
            type: 'step', 
            isLoss: diff < 0,
            bottom: diff < 0 ? runningTotal : (runningTotal - diff),
            height: absDiff,
            val: diff
        };
    });

    // Dynamic scale zoom to make small losses visible
    // Scale from [minVal - padding, 100 or maxVal]
    const minVal = Math.floor(Math.max(0, minCalcVal - 2) / 5) * 5; 
    const maxVal = Math.max(100, maxCalcVal + 1); // Ensure 100 is typically visible for efficiency
    const range = maxVal - minVal;

    // Calculate Biggest Loser for attribution
    const losses = chartItems.filter(i => i.type === 'step' && i.isLoss);
    const totalLoss = losses.reduce((acc, i) => acc + (i.height || 0), 0);
    let maxLossVal = 0;
    let maxLossIdx = -1;
    chartItems.forEach((item, idx) => {
        if (item.type === 'step' && item.isLoss && (item.height || 0) > maxLossVal) {
            maxLossVal = item.height || 0;
            maxLossIdx = idx;
        }
    });

    const biggestLoserLabel = maxLossIdx !== -1 ? chartItems[maxLossIdx].label : "N/A";
    const biggestLoserShare = totalLoss > 0 ? ((maxLossVal / totalLoss) * 100).toFixed(0) : "0";

    return (
        <div className="h-full w-full flex flex-col px-2 py-1 min-h-[300px]">
             {/* Header */}
             <div className="flex justify-between items-start mb-6">
                <div>
                   <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-widest block mb-1">Efficiency Loss Analysis</span>
                   <div className="flex items-baseline gap-2">
                     <span className="text-xl font-black text-neutral-100">{startVal}%</span>
                     <MoveRight className="w-4 h-4 text-neutral-400" />
                     <span className={`text-xl font-black ${(demoData.valueB || 0) < 95 ? 'text-amber-600' : 'text-blue-600'}`}>{demoData.valueB}%</span>
                   </div>
                </div>
                <div className="text-right">
                   <div className="flex items-center justify-end gap-1.5 mb-1">
                       <span className="text-[10px] text-neutral-400 uppercase tracking-wide font-bold">Total Drop</span>
                       <span className="text-sm font-bold text-red-600">{(demoData.valueB - startVal).toFixed(1)} pp</span>
                   </div>
                   <div className="bg-neutral-800 px-2 py-1 rounded text-[10px] font-medium text-neutral-400 inline-block">
                      Metric: Efficiency (%)
                   </div>
                </div>
             </div>

             {/* Chart */}
             <div className="flex-1 relative border-b border-neutral-700 mt-2 min-h-[160px] overflow-hidden">
                 {/* Benchmark Line (95%) - Only show if within view range */}
                 {95 >= minVal && (
                    <div className="absolute left-0 right-0 border-t border-dashed border-green-500/50 z-0 flex items-end justify-start" 
                         style={{ bottom: `${((95 - minVal) / range) * 100}%` }}>
                        <span className="text-[9px] font-bold text-green-400 bg-neutral-900/90 px-1 -mt-3.5 backdrop-blur-sm shadow-sm border border-green-800 rounded">
                            Expected ≥ 95%
                        </span>
                    </div>
                 )}

                 {/* Grid Lines */}
                 {[80, 85, 90, 95, 100].filter(t => t > minVal && t <= maxVal).map(tick => (
                     <div key={tick} className="absolute left-0 right-0 border-t border-neutral-700 z-0" style={{ bottom: `${((tick - minVal) / range) * 100}%` }}></div>
                 ))}

                 {/* Bars Container */}
                 <div className="absolute inset-0 flex items-end justify-between gap-3 z-10 px-2 pb-6">
                     {chartItems.map((item, idx) => {
                         const isTotal = item.type === 'total';
                         const isBiggestLoser = idx === maxLossIdx;
                         
                         let drawBottom = 0;
                         let drawHeight = 0;

                         if (isTotal) {
                             const val = item.val || 0;
                             // For total bars in a zoomed view, we render from the minVal (chart floor) up to the value.
                             drawBottom = minVal;
                             drawHeight = val - minVal;
                         } else {
                             drawBottom = item.bottom || 0;
                             drawHeight = item.height || 0;
                         }

                         const barBottomPct = (drawBottom - minVal) / range * 100;
                         const barHeightPct = drawHeight / range * 100;
                         
                         return (
                             <div key={idx} className="flex-1 relative h-full group">
                                 {/* The Bar */}
                                 <div 
                                     className={`absolute w-full rounded-sm border shadow-sm transition-all duration-300
                                        ${isTotal
                                            ? idx === 0 ? 'bg-neutral-500 border-neutral-600' : 'bg-blue-500 border-blue-600'
                                            : isBiggestLoser
                                                ? 'bg-red-500 border-red-600 shadow-[0_4px_6px_-1px_rgba(220,38,38,0.2)]'
                                                : 'bg-rose-500 border-rose-600'
                                        }
                                     `}
                                     style={{ 
                                         height: `${Math.max(barHeightPct, 1)}%`, // ensure at least 1% visual height
                                         bottom: `${Math.max(barBottomPct, 0)}%` 
                                     }}
                                 >
                                    {/* Value Label */}
                                    <div className={`absolute left-1/2 -translate-x-1/2 text-[10px] font-bold whitespace-nowrap
                                        ${isTotal ? '-top-5 text-neutral-300' : 'top-1/2 -translate-y-1/2 text-white drop-shadow-md'}
                                    `}>
                                        {isTotal ? `${item.val}%` : item.val}
                                    </div>
                                 </div>

                                 {/* Connector Line (to left) */}
                                 {idx > 0 && !isTotal && (
                                     <div 
                                        className="absolute -left-[50%] w-[100%] border-t border-dashed border-neutral-600 pointer-events-none opacity-50"
                                        style={{ bottom: `${((item.bottom! + item.height!) - minVal) / range * 100}%` }}
                                     ></div>
                                 )}

                                 {/* Axis Label */}
                                 <div className="absolute -bottom-6 left-0 right-0 text-center flex flex-col items-center">
                                     <span className={`text-[9px] font-bold uppercase tracking-tight block truncate w-full
                                        ${isBiggestLoser ? 'text-red-400' : 'text-neutral-400'}
                                     `}>{item.label}</span>
                                     {isBiggestLoser && (
                                         <span className="text-[8px] bg-red-900/50 text-red-400 px-1 rounded-full font-bold mt-0.5 whitespace-nowrap">
                                             {biggestLoserShare}% of loss
                                         </span>
                                     )}
                                 </div>
                             </div>
                         )
                     })}
                 </div>
             </div>

             {/* Footer Causal Summary */}
             <div className="mt-8 flex items-start gap-2 bg-neutral-800/50 p-2 rounded border border-neutral-700">
                 <div className="mt-0.5">
                     <Info className="w-3.5 h-3.5 text-blue-400" />
                 </div>
                 <div className="flex-1">
                     <p className="text-[10px] text-neutral-400 leading-snug">
                         <span className="font-bold text-neutral-200">Causal Analysis:</span> Primary efficiency driver is <span className="font-bold text-red-400 lowercase">{biggestLoserLabel}</span>, accounting for {biggestLoserShare}% of total loss.
                         {Number(demoData.valueB) < 95 && " System is operating below 95% benchmark."}
                     </p>
                 </div>
             </div>
        </div>
    );
  };

  const renderSmallMultiplesVisual = () => (
     <div className="grid grid-cols-2 gap-x-6 gap-y-4 w-full h-full">
       {demoData.items?.map((item, idx) => {
         const diff = item.diff || 0;
         const isPos = diff > 0;
         return (
         <div key={idx} className="flex flex-col justify-center p-3 rounded-lg border border-neutral-100 bg-neutral-50/50 hover:bg-white hover:shadow-md transition-all duration-200 group">
           <div className="flex justify-between items-start mb-2">
              <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-wide group-hover:text-neutral-900 transition-colors">{item.label}</span>
              <div className={`flex items-center text-[10px] font-black ${isPos ? 'text-rose-500' : 'text-emerald-500'} bg-white px-1.5 py-0.5 rounded shadow-sm border border-neutral-100`}>
                {isPos ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                {diff > 0 ? '+' : ''}{diff}
              </div>
           </div>
           
           <div className="h-12 w-full flex items-end gap-1.5 mt-auto">
              {/* Comparison Bars */}
              <div className="flex-1 flex flex-col gap-1 h-full justify-end">
                <div className="w-full bg-neutral-300 rounded-sm relative" style={{ height: `${Math.min((item.valueA || 0), 100)}%` }}></div>
              </div>
              <div className="flex-1 flex flex-col gap-1 h-full justify-end">
                <div className="w-full bg-blue-600 rounded-sm relative shadow-sm" style={{ height: `${Math.min((item.valueB || 0), 100)}%` }}></div>
              </div>
           </div>
         </div>
       )})}
     </div>
  );

  const renderCompositionSplitVisual = () => (
    <div className="w-full flex flex-col justify-center h-full gap-5">
       {[demoData.labelA, demoData.labelB].map((lbl, idx) => (
         <div key={idx} className="flex flex-col gap-2">
           <div className="flex justify-between items-end px-1">
              <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-widest">{lbl}</span>
           </div>
           
           <div className="h-12 w-full flex rounded-lg overflow-hidden ring-1 ring-neutral-200 shadow-sm bg-neutral-100 p-[2px] gap-[1px]">
             {demoData.composition?.map((part, pIdx) => {
               const val = idx === 0 ? part.valueA : part.valueB;
               return (
               <div 
                 key={pIdx} 
                 className={`${part.color} relative group flex items-center justify-center first:rounded-l-md last:rounded-r-md hover:brightness-110 transition-all cursor-crosshair`} 
                 style={{ width: `${val}%` }}
               >
                 {val > 8 && (
                   <span className="text-[10px] font-bold text-white shadow-sm drop-shadow-md">
                     {val}%
                   </span>
                 )}
               </div>
             )})}
           </div>
         </div>
       ))}

       <div className="flex flex-wrap gap-4 justify-center mt-2">
          {demoData.composition?.map((part, i) => (
            <div key={i} className="flex items-center gap-2 bg-white px-2 py-1 rounded-md border border-neutral-100 shadow-sm">
               <div className={`w-2 h-2 rounded-full ${part.color}`}></div>
               <span className="text-[10px] font-bold text-neutral-600 uppercase tracking-wide">{part.label}</span>
            </div>
          ))}
       </div>
    </div>
  );

  return (
    <div className={`
      ${layout.padding} 
      ${visual.background} 
      ${visual.border}
      w-full h-full min-h-[160px] flex flex-col justify-center relative
    `}>
      {variant === 'SIDE_BY_SIDE_VISUAL' && renderSideBySideVisual()}
      {variant === 'DELTA_BAR_VISUAL' && renderDeltaBarVisual()}
      {variant === 'GROUPED_BAR_VISUAL' && renderGroupedBarVisual()}
      {variant === 'WATERFALL_VISUAL' && renderWaterfallVisual()}
      {variant === 'SMALL_MULTIPLES_VISUAL' && renderSmallMultiplesVisual()}
      {variant === 'COMPOSITION_SPLIT_VISUAL' && renderCompositionSplitVisual()}
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
  return <ComparisonRenderer spec={data} />;
}
