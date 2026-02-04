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
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend,
  ComposedChart,
  Line,
  CartesianGrid
} from 'recharts';

interface UniversalChartProps {
  spec: any;
}

const CustomTooltip = ({ active, payload, label, unit }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-neutral-900 text-white p-3 rounded-lg shadow-xl border border-neutral-700 z-50">
        <div className="text-xs font-bold text-neutral-400 mb-1 uppercase tracking-wider">{label || payload[0].name}</div>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
            <span className="font-mono text-sm">
              {entry.value} {unit}
            </span>
            {entry.payload.percentage && (
               <span className="text-xs text-neutral-500 ml-1">({entry.payload.percentage}%)</span>
            )}
          </div>
        ))}
      </div>
    );
  }
  return null;
};

const UniversalChart: React.FC<UniversalChartProps> = ({ spec }) => {
  const { representation, visual, demoData } = spec;
  const isPieOrDonut = representation === 'Pie' || representation === 'Donut';
  const isHorizontal = representation === 'Horizontal Bar';
  const isStacked = representation.includes('Stacked');
  const isPareto = representation === 'Pareto Bar';
  const isGrouped = representation === 'Grouped Bar';

  if (isPieOrDonut) {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={demoData.series}
            cx="50%"
            cy="50%"
            innerRadius={visual.innerRadius || 0}
            outerRadius={visual.outerRadius || "80%"}
            paddingAngle={2}
            dataKey="value"
          >
            {demoData.series.map((entry: any, index: number) => (
              <Cell key={`cell-${index}`} fill={entry.color} strokeWidth={0} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip unit={demoData.unit} />} />
          <Legend 
            verticalAlign="bottom" 
            height={36} 
            iconType="circle"
            formatter={(value) => <span className="text-xs font-mono text-neutral-500 ml-1">{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (isHorizontal) {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={demoData.series}
          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid horizontal={false} stroke="#f5f5f5" />
          <XAxis type="number" hide />
          <YAxis 
            dataKey="label" 
            type="category" 
            tick={{ fontSize: 10, fill: '#737373', fontFamily: 'monospace' }} 
            width={80}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip unit={demoData.unit} />} cursor={{fill: '#f5f5f5'}} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={visual.barSize || 20}>
             {demoData.series.map((entry: any, index: number) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (isPareto) {
     return (
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={demoData.series} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid vertical={false} stroke="#f5f5f5" />
          <XAxis 
            dataKey="label" 
            tick={{ fontSize: 10, fill: '#737373', fontFamily: 'monospace' }} 
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            yAxisId="left"
            tick={{ fontSize: 10, fill: '#737373', fontFamily: 'monospace' }} 
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            unit="%"
            tick={{ fontSize: 10, fill: '#737373', fontFamily: 'monospace' }} 
            tickLine={false}
            axisLine={false}
          />
          <Tooltip content={<CustomTooltip unit={demoData.unit} />} />
          <Bar yAxisId="left" dataKey="value" barSize={30} fill="#171717" radius={[4, 4, 0, 0]} >
            {demoData.series.map((entry: any, index: number) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
          <Line yAxisId="right" type="monotone" dataKey="cumulative" stroke="#2563eb" strokeWidth={2} dot={{r: 3}} />
        </ComposedChart>
      </ResponsiveContainer>
     );
  }

  if (isGrouped) {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={demoData.categories.map((cat: string, i: number) => ({
                name: cat,
                ...demoData.series.reduce((acc: any, s: any) => ({...acc, [s.name]: s.data[i]}), {})
            }))}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid vertical={false} stroke="#f5f5f5" />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#737373' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: '#737373' }} tickLine={false} axisLine={false} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
            {demoData.series.map((s: any, i: number) => (
                <Bar key={i} dataKey={s.name} fill={s.color} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      )
  }

  // Default to Simple Bar or 100% Stacked (simplified as horizontal bar for this demo)
  if (representation === '100% Stacked Bar') {
      // Simplified simulation of 100% stacked bar using a single stacked bar chart layout
       return (
        <div className="w-full h-12 flex rounded-md overflow-hidden my-auto">
            {demoData.series.map((item: any, i: number) => (
                <div 
                    key={i} 
                    style={{ width: `${item.value}%`, backgroundColor: item.color }} 
                    className="h-full relative group"
                >
                    <div className="opacity-0 group-hover:opacity-100 absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-neutral-900 text-white text-xs p-1 rounded whitespace-nowrap z-10 pointer-events-none">
                        {item.label}: {item.value}%
                    </div>
                </div>
            ))}
        </div>
       )
  }

  return <div>Unknown Chart Type</div>;
};


export default function ScenarioComponent({ data }) {
  if (!data) {
    return (
      <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#991b1b' }}>
        <strong>Missing spec data</strong>
      </div>
    );
  }
  return <UniversalChart spec={data} />;
}
