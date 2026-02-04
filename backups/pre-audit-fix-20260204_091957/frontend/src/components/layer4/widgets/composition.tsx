// @ts-nocheck
import React from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  Treemap,
  XAxis,
  YAxis
} from 'recharts';


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

// Composition chart spec data
const DONUT_PIE_SPEC = {
  demoData: [
    { name: 'Running', value: 65 },
    { name: 'Idle', value: 20 },
    { name: 'Maintenance', value: 10 },
    { name: 'Fault', value: 5 }
  ],
  visual: { palette: ['#16a34a', '#a3a3a3', '#f59e0b', '#ef4444'] }
};

const STACKED_BAR_SPEC = {
  demoData: [
    { name: 'Shift 1', Solar: 4000, Grid: 2400, Diesel: 2400 },
    { name: 'Shift 2', Solar: 3000, Grid: 1398, Diesel: 2210 },
    { name: 'Shift 3', Solar: 2000, Grid: 9800, Diesel: 2290 }
  ],
  visual: { palette: ['#16a34a', '#2563eb', '#f59e0b'] }
};

const STACKED_AREA_SPEC = {
  demoData: [
    { time: '08:00', Running: 65, Idle: 20, Down: 15 },
    { time: '09:00', Running: 70, Idle: 25, Down: 5 },
    { time: '10:00', Running: 60, Idle: 30, Down: 10 },
    { time: '11:00', Running: 75, Idle: 15, Down: 10 },
    { time: '12:00', Running: 80, Idle: 10, Down: 10 }
  ],
  visual: { palette: ['#16a34a', '#a3a3a3', '#ef4444'] }
};

const TREEMAP_SPEC = {
  demoData: [
    { name: 'Machinery', size: 450, category: 'Production' },
    { name: 'HVAC', size: 280, category: 'Facilities' },
    { name: 'Lighting', size: 120, category: 'Facilities' },
    { name: 'Compressors', size: 200, category: 'Production' },
    { name: 'IT Systems', size: 80, category: 'Support' }
  ],
  visual: { palette: ['#171717', '#404040', '#525252', '#737373', '#a3a3a3'] }
};

const WATERFALL_SPEC = {
  demoData: [
    { name: 'Baseline', value: 1000, fill: '#737373' },
    { name: 'Efficiency', value: -150, fill: '#16a34a' },
    { name: 'Load Growth', value: 200, fill: '#ef4444' },
    { name: 'Solar', value: -80, fill: '#16a34a' },
    { name: 'Final', value: 970, fill: '#2563eb' }
  ],
  visual: { palette: ['#737373', '#16a34a', '#ef4444', '#16a34a', '#2563eb'] }
};



const DonutPreview: React.FC = () => {
  const data = DONUT_PIE_SPEC.demoData;
  const palette = DONUT_PIE_SPEC.visual.palette;

  // Calculate total for center label
  const total = data.reduce((acc, curr) => acc + curr.value, 0);

  return (
    <div className="w-full h-full relative">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="text-center">
            <div className="text-2xl font-bold tracking-tight text-neutral-900">{total}</div>
            <div className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">Total Assets</div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="45%"
            innerRadius="45%"
            outerRadius="65%"
            paddingAngle={5}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={palette[index % palette.length]} />
            ))}
          </Pie>
          <Tooltip
             contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e5e5',
              borderRadius: '8px',
              fontSize: '12px',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            }}
            formatter={(value: number) => [`${value} Units`, 'Count']}
          />
          <Legend
            verticalAlign="bottom"
            height={24}
            iconType="circle"
            wrapperStyle={{ fontSize: '11px', color: '#737373', paddingTop: '8px' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};




const StackedAreaPreview: React.FC = () => {
  const data = STACKED_AREA_SPEC.demoData;
  const palette = STACKED_AREA_SPEC.visual.palette;

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f5f5f5" />
          <XAxis 
            dataKey="time" 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false}
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false} 
          />
          <Tooltip 
             contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e5e5', 
              borderRadius: '8px',
              fontSize: '12px',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            }}
          />
          <Legend 
             verticalAlign="top" 
             align="right" 
             wrapperStyle={{ fontSize: '12px', fontWeight: 500, paddingBottom: '20px', color: '#737373' }} 
             iconType="circle"
          />
          <Area type="monotone" dataKey="Running" stackId="1" stroke={palette[0]} fill={palette[0]} fillOpacity={0.9} />
          <Area type="monotone" dataKey="Idle" stackId="1" stroke={palette[1]} fill={palette[1]} fillOpacity={0.9} />
          <Area type="monotone" dataKey="Down" stackId="1" stroke={palette[2]} fill={palette[2]} fillOpacity={0.9} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};




const StackedBarPreview: React.FC = () => {
  const data = STACKED_BAR_SPEC.demoData;
  const palette = STACKED_BAR_SPEC.visual.palette;

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f5f5f5" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false}
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false} 
          />
          <Tooltip 
            cursor={{ fill: 'rgba(0,0,0,0.02)' }}
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e5e5', 
              borderRadius: '8px',
              fontSize: '12px',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
            }}
          />
          <Legend 
            verticalAlign="top" 
            align="right" 
            wrapperStyle={{ fontSize: '12px', fontWeight: 500, paddingBottom: '20px', color: '#737373' }} 
            iconType="circle"
          />
          <Bar dataKey="Solar" stackId="a" fill={palette[0]} barSize={40} />
          <Bar dataKey="Grid" stackId="a" fill={palette[1]} barSize={40} />
          <Bar dataKey="Diesel" stackId="a" fill={palette[2]} barSize={40} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};




// Custom content renderer for Treemap cells to match Industrial look
const CustomContent = (props: any) => {
  const { root, depth, x, y, width, height, index, name, value, colors } = props;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: colors[index % colors.length],
          stroke: '#fff',
          strokeWidth: 2,
        }}
      />
      {width > 50 && height > 30 && (
        <>
          <text
            x={x + 8}
            y={y + 18}
            textAnchor="start"
            fill="#fff"
            fontSize={10}
            fontWeight="bold"
            className="uppercase tracking-wide"
          >
            {name}
          </text>
           <text
            x={x + 8}
            y={y + 34}
            textAnchor="start"
            fill="rgba(255,255,255,0.8)"
            fontSize={12}
            fontFamily="monospace"
          >
            {value}
          </text>
        </>
      )}
    </g>
  );
};

const TreemapPreview: React.FC = () => {
  const data = TREEMAP_SPEC.demoData;
  const palette = TREEMAP_SPEC.visual.palette;

  // Treemap expects a root object with children
  const treeData = [
    {
      name: 'root',
      children: data
    }
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-100 shadow-lg rounded-lg">
          <p className="text-xs font-bold text-neutral-900 mb-1">{data.name}</p>
          <div className="flex items-center space-x-2">
             <p className="text-xs text-neutral-500">
               Value: <span className="font-mono text-neutral-900 font-medium">{data.value}</span>
             </p>
          </div>
           <p className="text-[10px] text-neutral-400 mt-1 uppercase tracking-wider">
             Category: {data.category}
           </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <Treemap
          data={treeData}
          dataKey="size"
          aspectRatio={4 / 3}
          stroke="#fff"
          fill="#171717"
          content={<CustomContent colors={palette} />}
        >
          <Tooltip content={<CustomTooltip />} />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
};




const WaterfallPreview: React.FC = () => {
  const rawData = WATERFALL_SPEC.demoData;
  const palette = WATERFALL_SPEC.visual.palette;

  // Process data for waterfall visual
  // We need to calculate start and end values for floating bars
  let currentTotal = 0;
  const processedData = rawData.map((item) => {
    const prevTotal = currentTotal;
    
    // Logic for displaying the bar correctly
    let uv = 0; // The bottom of the floating bar (transparent stack)
    let pv = 0; // The height of the actual bar
    let color = palette[0]; // default black
    
    if (item.type === 'start' || item.type === 'end') {
      currentTotal = item.value;
      uv = 0;
      pv = item.value;
      color = item.type === 'end' ? palette[3] : palette[0];
    } else {
      // Impact (positive or negative)
      currentTotal += item.value;
      if (item.value >= 0) {
        uv = prevTotal;
        pv = item.value;
        color = palette[1]; // Green
      } else {
        uv = currentTotal;
        pv = Math.abs(item.value);
        color = palette[2]; // Red
      }
    }

    return {
      name: item.name,
      base: uv, // transparent placeholder
      value: pv, // actual bar height
      color: color,
      actualValue: item.value, // for tooltip
      displayTotal: item.type === 'end' ? item.value : currentTotal
    };
  });

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[1].payload; // Access the main data payload
      return (
        <div className="bg-white p-3 border border-gray-100 shadow-lg rounded-lg">
          <p className="text-xs font-bold text-neutral-900 mb-1">{label}</p>
          <div className="flex items-center space-x-2">
             <div className="w-2 h-2 rounded-full" style={{ backgroundColor: data.color }}></div>
             <p className="text-xs text-neutral-500">
               Value: <span className="font-mono text-neutral-900 font-medium">{data.actualValue}</span>
             </p>
          </div>
           <p className="text-[10px] text-neutral-400 mt-1">
             Running Total: {data.displayTotal}
           </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={processedData}
          margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f5f5f5" />
          <XAxis 
            dataKey="name" 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false}
            interval={0}
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 10, fill: '#a3a3a3' }} 
            axisLine={false} 
            tickLine={false} 
          />
          <Tooltip content={<CustomTooltip />} cursor={{fill: 'transparent'}} />
          <ReferenceLine y={0} stroke="#e5e5e5" />
          
          {/* Transparent Base Stack */}
          <Bar dataKey="base" stackId="a" fill="transparent" />
          
          {/* Actual Value Stack */}
          <Bar dataKey="value" stackId="a" radius={[2, 2, 2, 2]}>
            {processedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};




const CompositionPreview = ({ data }) => {
  const variant = data && data.variant ? data.variant : data && data.spec ? data.spec.variant : data ? data.id : undefined;
  switch (variant) {
    case "STACKED_AREA":
      return <StackedAreaPreview />;
    case "STACKED_BAR":
      return <StackedBarPreview />;
    case "TREEMAP":
      return <TreemapPreview />;
    case "WATERFALL":
      return <WaterfallPreview />;
    default:
      return <DonutPreview />;
  }
};

export default CompositionPreview;
