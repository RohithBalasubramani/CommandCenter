// @ts-nocheck

const WidgetVariant = {
  VERTICAL: 'CATEGORY_BAR_VERTICAL',
  HORIZONTAL: 'CATEGORY_BAR_HORIZONTAL',
  STACKED: 'CATEGORY_BAR_STACKED',
  STACKED_PERCENT: 'CATEGORY_BAR_STACKED_PERCENT',
  GROUPED: 'CATEGORY_BAR_GROUPED',
  DIVERGING: 'CATEGORY_BAR_DIVERGING',
  DENSE: 'CATEGORY_BAR_DENSE'
};

const COLORS = {
  neutral: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#e5e5e5',
    300: '#d4d4d4',
    400: '#a3a3a3',
    500: '#737373',
    600: '#525252',
    700: '#404040',
    800: '#262626',
    900: '#171717',
  },
  primary: '#2563eb',
  success: '#16a34a',
  warning: '#d97706',
  danger: '#ef4444',
};

const TYPO = {
  label: "text-[10px] uppercase font-bold tracking-widest text-neutral-400",
  sectionHeader: "text-xs uppercase font-bold tracking-widest text-neutral-500",
  h1: "text-2xl font-bold tracking-tight text-neutral-900",
  heroValue: "text-[3rem] font-bold leading-none tracking-tighter text-neutral-900",
  cardValue: "text-3xl font-bold tracking-tight text-neutral-900",
  mono: "font-mono text-xs text-neutral-700",
};


import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Brush
} from 'recharts';


interface Props {
  data: DataPoint[];
  config: ChartConfig;
  height?: number;
  enableBrush?: boolean;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-3 border border-neutral-200 shadow-lg rounded-lg z-50">
        <p className={TYPO.label}>{label}</p>
        <div className="mt-1 space-y-1">
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center gap-2">
              <div 
                className="w-2 h-2 rounded-full" 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm font-medium text-neutral-900">
                {entry.name}: <span className="font-mono">{entry.value}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

const CategoryBarChart: React.FC<Props> = ({ data, config, height = 320, enableBrush = false }) => {
  const isHorizontal = config.layout === 'vertical';
  
  // Transform data for STACKED_PERCENT
  const processedData = useMemo(() => {
    if (config.variant !== WidgetVariant.STACKED_PERCENT) return data;
    
    return data.map(item => {
      const total = (item.value || 0) + (item.value2 || 0) + (item.value3 || 0);
      if (total === 0) return item;
      return {
        ...item,
        value: Number(((item.value || 0) / total * 100).toFixed(1)),
        value2: Number(((item.value2 || 0) / total * 100).toFixed(1)),
        value3: Number(((item.value3 || 0) / total * 100).toFixed(1)),
      };
    });
  }, [data, config.variant]);

  // Dynamic Bar Rendering Logic
  const renderBars = () => {
    if (config.variant === WidgetVariant.DIVERGING) {
      return (
        <Bar dataKey="value" fill={COLORS.neutral[800]} radius={[2, 2, 0, 0]}>
          {processedData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.value < 0 ? COLORS.danger : COLORS.neutral[800]} 
            />
          ))}
        </Bar>
      );
    }

    if (config.variant === WidgetVariant.DENSE) {
       return (
        <Bar dataKey="value" fill={COLORS.neutral[800]} radius={[1, 1, 0, 0]} />
       )
    }

    return config.dataKeys.map((key, index) => (
      <Bar
        key={key}
        dataKey={key}
        stackId={config.stacked ? 'a' : undefined}
        fill={config.colors[index % config.colors.length]}
        radius={config.stacked ? [0, 0, 0, 0] : (isHorizontal ? [0, 4, 4, 0] : [4, 4, 0, 0])}
        barSize={config.variant === WidgetVariant.DENSE ? 4 : undefined}
      />
    ));
  };

  return (
    <div className="w-full h-full relative">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout={config.layout || 'horizontal'}
          data={processedData}
          margin={{ top: 20, right: 30, left: 20, bottom: enableBrush ? 40 : 20 }}
          barCategoryGap={config.variant === WidgetVariant.DENSE ? '10%' : '20%'}
        >
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke={COLORS.neutral[100]} 
            horizontal={!isHorizontal} 
            vertical={isHorizontal}
          />
          
          <XAxis 
            type={isHorizontal ? 'number' : 'category'}
            dataKey={isHorizontal ? undefined : 'category'}
            tick={{ fontSize: 10, fill: COLORS.neutral[400], fontWeight: 500 }}
            tickLine={false}
            axisLine={{ stroke: COLORS.neutral[200] }}
            interval={config.variant === WidgetVariant.DENSE ? 4 : 0}
          />
          
          <YAxis 
            type={isHorizontal ? 'category' : 'number'}
            dataKey={isHorizontal ? 'category' : undefined}
            tick={{ fontSize: 10, fill: COLORS.neutral[400], fontWeight: 500 }}
            tickLine={false}
            axisLine={{ stroke: COLORS.neutral[200] }}
            width={50}
          />
          
          <Tooltip content={<CustomTooltip />} cursor={{ fill: COLORS.neutral[50] }} />
          
          {config.showLegend !== false && config.dataKeys.length > 1 && (
            <Legend 
              verticalAlign="top" 
              height={36} 
              iconSize={8}
              iconType="circle"
              wrapperStyle={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.05em', color: COLORS.neutral[500] }}
            />
          )}

          {config.variant === WidgetVariant.DIVERGING && (
             <ReferenceLine y={0} stroke={COLORS.neutral[300]} />
          )}

          {renderBars()}
          
          {enableBrush && (
            <Brush 
              dataKey="category"
              height={24}
              stroke={COLORS.neutral[200]}
              fill={COLORS.neutral[50]}
              tickFormatter={() => ''}
              travellerWidth={8}
            />
          )}

        </BarChart>
      </ResponsiveContainer>
      
      {/* Axis Labels (Custom Overlay) */}
      <div className={`absolute bottom-0 left-1/2 transform -translate-x-1/2 ${enableBrush ? 'translate-y-8' : 'translate-y-4'}`}>
        <span className={TYPO.label}>{config.xAxisLabel}</span>
      </div>
       <div className="absolute top-1/2 left-0 transform -translate-x-6 -translate-y-1/2 -rotate-90">
        <span className={TYPO.label}>{config.yAxisLabel}</span>
      </div>
    </div>
  );
};



export default function ScenarioComponent({ data }) {
  // Default config with all required fields
  const defaultConfig = {
    variant: 'CATEGORY_BAR_VERTICAL',
    title: 'Category Bar Chart',
    dataKeys: ['value'],
    colors: ['#262626'],
    layout: 'horizontal'
  };
  const config = data && data.config ? { ...defaultConfig, ...data.config } : defaultConfig;
  // Ensure dataKeys is always an array
  if (!config.dataKeys || !Array.isArray(config.dataKeys)) {
    config.dataKeys = ['value'];
  }
  const chartData = data && data.data && Array.isArray(data.data) ? data.data : [];
  const height = data && data.height ? data.height : 320;
  const enableBrush = data && data.enableBrush ? data.enableBrush : false;

  if (chartData.length === 0) {
    return (
      <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#991b1b', textAlign: 'center' }}>
        <strong>No chart data available</strong>
        <p style={{ fontSize: 12, marginTop: 8 }}>Config parsing may have failed</p>
      </div>
    );
  }

  return (
    <CategoryBarChart
      data={chartData}
      config={config}
      height={height}
      enableBrush={enableBrush}
    />
  );
}
