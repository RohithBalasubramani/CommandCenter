// @ts-nocheck
import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Box,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Circle,
  Clock,
  Cpu,
  FileText,
  Filter,
  History,
  Info,
  Layers,
  Search,
  Shield,
  ShieldAlert,
  Terminal,
  User,
  Wrench,
  X
} from 'lucide-react';
import {
  Area,
  AreaChart,
  ResponsiveContainer,
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

// EventLogStream enums
const WidgetVariation = {
  CHRONOLOGICAL_TIMELINE: 'Chronological Timeline',
  COMPACT_CARD_FEED: 'Compact Card Feed',
  TABULAR_LOG_VIEW: 'Tabular Log View',
  CORRELATION_STACK: 'Correlation Stack',
  GROUPED_BY_ASSET: 'Grouped by Asset',
};

const WidgetRepresentation = {
  TIMELINE_RAIL: 'Severity Rail',
  LIST_CARD: 'Card Feed',
  DATA_TABLE: 'Tabular View',
  GROUP_HEADER: 'Group Header',
};

const EventSeverity = {
  INFO: 'INFO',
  WARNING: 'WARNING',
  CRITICAL: 'CRITICAL',
};

const EventType = {
  ALARM: 'ALARM',
  FAULT: 'FAULT',
  STATUS: 'STATUS',
  ACTION: 'ACTION',
  COMMENT: 'COMMENT',
  OVERRIDE: 'OVERRIDE',
  INSPECTION: 'INSPECTION',
};

const EventStatus = {
  OPEN: 'OPEN',
  ACKED: 'ACKED',
  RESOLVED: 'RESOLVED',
};

const generateTimestamp = (minutesBack) => {
  const d = new Date();
  d.setMinutes(d.getMinutes() - minutesBack);
  return d.toISOString();
};

const MOCK_EVENTS = [
  {
    id: 'EVT-1024',
    timestamp: generateTimestamp(2),
    source: 'DG-Set-A',
    assetPath: 'Plant 1 > Power House > DG A',
    severity: EventSeverity.CRITICAL,
    type: EventType.ALARM,
    title: 'High Coolant Temp',
    shortMessage: 'Coolant temperature exceeded threshold (98°C).',
    longMessage: 'Critical thermal event detected in Cylinder Block B. Immediate shutdown sequence initiated automatically by PLC-04.',
    status: EventStatus.OPEN,
    correlationId: 'GRP-99',
    hasEvidence: true,
    tags: ['Thermal', 'Shutdown'],
  },
  {
    id: 'EVT-1023',
    timestamp: generateTimestamp(3),
    source: 'PLC-04',
    assetPath: 'Plant 1 > Power House > PLC 04',
    severity: EventSeverity.WARNING,
    type: EventType.FAULT,
    title: 'Pre-Alarm: Temp Rising',
    shortMessage: 'Rate of rise > 5°C/min detected.',
    status: EventStatus.ACKED,
    acknowledgedBy: 'J. Doe',
    correlationId: 'GRP-99',
    hasEvidence: true,
  },
  {
    id: 'EVT-1025',
    timestamp: generateTimestamp(4),
    source: 'Sensor-TH-02',
    assetPath: 'Plant 1 > Power House > Sensors',
    severity: EventSeverity.INFO,
    type: EventType.STATUS,
    title: 'Sensor Data Stream Interrupted',
    shortMessage: 'Momentary packet loss on sensor link.',
    status: EventStatus.RESOLVED,
    correlationId: 'GRP-99',
    hasEvidence: false,
  },
  {
    id: 'EVT-1022',
    timestamp: generateTimestamp(15),
    source: 'Operator Station 2',
    assetPath: 'Plant 1 > Control Room',
    severity: EventSeverity.INFO,
    type: EventType.ACTION,
    title: 'Shift Handover Note',
    shortMessage: 'Morning shift handover complete. No major incidents prior to 09:00.',
    status: EventStatus.RESOLVED,
    user: 'M. Vance',
    tags: ['Admin', 'Handover'],
  },
  {
    id: 'EVT-1021',
    timestamp: generateTimestamp(45),
    source: 'HVAC-02',
    assetPath: 'Plant 1 > Ventilation > AHU 2',
    severity: EventSeverity.WARNING,
    type: EventType.INSPECTION,
    title: 'Filter Maintenance Due',
    shortMessage: 'Differential pressure high across HEPA filter.',
    status: EventStatus.OPEN,
    hasEvidence: true,
  },
  {
    id: 'EVT-1020',
    timestamp: generateTimestamp(60),
    source: 'Security Sys',
    assetPath: 'Plant 1 > Perimeter > Gate 3',
    severity: EventSeverity.INFO,
    type: EventType.STATUS,
    title: 'Access Granted',
    shortMessage: 'Maintenance contractor access verified.',
    status: EventStatus.RESOLVED,
  },
  {
    id: 'EVT-1019',
    timestamp: generateTimestamp(120),
    source: 'UPS-B',
    assetPath: 'Plant 1 > Backup Power > UPS B',
    severity: EventSeverity.CRITICAL,
    type: EventType.ALARM,
    title: 'Battery Impedance High',
    shortMessage: 'Cell block 4 showing abnormal resistance.',
    status: EventStatus.OPEN,
    hasEvidence: true,
    tags: ['Electrical', 'Maintenance'],
  },
  {
    id: 'EVT-1018',
    timestamp: generateTimestamp(125),
    source: 'Operator Station 1',
    assetPath: 'Plant 1 > Control Room',
    severity: EventSeverity.INFO,
    type: EventType.OVERRIDE,
    title: 'Manual Pump Start',
    shortMessage: 'Operator manually started Feed Pump 2.',
    status: EventStatus.RESOLVED,
    user: 'S. Connor',
  },
  {
    id: 'EVT-1017',
    timestamp: generateTimestamp(180),
    source: 'Conveyor-Main',
    assetPath: 'Plant 1 > Assembly > Line 4',
    severity: EventSeverity.WARNING,
    type: EventType.FAULT,
    title: 'Belt Slip Detected',
    shortMessage: 'Motor current oscillation detected.',
    status: EventStatus.ACKED,
    acknowledgedBy: 'Auto-Bot',
    hasEvidence: true,
  },
  {
    id: 'EVT-1016',
    timestamp: generateTimestamp(300),
    source: 'System',
    assetPath: 'Plant 1 > Network',
    severity: EventSeverity.INFO,
    type: EventType.COMMENT,
    title: 'Daily Backup Complete',
    shortMessage: 'Database snapshot saved successfully.',
    status: EventStatus.RESOLVED,
  }
];



const data = [
  { name: '1', value: 40 },
  { name: '2', value: 30 },
  { name: '3', value: 20 },
  { name: '4', value: 27 },
  { name: '5', value: 18 },
  { name: '6', value: 23 },
  { name: '7', value: 34 },
  { name: '8', value: 85 }, // Spike
  { name: '9', value: 98 }, // Peak
  { name: '10', value: 0 }, // Fail
];

interface MiniTrendChartProps {
  color?: string;
  height?: number;
}

const MiniTrendChart: React.FC<MiniTrendChartProps> = ({ color = '#2563eb', height = 64 }) => {
  return (
    <div 
      className="w-full bg-neutral-50 rounded border border-neutral-100 overflow-hidden" 
      style={{ height: `${height}px` }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`colorValue-${color}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3} />
              <stop offset="95%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="name" hide />
          <YAxis hide domain={[0, 100]} />
          <Area
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            fillOpacity={1}
            fill={`url(#colorValue-${color})`}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};


interface AssetPathChipsProps {
  path: string;
}

const AssetPathChips: React.FC<AssetPathChipsProps> = ({ path }) => {
  const segments = path.split(' > ');

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {segments.map((segment, index) => (
        <React.Fragment key={index}>
          {index > 0 && <ChevronRight className="w-3 h-3 text-neutral-300" />}
          <div className="group relative">
            <button 
              className="text-[10px] uppercase font-bold tracking-wider text-neutral-500 hover:text-blue-600 hover:underline hover:bg-blue-50 px-1.5 py-0.5 rounded transition-all cursor-pointer flex items-center gap-1"
              onClick={(e) => {
                e.stopPropagation();
                // Placeholder for navigation
                console.log(`Navigating to asset: ${segment}`);
              }}
            >
              {index === segments.length - 1 && <Cpu className="w-3 h-3 inline-block mr-0.5" />}
              {segment}
            </button>
            
            {/* Context Tooltip on Hover */}
            <div className="absolute bottom-full left-0 mb-2 w-48 bg-neutral-900 text-white p-3 rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 flex flex-col gap-2">
               <div className="flex items-center justify-between border-b border-neutral-700 pb-1">
                 <span className="text-xs font-bold text-white">{segment}</span>
                 <span className="text-[10px] text-green-400">ONLINE</span>
               </div>
               <div className="flex justify-between text-[10px] text-neutral-400">
                  <span>Recent Events</span>
                  <span className="text-white font-mono">12</span>
               </div>
               <div className="flex justify-between text-[10px] text-neutral-400">
                  <span>Health Score</span>
                  <span className="text-white font-mono">98%</span>
               </div>
               <div className="absolute left-4 bottom-[-4px] w-2 h-2 bg-neutral-900 rotate-45"></div>
            </div>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
};




interface EventItemProps {
  data: EventItemData;
  representation: WidgetRepresentation;
  onClick: (data: EventItemData) => void;
  isCompact?: boolean;
}

const EventItem: React.FC<EventItemProps> = ({ data, representation, onClick, isCompact }) => {
  
  // Encoding: Color by Severity
  const getSeverityStyles = (severity: EventSeverity) => {
    switch (severity) {
      case EventSeverity.CRITICAL:
        return {
          border: 'border-red-500',
          bg: 'bg-red-50',
          text: 'text-red-700',
          icon: <Shield className="w-4 h-4 text-red-600" />,
          chip: 'border-red-200 bg-red-100 text-red-800',
          sparklineColor: '#ef4444'
        };
      case EventSeverity.WARNING:
        return {
          border: 'border-amber-500',
          bg: 'bg-amber-50',
          text: 'text-amber-700',
          icon: <AlertTriangle className="w-4 h-4 text-amber-600" />,
          chip: 'border-amber-200 bg-amber-100 text-amber-800',
          sparklineColor: '#d97706'
        };
      case EventSeverity.INFO:
      default:
        return {
          border: 'border-blue-500',
          bg: 'bg-blue-50',
          text: 'text-blue-700',
          icon: <Info className="w-4 h-4 text-blue-600" />,
          chip: 'border-blue-200 bg-blue-100 text-blue-800',
          sparklineColor: '#2563eb'
        };
    }
  };

  const getSourceIcon = (type: EventType) => {
     switch(type) {
         case EventType.ACTION: return <User className="w-3 h-3" />;
         case EventType.ALARM: return <Shield className="w-3 h-3" />;
         case EventType.COMMENT: return <FileText className="w-3 h-3" />;
         default: return <Terminal className="w-3 h-3" />;
     }
  };

  const getStatusBadge = (status: EventStatus) => {
      switch(status) {
          case EventStatus.ACKED:
              return (
                  <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-blue-600 bg-blue-50 border border-blue-100 px-1.5 py-0.5 rounded">
                      <CheckCircle className="w-3 h-3" /> Acked
                  </span>
              );
          case EventStatus.RESOLVED:
              return (
                  <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-green-600 bg-green-50 border border-green-100 px-1.5 py-0.5 rounded">
                      <CheckCircle className="w-3 h-3" /> Resolved
                  </span>
              );
          case EventStatus.OPEN:
          default:
              return (
                  <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-neutral-500 bg-neutral-100 border border-neutral-200 px-1.5 py-0.5 rounded">
                      <Circle className="w-3 h-3" /> Open
                  </span>
              );
      }
  };

  const styles = getSeverityStyles(data.severity);
  const timeString = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  const isCritical = data.severity === EventSeverity.CRITICAL;
  const isInfo = data.severity === EventSeverity.INFO;

  // R4: Data Table Representation
  if (representation === WidgetRepresentation.DATA_TABLE) {
      return (
          <tr onClick={() => onClick(data)} className="border-b border-neutral-100 hover:bg-neutral-50 cursor-pointer transition-colors group text-sm">
              <td className="py-3 px-4 w-12 text-center">
                  <div className={`w-2 h-2 rounded-full mx-auto ${styles.bg.replace('50', '500')}`}></div>
              </td>
              <td className="py-3 px-4 font-mono text-xs text-neutral-500 whitespace-nowrap">{data.timestamp.substring(11, 19)}</td>
              <td className="py-3 px-4 text-xs font-bold text-neutral-700 uppercase tracking-wide whitespace-nowrap">{data.source}</td>
              <td className="py-3 px-4">
                  <div className="font-medium text-neutral-900">{data.title}</div>
              </td>
              <td className="py-3 px-4 text-neutral-500 max-w-xs truncate">{data.shortMessage}</td>
              <td className="py-3 px-4 text-right">
                {getStatusBadge(data.status)}
              </td>
          </tr>
      );
  }

  return (
    <div className={`relative group mb-3 ${representation === WidgetRepresentation.TIMELINE_RAIL ? 'pl-8 pb-4' : ''}`}>
      
      {/* Timeline Rail Elements */}
      {representation === WidgetRepresentation.TIMELINE_RAIL && (
        <>
            <div className="absolute left-[11px] top-3 bottom-0 w-px bg-neutral-200 group-last:hidden"></div>
            <div className={`absolute left-0 top-1.5 w-6 h-6 rounded-full border-4 border-white shadow-sm flex items-center justify-center ${styles.bg}`}>
                {isCritical ? <AlertCircle className={`w-3 h-3 ${styles.text}`} /> : <div className={`w-2 h-2 rounded-full ${styles.bg.replace('50', '500')}`}></div>}
            </div>
        </>
      )}

      {/* Card Body */}
      <div 
        onClick={() => onClick(data)}
        className={`
            bg-white border rounded-lg transition-all cursor-pointer hover:shadow-md
            ${isCritical ? 'border-red-200 shadow-sm' : 'border-neutral-200'}
            ${isCompact ? 'p-3' : 'p-4'}
        `}
      >
        {/* Top Row: Meta & Time */}
        <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide border rounded ${styles.chip}`}>
                    {data.severity}
                </span>
                {!isCompact && (
                    <>
                        <div className="w-px h-3 bg-neutral-200 mx-1"></div>
                        <div className="flex items-center gap-1 text-neutral-500">
                             {getSourceIcon(data.type)}
                             <span className="text-[10px] uppercase font-bold tracking-widest">{data.source}</span>
                        </div>
                    </>
                )}
            </div>
            <div className="flex items-center gap-2">
                 {/* Status Badge */}
                 {getStatusBadge(data.status)}
                 
                 {/* Correlation Badge */}
                 {data.correlationId && (
                    <span className="text-[10px] bg-neutral-100 text-neutral-500 px-1.5 py-0.5 rounded border border-neutral-200 font-mono hidden sm:inline-block">
                        #{data.correlationId.split('-')[1]}
                    </span>
                 )}
                 
                 <span className="text-xs font-mono text-neutral-400 ml-2">{timeString}</span>
            </div>
        </div>

        {/* Middle: Content */}
        <div className="flex justify-between gap-4">
            <div className="flex-1">
                <h3 className={`font-bold text-neutral-900 ${isCritical ? 'text-base' : 'text-sm'}`}>
                    {data.title}
                </h3>
                <p className={`text-xs text-neutral-600 mt-1 ${isInfo ? 'line-clamp-1' : 'line-clamp-2'}`}>
                    {data.shortMessage}
                </p>
                
                {/* Asset Path - Always clickable */}
                {!isCompact && !isInfo && (
                    <div className="mt-3">
                         <AssetPathChips path={data.assetPath} />
                    </div>
                )}
            </div>

            {/* Right: Inline Sparkline for CRITICAL items only */}
            {isCritical && data.hasEvidence && !isCompact && (
                <div className="w-24 shrink-0 hidden sm:block">
                     <MiniTrendChart color={styles.sparklineColor} height={40} />
                     <div className="text-[9px] text-right font-mono text-neutral-400 mt-1">1H TREND</div>
                </div>
            )}
        </div>

        {/* Bottom: Footer / Actions on Hover */}
        <div className="mt-3 pt-2 border-t border-neutral-50 flex justify-between items-center opacity-0 group-hover:opacity-100 transition-opacity">
             <div className="flex gap-2">
                 <span className="text-[10px] text-neutral-400 uppercase tracking-widest">Quick Actions</span>
             </div>
             <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wide">
                 View Details <ChevronRight className="w-3 h-3" />
             </div>
        </div>
      </div>
    </div>
  );
};



interface CorrelationGroupProps {
  groupId: string;
  events: EventItemData[];
  representation: WidgetRepresentation;
  onEventClick: (data: EventItemData) => void;
}

const CorrelationGroup: React.FC<CorrelationGroupProps> = ({ groupId, events, representation, onEventClick }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (events.length === 0) return null;

  // Determine aggregate severity
  const hasCritical = events.some(e => e.severity === EventSeverity.CRITICAL);
  const hasWarning = events.some(e => e.severity === EventSeverity.WARNING);
  
  const aggregateSeverity = hasCritical ? EventSeverity.CRITICAL : (hasWarning ? EventSeverity.WARNING : EventSeverity.INFO);

  const getHeaderStyles = () => {
     if (hasCritical) return 'bg-red-50 border-red-200';
     if (hasWarning) return 'bg-amber-50 border-amber-200';
     return 'bg-neutral-50 border-neutral-200';
  };

  const startTime = new Date(events[events.length - 1].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const endTime = new Date(events[0].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`mb-4 rounded-xl border overflow-hidden transition-all ${getHeaderStyles()} ${isExpanded ? 'shadow-md' : 'shadow-sm'}`}>
        {/* Parent Header */}
        <div 
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-4 cursor-pointer flex items-center justify-between hover:bg-white/50 transition-colors"
        >
            <div className="flex items-center gap-4">
                <div className="p-2 bg-white rounded-lg border border-black/5 shadow-sm">
                    <Layers className="w-5 h-5 text-neutral-600" />
                </div>
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-bold uppercase tracking-widest text-neutral-500">Correlation Group</span>
                        <span className="font-mono text-xs bg-white px-1.5 rounded border border-neutral-200 text-neutral-600">#{groupId}</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-neutral-900">{events.length} Related Events</span>
                        {hasCritical && (
                            <span className="flex items-center gap-1 text-xs text-red-600 font-bold">
                                <AlertCircle className="w-3 h-3" /> Critical
                            </span>
                        )}
                    </div>
                </div>
            </div>

            <div className="text-right">
                <div className="text-xs font-mono text-neutral-500">{startTime} - {endTime}</div>
                <div className="text-[10px] text-neutral-400 mt-1 uppercase tracking-wide">
                    {isExpanded ? 'Click to Collapse' : 'Click to Expand'}
                </div>
            </div>
        </div>

        {/* Children List */}
        {isExpanded && (
            <div className="bg-white border-t border-black/5 p-4 pl-8 space-y-4">
                 {/* Visual Connector Line */}
                 <div className="absolute left-[34px] top-20 bottom-8 w-0.5 bg-neutral-200 border-l border-dashed border-neutral-300 z-0"></div>
                 
                 {events.map((event, idx) => (
                     <div key={event.id} className="relative z-10">
                        <EventItem 
                            data={event}
                            representation={representation}
                            onClick={onEventClick}
                            isCompact={true}
                        />
                     </div>
                 ))}
            </div>
        )}
    </div>
  );
};



interface AssetGroupProps {
  assetName: string;
  events: EventItemData[];
  representation: WidgetRepresentation;
  onEventClick: (data: EventItemData) => void;
}

const AssetGroup: React.FC<AssetGroupProps> = ({ assetName, events, representation, onEventClick }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (events.length === 0) return null;

  // Aggregate Severity
  const hasCritical = events.some(e => e.severity === EventSeverity.CRITICAL);
  const hasWarning = events.some(e => e.severity === EventSeverity.WARNING);
  
  // Mock Health Score based on severity
  const healthScore = hasCritical ? 45 : (hasWarning ? 78 : 98);
  const healthColor = hasCritical ? 'text-red-600' : (hasWarning ? 'text-amber-600' : 'text-green-600');

  const getHeaderStyles = () => {
     if (hasCritical) return 'bg-white border-l-4 border-l-red-500 border-y border-r border-neutral-200';
     if (hasWarning) return 'bg-white border-l-4 border-l-amber-500 border-y border-r border-neutral-200';
     return 'bg-white border-l-4 border-l-blue-500 border-y border-r border-neutral-200';
  };

  return (
    <div className={`mb-4 rounded-r-lg shadow-sm overflow-hidden ${getHeaderStyles()}`}>
        {/* Asset Header */}
        <div 
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-4 cursor-pointer flex items-center justify-between hover:bg-neutral-50 transition-colors"
        >
            <div className="flex items-center gap-4">
                <div className="p-2 bg-neutral-100 rounded-lg text-neutral-600">
                    <Box className="w-5 h-5" />
                </div>
                <div>
                    <h3 className="text-sm font-bold text-neutral-900 leading-none mb-1">{assetName}</h3>
                    <div className="flex items-center gap-3">
                         <span className="text-xs text-neutral-500">{events.length} Events</span>
                         <span className="text-[10px] text-neutral-300">|</span>
                         <div className={`flex items-center gap-1 text-xs font-bold font-mono ${healthColor}`}>
                            <Activity className="w-3 h-3" />
                            {healthScore}% HEALTH
                         </div>
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-4">
                {hasCritical && (
                    <span className="px-2 py-1 bg-red-100 text-red-700 text-[10px] font-bold uppercase tracking-wide rounded">
                        Critical Attention
                    </span>
                )}
                {isExpanded ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
            </div>
        </div>

        {/* Children List */}
        {isExpanded && (
            <div className="bg-neutral-50/50 p-4 border-t border-neutral-100 space-y-3">
                 {events.map((event) => (
                     <EventItem 
                        key={event.id} 
                        data={event}
                        representation={WidgetRepresentation.LIST_CARD} // Use simpler card inside groups
                        onClick={onEventClick}
                        isCompact={true}
                     />
                 ))}
            </div>
        )}
    </div>
  );
};


interface FilterBarProps {
  currentVariation: WidgetVariation;
  setVariation: (v: WidgetVariation) => void;
  searchTerm: string;
  setSearchTerm: (s: string) => void;
}

const FilterBar: React.FC<FilterBarProps> = ({
  currentVariation,
  setVariation,
  searchTerm,
  setSearchTerm
}) => {
  return (
    <div className="bg-white border-b border-neutral-200 sticky top-0 z-20 px-3 py-2 flex flex-wrap items-center gap-2 shadow-sm">

      {/* Left: Search & Global Filter */}
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <div className="relative group flex-1 min-w-[100px] max-w-[200px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-neutral-400 group-focus-within:text-blue-600 transition-colors" />
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="h-7 w-full pl-7 pr-2 rounded-md bg-neutral-100 border border-transparent focus:bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-xs transition-all outline-none"
          />
        </div>

        <div className="flex gap-1 shrink-0">
            <button className="h-7 px-2 flex items-center gap-1 rounded-md border border-neutral-200 bg-white text-[10px] font-medium text-neutral-600 hover:bg-neutral-50 transition-colors">
                <Clock className="w-3 h-3" />
                <span className="hidden sm:inline">24h</span>
            </button>
            <button className="h-7 px-2 flex items-center gap-1 rounded-md border border-neutral-200 bg-white text-[10px] font-medium text-neutral-600 hover:bg-neutral-50 transition-colors">
                <Filter className="w-3 h-3" />
                <span className="hidden sm:inline">Filter</span>
            </button>
        </div>
      </div>

      {/* Right: Variant Switcher */}
      <div className="flex items-center shrink-0">
         <div className="relative">
             <select
                value={currentVariation}
                onChange={(e) => setVariation(e.target.value as WidgetVariation)}
                className="appearance-none h-7 pl-2 pr-7 rounded-md bg-neutral-900 text-white text-[10px] font-bold uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-neutral-700"
             >
                 {Object.values(WidgetVariation).map(v => (
                     <option key={v} value={v}>{v}</option>
                 ))}
             </select>
             <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-neutral-400 pointer-events-none" />
         </div>
      </div>
    </div>
  );
};




interface DrillDownDrawerProps {
  event: EventItemData | null;
  onClose: () => void;
}

const DrillDownDrawer: React.FC<DrillDownDrawerProps> = ({ event, onClose }) => {
  const [activeTab, setActiveTab] = useState<'details' | 'evidence' | 'actions' | 'history'>('details');

  if (!event) return null;

  const getSeverityColor = (sev: EventSeverity) => {
    switch (sev) {
      case EventSeverity.CRITICAL: return 'text-red-500 bg-red-50 border-red-200';
      case EventSeverity.WARNING: return 'text-amber-600 bg-amber-50 border-amber-200';
      default: return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  const SeverityIcon = {
    [EventSeverity.CRITICAL]: ShieldAlert,
    [EventSeverity.WARNING]: AlertTriangle,
    [EventSeverity.INFO]: Info,
  }[event.severity];

  return (
    <div className="fixed inset-y-0 right-0 w-[520px] bg-white shadow-2xl border-l border-neutral-200 z-50 flex flex-col transform transition-transform duration-300 ease-in-out">
      {/* Header */}
      <div className="h-20 border-b border-neutral-100 flex items-center justify-between px-6 bg-neutral-50/50">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg border ${getSeverityColor(event.severity)}`}>
            <SeverityIcon className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">
              {event.id}
            </div>
            <h2 className="text-lg font-bold text-neutral-900 leading-tight">
              {event.title}
            </h2>
          </div>
        </div>
        <button onClick={onClose} className="text-neutral-400 hover:text-neutral-900">
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Tabs - Fixed Order */}
      <div className="flex border-b border-neutral-100 px-6 gap-6">
        {['details', 'evidence', 'actions', 'history'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            className={`h-12 text-xs font-bold uppercase tracking-widest border-b-2 transition-colors ${
              activeTab === tab
                ? 'border-neutral-900 text-neutral-900'
                : 'border-transparent text-neutral-400 hover:text-neutral-600'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 bg-neutral-50/30">
        
        {/* DETAILS TAB */}
        {activeTab === 'details' && (
          <div className="space-y-6">
            <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
              <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 block mb-1">
                Full Message
              </label>
              <p className="text-sm text-neutral-700 leading-relaxed font-medium">
                {event.longMessage || event.shortMessage}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
                <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 block mb-1">
                  Source
                </label>
                <div className="flex items-center gap-2 text-neutral-900 font-medium text-sm">
                  <Activity className="w-4 h-4 text-neutral-400" />
                  {event.source}
                </div>
              </div>
              <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
                <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 block mb-1">
                  Timestamp
                </label>
                <div className="flex items-center gap-2 text-neutral-900 font-medium text-sm">
                  <Clock className="w-4 h-4 text-neutral-400" />
                  {new Date(event.timestamp).toLocaleString()}
                </div>
              </div>
            </div>

            <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
               <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 block mb-3">
                  Asset Context
                </label>
                <AssetPathChips path={event.assetPath} />
            </div>
          </div>
        )}

        {/* EVIDENCE TAB */}
        {activeTab === 'evidence' && (
          <div className="space-y-4">
            {event.hasEvidence ? (
              <>
                <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
                   <div className="flex justify-between items-center mb-2">
                    <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400">
                      Trend Snapshot (1H)
                    </label>
                    <span className="text-xs font-mono text-red-500 font-bold">98°C PEAK</span>
                   </div>
                   <MiniTrendChart color={event.severity === EventSeverity.CRITICAL ? '#ef4444' : '#2563eb'} />
                </div>
                <div className="bg-white p-4 rounded-xl border border-neutral-100 shadow-sm">
                  <label className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 block mb-2">
                    Telemetry at Trigger
                  </label>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-neutral-100 text-left text-neutral-400">
                        <th className="pb-2 font-normal">Parameter</th>
                        <th className="pb-2 font-normal">Value</th>
                        <th className="pb-2 font-normal">Ref</th>
                      </tr>
                    </thead>
                    <tbody className="text-neutral-700">
                      <tr className="border-b border-neutral-50">
                        <td className="py-2">Cylinder Temp A</td>
                        <td className="py-2 font-mono">88°C</td>
                        <td className="py-2 text-neutral-400">OK</td>
                      </tr>
                      <tr className="border-b border-neutral-50 font-bold bg-red-50 text-red-700">
                        <td className="py-2 pl-2 rounded-l">Cylinder Temp B</td>
                        <td className="py-2 font-mono">98°C</td>
                        <td className="py-2 pr-2 rounded-r">HIGH</td>
                      </tr>
                      <tr>
                        <td className="py-2">Coolant Flow</td>
                        <td className="py-2 font-mono">12 L/m</td>
                        <td className="py-2 text-neutral-400">LOW</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="text-center py-10 text-neutral-400 bg-white rounded-xl border border-neutral-200 border-dashed">
                <FileText className="w-10 h-10 mx-auto mb-3 opacity-20" />
                <p>No telemetry snapshot available.</p>
              </div>
            )}
          </div>
        )}

        {/* ACTIONS TAB */}
        {activeTab === 'actions' && (
          <div className="space-y-3">
             <div className="grid grid-cols-2 gap-3">
                <button className="py-3 px-4 bg-blue-600 text-white rounded-lg shadow-sm hover:bg-blue-700 transition-all flex flex-col items-center justify-center gap-1">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-xs font-bold uppercase">Acknowledge</span>
                </button>
                <button className="py-3 px-4 bg-white border border-neutral-200 rounded-lg shadow-sm hover:border-blue-400 hover:text-blue-600 transition-all flex flex-col items-center justify-center gap-1">
                    <CheckCircle className="w-5 h-5 text-neutral-400" />
                    <span className="text-xs font-bold uppercase">Resolve</span>
                </button>
             </div>
             
             <div className="h-px bg-neutral-200 my-4"></div>

             <button className="w-full py-3 px-4 bg-white border border-neutral-200 rounded-lg shadow-sm hover:border-blue-400 hover:text-blue-600 transition-all flex items-center gap-3 group text-left">
                <div className="p-2 bg-neutral-50 rounded group-hover:bg-blue-50">
                    <Wrench className="w-4 h-4 text-neutral-500 group-hover:text-blue-600" />
                </div>
                <div>
                    <div className="text-sm font-bold text-neutral-900">Create Ticket</div>
                    <div className="text-xs text-neutral-500">Link to Maintenance OS</div>
                </div>
             </button>
             <button className="w-full py-3 px-4 bg-white border border-neutral-200 rounded-lg shadow-sm hover:border-blue-400 hover:text-blue-600 transition-all flex items-center gap-3 group text-left">
                <div className="p-2 bg-neutral-50 rounded group-hover:bg-blue-50">
                    <History className="w-4 h-4 text-neutral-500 group-hover:text-blue-600" />
                </div>
                <div>
                    <div className="text-sm font-bold text-neutral-900">Assign to Shift</div>
                    <div className="text-xs text-neutral-500">Forward to next operator</div>
                </div>
             </button>
          </div>
        )}

        {/* HISTORY TAB */}
        {activeTab === 'history' && (
           <div className="bg-white rounded-xl border border-neutral-100 shadow-sm p-6">
                <div className="space-y-6 border-l-2 border-neutral-100 ml-2 pl-6">
                   <div className="relative">
                      <div className="absolute -left-[31px] top-1 w-3 h-3 rounded-full bg-neutral-300 border-2 border-white box-content"></div>
                      <p className="text-xs text-neutral-500 mb-1">{new Date(event.timestamp).toLocaleString()}</p>
                      <p className="text-sm font-bold text-neutral-900">Event Created</p>
                      <p className="text-xs text-neutral-600">Generated by System Rule #402</p>
                   </div>
                   {event.status !== 'OPEN' && (
                     <div className="relative">
                        <div className="absolute -left-[31px] top-1 w-3 h-3 rounded-full bg-blue-500 border-2 border-white box-content"></div>
                        <p className="text-xs text-neutral-500 mb-1">2 minutes later</p>
                        <p className="text-sm font-bold text-neutral-900">Status Changed: {event.status}</p>
                        <p className="text-xs text-neutral-600">Action by Operator {event.acknowledgedBy || 'Unknown'}</p>
                     </div>
                   )}
                </div>
           </div>
        )}
      </div>
    </div>
  );
};








const EventLogStream = ({ initialVariation }) => {
  const [variation, setVariation] = useState(initialVariation || WidgetVariation.CHRONOLOGICAL_TIMELINE);
  const [selectedEvent, setSelectedEvent] = useState<EventItemData | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Map Variation to Internal Representation
  const representation = useMemo(() => {
     switch (variation) {
         case WidgetVariation.TABULAR_LOG_VIEW: return WidgetRepresentation.DATA_TABLE;
         case WidgetVariation.COMPACT_CARD_FEED: return WidgetRepresentation.LIST_CARD;
         case WidgetVariation.CHRONOLOGICAL_TIMELINE: return WidgetRepresentation.TIMELINE_RAIL;
         case WidgetVariation.CORRELATION_STACK: return WidgetRepresentation.TIMELINE_RAIL;
         case WidgetVariation.GROUPED_BY_ASSET: return WidgetRepresentation.LIST_CARD;
         default: return WidgetRepresentation.TIMELINE_RAIL;
     }
  }, [variation]);

  const filteredEvents = useMemo(() => {
    return MOCK_EVENTS.filter(e => 
        e.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
        e.source.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [searchTerm]);

  // Grouping Logic for Correlation Stack
  const correlationGroups = useMemo(() => {
      if (variation !== WidgetVariation.CORRELATION_STACK) return null;
      
      const groups: Record<string, EventItemData[]> = {};
      const singles: EventItemData[] = [];

      filteredEvents.forEach(event => {
          if (event.correlationId) {
              if (!groups[event.correlationId]) groups[event.correlationId] = [];
              groups[event.correlationId].push(event);
          } else {
              singles.push(event);
          }
      });
      return { groups, singles };
  }, [filteredEvents, variation]);

  // Grouping Logic for Assets
  const assetGroups = useMemo(() => {
      if (variation !== WidgetVariation.GROUPED_BY_ASSET) return null;

      const groups: Record<string, EventItemData[]> = {};
      filteredEvents.forEach(event => {
          // Use the 'System' part of asset path for grouping, or Source if path is short
          const parts = event.assetPath.split(' > ');
          // E.g., Plant 1 > Power House > DG A. We want "Power House" or "DG A".
          // Let's use Source for simplicity and clarity in demo
          const key = event.source; 
          if (!groups[key]) groups[key] = [];
          groups[key].push(event);
      });
      return groups;
  }, [filteredEvents, variation]);


  return (
    <div className="h-full flex flex-col bg-neutral-50">
      <FilterBar 
        currentVariation={variation} 
        setVariation={setVariation}
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
      />

      <div className="flex-1 overflow-y-auto overflow-x-hidden relative">
        <div className="w-full p-4">

            {/* Header */}
            <div className="mb-3 flex items-end justify-between">
                <div>
                    <h1 className="text-lg font-bold text-neutral-900 tracking-tight">Event Stream</h1>
                    <p className="text-xs text-neutral-500 mt-0.5">
                        {variation} • {filteredEvents.length} Events
                    </p>
                </div>
            </div>

            {/* Render Logic Based on Variant */}
            
            {/* 1. Tabular View */}
            {representation === WidgetRepresentation.DATA_TABLE ? (
                <div className="bg-white rounded-xl border border-neutral-200 shadow-sm overflow-hidden">
                    <table className="w-full text-left">
                        <thead className="bg-neutral-50 border-b border-neutral-200">
                            <tr>
                                <th className="py-3 px-4 w-12"></th>
                                <th className="py-3 px-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400">Time</th>
                                <th className="py-3 px-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400">Source</th>
                                <th className="py-3 px-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400">Event</th>
                                <th className="py-3 px-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400">Message</th>
                                <th className="py-3 px-4 text-[10px] uppercase font-bold tracking-widest text-neutral-400 text-right">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredEvents.map(event => (
                                <EventItem 
                                    key={event.id} 
                                    data={event} 
                                    representation={representation}
                                    onClick={setSelectedEvent}
                                />
                            ))}
                        </tbody>
                    </table>
                </div>

            /* 2. Correlation Stack */
            ) : variation === WidgetVariation.CORRELATION_STACK && correlationGroups ? (
                <div className="space-y-4">
                    {Object.entries(correlationGroups.groups).map(([groupId, events]) => (
                        <CorrelationGroup 
                            key={groupId}
                            groupId={groupId}
                            events={events}
                            representation={WidgetRepresentation.TIMELINE_RAIL}
                            onEventClick={setSelectedEvent}
                        />
                    ))}
                    {correlationGroups.singles.length > 0 && (
                         <div className="border-t border-dashed border-neutral-200 pt-6 mt-6">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-neutral-400 mb-4">Uncorrelated Events</h3>
                            {correlationGroups.singles.map(event => (
                                <EventItem 
                                    key={event.id} 
                                    data={event} 
                                    representation={WidgetRepresentation.LIST_CARD}
                                    onClick={setSelectedEvent}
                                />
                            ))}
                         </div>
                    )}
                </div>

            /* 3. Grouped by Asset */
            ) : variation === WidgetVariation.GROUPED_BY_ASSET && assetGroups ? (
                 <div className="space-y-4">
                    {Object.entries(assetGroups).map(([assetName, events]) => (
                        <AssetGroup 
                            key={assetName}
                            assetName={assetName}
                            events={events}
                            representation={representation}
                            onEventClick={setSelectedEvent}
                        />
                    ))}
                 </div>

            /* 4. Timeline & Compact Feed */
            ) : (
                <div className="flex flex-col">
                    {filteredEvents.map(event => (
                        <EventItem 
                            key={event.id} 
                            data={event} 
                            representation={representation}
                            onClick={setSelectedEvent}
                            isCompact={variation === WidgetVariation.COMPACT_CARD_FEED}
                        />
                    ))}
                </div>
            )}
            
            {/* Empty State */}
            {filteredEvents.length === 0 && (
                <div className="text-center py-20 border-2 border-dashed border-neutral-200 rounded-xl">
                    <p className="text-neutral-400 font-medium">No events found matching your filter.</p>
                </div>
            )}
        </div>
      </div>

      <DrillDownDrawer 
        event={selectedEvent} 
        onClose={() => setSelectedEvent(null)} 
      />
    </div>
  );
};


export default function ScenarioComponent({ data }) {
  const variation = data ? data.variation : undefined;
  return <EventLogStream initialVariation={variation} />;
}
