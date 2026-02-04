// @ts-nocheck
import React from 'react';
import { 
  AlertTriangle, 
  CheckCircle2, 
  Info, 
  XCircle, 
  X, 
  Clock, 
  MoreHorizontal, 
  ShieldAlert,
  ArrowUpCircle,
  PlayCircle,
  Eye,
  Check,
  PauseCircle,
  RotateCcw
} from 'lucide-react';

interface AlertProps {
  data: AlertData;
  variant: AlertVariant;
  onAction?: (id: string, action: string) => void;
  onDismiss?: (id: string) => void;
  className?: string;
}

// Visual Config for Severity (Priority, Color)
const SEVERITY_CONFIG: Record<string, { color: string; bg: string; border: string; accent: string; icon: React.ElementType }> = {
  critical: {
    color: 'text-red-600',
    bg: 'bg-red-50',
    border: 'border-red-500',
    accent: 'bg-red-500',
    icon: XCircle
  },
  high: {
    color: 'text-red-600',
    bg: 'bg-red-50',
    border: 'border-red-500',
    accent: 'bg-red-500',
    icon: XCircle
  },
  warning: {
    color: 'text-amber-600',
    bg: 'bg-amber-50',
    border: 'border-amber-500',
    accent: 'bg-amber-500',
    icon: AlertTriangle
  },
  medium: {
    color: 'text-amber-600',
    bg: 'bg-amber-50',
    border: 'border-amber-500',
    accent: 'bg-amber-500',
    icon: AlertTriangle
  },
  low: {
    color: 'text-blue-600',
    bg: 'bg-blue-50',
    border: 'border-blue-500',
    accent: 'bg-blue-500',
    icon: Info
  },
  success: {
    color: 'text-green-600',
    bg: 'bg-green-50',
    border: 'border-green-500',
    accent: 'bg-green-500',
    icon: CheckCircle2
  },
  info: {
    color: 'text-blue-600',
    bg: 'bg-blue-50',
    border: 'border-blue-500',
    accent: 'bg-blue-500',
    icon: Info
  }
};

// Visual Config for States (Icon, Label, Opacity)
const STATE_CONFIG: Record<AlertState, { icon: React.ElementType; label: string; style: string }> = {
  new: { icon: ZapIcon, label: 'New', style: 'text-blue-600 bg-blue-50 border-blue-200' },
  seen: { icon: Eye, label: 'Seen', style: 'text-neutral-500 bg-transparent border-transparent' },
  acknowledged: { icon: Check, label: 'Ack', style: 'text-neutral-600 bg-neutral-100 border-neutral-200' },
  in_progress: { icon: PlayCircle, label: 'Active', style: 'text-purple-600 bg-purple-50 border-purple-200' },
  resolved: { icon: CheckCircle2, label: 'Resolved', style: 'text-green-600 bg-green-50 border-transparent opacity-75' },
  auto_resolved: { icon: RotateCcw, label: 'Auto-Resolved', style: 'text-teal-600 bg-teal-50 border-teal-200 border-dashed' },
  escalated: { icon: ArrowUpCircle, label: 'Escalated', style: 'text-orange-600 bg-orange-50 border-orange-200 font-bold' },
  suppressed: { icon: PauseCircle, label: 'Snoozed', style: 'text-neutral-400 bg-neutral-50 border-neutral-100 opacity-60' },
};

function ZapIcon({ className }: { className?: string }) {
   return <svg className={className} viewBox="0 0 24 24" fill="currentColor" stroke="none"><path d="M13 2L3 14H12L11 22L21 10H12L13 2Z"></path></svg>;
}

const formatTime = (isoString: string) => {
  const date = new Date(isoString);
  const now = new Date();
  const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / 60000);

  if (diffInMinutes < 1) return 'Just now';
  if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
  if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
  return date.toLocaleDateString();
};

const AlertNotification: React.FC<AlertProps> = ({ 
  data, 
  variant, 
  onAction,
  onDismiss,
  className = ''
}) => {
  const severityCfg = SEVERITY_CONFIG[data.severity] || SEVERITY_CONFIG.info;
  const stateCfg = STATE_CONFIG[data.state] || STATE_CONFIG.new;
  const Icon = severityCfg.icon;

  // -- VARIANT: BADGE (Compact Status) --
  if (variant === 'badge') {
    return (
      <button 
        onClick={() => onAction?.(data.id, 'open')}
        className={`group inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border transition-all hover:shadow-sm ${severityCfg.bg} ${severityCfg.border} border-opacity-30 hover:brightness-95 ${className}`}
        title={`${data.severity.toUpperCase()}: ${data.title}`}
      >
        <Icon className={`w-3.5 h-3.5 ${severityCfg.color}`} />
        <span className={`text-[10px] font-bold uppercase tracking-wider ${severityCfg.color}`}>
          {data.severity}
        </span>
      </button>
    );
  }

  // -- VARIANT: TOAST (Notification) --
  if (variant === 'toast') {
    return (
      <div className={`group relative flex w-full overflow-hidden rounded-lg bg-white shadow-xl border border-neutral-200 transition-all duration-300 animate-slideUp ${className}`}>
        {/* Severity Indicator using explicit accent class */}
        <div className={`w-1.5 ${severityCfg.accent}`} />
        <div className="p-4 flex-1">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <Icon className={`h-5 w-5 ${severityCfg.color}`} aria-hidden="true" />
            </div>
            <div className="ml-3 w-0 flex-1 pt-0.5">
              <p className="text-sm font-semibold text-neutral-900">{data.title}</p>
              <p className="mt-1 text-xs text-neutral-500 line-clamp-2">{data.message}</p>
            </div>
            <div className="ml-4 flex flex-shrink-0">
              <button
                type="button"
                className="inline-flex rounded-md bg-white text-neutral-400 hover:text-neutral-500 focus:outline-none"
                onClick={() => onDismiss?.(data.id)}
              >
                <span className="sr-only">Close</span>
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // -- VARIANT: BANNER (Inline Context) --
  if (variant === 'banner') {
    return (
      <div className={`relative flex flex-col md:flex-row md:items-center justify-between p-4 rounded-lg border-l-4 ${severityCfg.bg} ${severityCfg.border.replace('border-', 'border-l-')} border-neutral-100 ${className} w-full`}>
        <div className="flex items-start md:items-center gap-4 flex-1">
          <div className={`p-2 rounded-full bg-white/60 ${severityCfg.color} hidden md:block flex-shrink-0`}>
            <Icon className="w-5 h-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-1 flex-wrap">
              <span className="text-sm font-semibold text-neutral-900 whitespace-nowrap">{data.title}</span>
              <span className="text-[10px] uppercase tracking-wider font-bold text-neutral-400 border border-neutral-200 px-1.5 rounded whitespace-nowrap">
                {data.source}
              </span>
              {data.state === 'escalated' && (
                 <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-100 text-orange-700 font-bold uppercase whitespace-nowrap">Escalated</span>
              )}
            </div>
            <p className="text-xs text-neutral-600 truncate">{data.message}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4 mt-3 md:mt-0 pl-10 md:pl-0 flex-shrink-0">
          {data.evidence && (
            <div className="flex flex-col items-end px-4 border-r border-neutral-200/50 hidden sm:flex">
               <span className="text-[9px] uppercase tracking-wider text-neutral-400 font-bold">{data.evidence.label}</span>
               <div className="flex items-center gap-1 font-mono text-neutral-900 font-medium">
                  {data.evidence.value} <span className="text-neutral-500 text-xs">{data.evidence.unit}</span>
               </div>
            </div>
          )}
          <div className="flex gap-2">
            {data.actions.slice(0, 2).map((action, idx) => (
              <button 
                key={idx}
                onClick={() => onAction?.(data.id, action.intent)}
                className={`text-xs font-bold uppercase tracking-wide px-3 py-1.5 rounded border transition-all shadow-sm active:scale-95
                  ${action.type === 'primary' 
                    ? 'bg-neutral-900 text-white border-neutral-900 hover:bg-neutral-800' 
                    : 'bg-white text-neutral-600 border-neutral-200 hover:border-neutral-300 hover:text-neutral-900 hover:bg-neutral-50'
                  }`}
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // -- VARIANT: MODAL (Critical/Blocking) --
  if (variant === 'modal') {
    return (
      <div className="absolute inset-0 z-50 flex items-center justify-center p-4 sm:p-0">
        <div className="absolute inset-0 bg-neutral-900/60 backdrop-blur-sm transition-opacity" onClick={() => onDismiss?.(data.id)} />
        
        <div className="relative transform overflow-y-auto rounded-xl bg-white text-left shadow-2xl transition-all sm:my-8 w-full max-w-lg max-h-full border-t-8 border-red-500">
          <div className="bg-white px-4 pb-4 pt-5 sm:p-6 sm:pb-4">
            <div className="sm:flex sm:items-start">
              <div className="mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-red-50 sm:mx-0 sm:h-10 sm:w-10">
                <ShieldAlert className="h-6 w-6 text-red-600" aria-hidden="true" />
              </div>
              <div className="mt-3 text-center sm:ml-4 sm:mt-0 sm:text-left w-full">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-lg font-bold leading-6 text-neutral-900 uppercase tracking-tight">
                      {data.severity} Alert
                    </h3>
                    <p className="text-xs text-red-600 font-medium mt-0.5 uppercase tracking-wide">Immediate Action Required</p>
                  </div>
                  <span className="font-mono text-xs text-neutral-400">{data.id}</span>
                </div>
                
                <div className="mt-5 bg-neutral-50 p-4 rounded-lg border border-neutral-100">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-500">{data.category} • {data.source}</span>
                        <span className="text-xs text-neutral-400 font-mono">{formatTime(data.timestamp)}</span>
                    </div>
                    <p className="text-base font-semibold text-neutral-900">{data.title}</p>
                    <p className="text-sm text-neutral-600 mt-1 leading-relaxed">{data.message}</p>

                    {data.evidence && (
                      <div className="mt-4 pt-4 border-t border-neutral-200">
                         <div className="flex items-center gap-4">
                            <div className="bg-white border border-neutral-200 rounded p-2 shadow-sm min-w-[120px]">
                               <span className="block text-[9px] uppercase tracking-wider text-neutral-400 font-bold mb-0.5">{data.evidence.label}</span>
                               <div className="flex items-baseline gap-1 font-mono text-neutral-900 font-bold text-xl leading-none">
                                  {data.evidence.value} <span className="text-neutral-500 text-xs font-normal">{data.evidence.unit}</span>
                                </div>
                            </div>
                            <div className="text-xs text-neutral-500">
                               <div className="flex items-center gap-1 mb-1"><span className="font-bold text-neutral-700">Trigger:</span> {data.triggerCondition || 'Threshold exceeded'}</div>
                               {data.threshold && <div className="flex items-center gap-1"><span className="font-bold text-neutral-700">Limit:</span> {data.threshold}</div>}
                            </div>
                         </div>
                      </div>
                    )}
                </div>

                {data.recommendedNextStep && (
                   <div className="mt-4 flex items-start gap-3 text-sm text-neutral-700 bg-blue-50/50 p-3 rounded border border-blue-100">
                      <Info className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold text-blue-700 text-xs uppercase block mb-0.5">Recommended Action</span>
                        {data.recommendedNextStep}
                      </div>
                   </div>
                )}
              </div>
            </div>
          </div>
          <div className="bg-neutral-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 gap-2 border-t border-neutral-100">
            {data.actions.map((action, idx) => (
               <button
                key={idx}
                type="button"
                className={`inline-flex w-full justify-center rounded-md px-3 py-2 text-sm font-semibold shadow-sm sm:w-auto transition-transform active:scale-95
                  ${action.type === 'primary' 
                    ? 'bg-red-600 text-white hover:bg-red-500' 
                    : 'bg-white text-neutral-900 ring-1 ring-inset ring-neutral-300 hover:bg-neutral-50'
                  }`}
                onClick={() => onAction?.(data.id, action.intent)}
              >
                {action.label}
              </button>
            ))}
             <button
              type="button"
              className="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-neutral-900 shadow-sm ring-1 ring-inset ring-neutral-300 hover:bg-neutral-50 sm:mt-0 sm:w-auto transition-transform active:scale-95"
              onClick={() => onDismiss?.(data.id)}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  // -- VARIANT: CARD (Alert Feed / Global Stream) --
  // Core default variant
  return (
    <div className={`group bg-white rounded-xl border border-neutral-200 p-4 shadow-sm hover:shadow-md transition-all duration-200 relative overflow-hidden w-full ${className}`}>
      
      {/* Severity Strip */}
      <div className={`absolute left-0 top-0 bottom-0 w-1 ${severityCfg.bg.replace('50', '500')}`} />

      <div className="pl-3 flex flex-col gap-3 w-full">
        {/* Header Row: Category, Source, State Badge, Time */}
        <div className="flex flex-wrap items-center justify-between gap-y-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-500 whitespace-nowrap">
              {data.category}
            </span>
            <span className="text-neutral-300">•</span>
            <span className="text-[10px] uppercase font-bold tracking-widest text-neutral-400 truncate max-w-[120px]" title={data.source}>
              {data.source}
            </span>
          </div>
          
          <div className="flex items-center gap-2 ml-auto flex-shrink-0">
            {/* State Badge with tooltip */}
            <div 
              className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase font-bold border ${stateCfg.style}`}
              title={`Status: ${stateCfg.label}`}
            >
              {React.createElement(stateCfg.icon, { className: "w-3 h-3" })}
              <span>{stateCfg.label}</span>
            </div>
            
            <span className="text-xs text-neutral-300">|</span>
            
            <span className="text-xs text-neutral-400 font-mono flex items-center gap-1 whitespace-nowrap" title={data.timestamp}>
              <Clock className="w-3 h-3" />
              {formatTime(data.timestamp)}
            </span>
          </div>
        </div>

        {/* Content Row: Title, Message, Primary Evidence */}
        <div className="flex justify-between items-start gap-4">
          <div className="flex-1 min-w-0">
            <h4 className={`text-sm font-bold leading-tight mb-1 flex items-center gap-2 ${data.state === 'resolved' || data.state === 'suppressed' ? 'text-neutral-500' : 'text-neutral-900'}`}>
              <span className="truncate">{data.title}</span>
              {data.autoActionEnabled && (
                <span className="inline-flex items-center gap-0.5 px-1 py-0.5 rounded bg-neutral-100 text-[9px] text-neutral-500 font-normal uppercase tracking-wide border border-neutral-200 flex-shrink-0" title="Automated Action Available">
                  <ZapIcon className="w-2.5 h-2.5" /> Auto
                </span>
              )}
            </h4>
            <p className={`text-xs leading-relaxed line-clamp-2 ${data.state === 'resolved' ? 'text-neutral-400' : 'text-neutral-500'}`}>
              {data.message}
            </p>
          </div>
          
          {/* Primary Evidence Chip */}
          {data.evidence && (
             <div className="flex-shrink-0 bg-neutral-50 border border-neutral-100 rounded px-2 py-1.5 flex flex-col items-end min-w-[80px]">
                <span className="text-[9px] uppercase font-bold text-neutral-400">{data.evidence.label}</span>
                <span className="text-sm font-mono font-medium text-neutral-900 whitespace-nowrap">{data.evidence.value}<span className="text-xs ml-0.5">{data.evidence.unit}</span></span>
             </div>
          )}
        </div>

        {/* Hover / Detail Layer: Metadata & Actions */}
        <div className="max-h-0 opacity-0 group-hover:max-h-32 group-hover:opacity-100 transition-all duration-300 overflow-hidden flex flex-col gap-2">
           
           {/* Detail Line: Trigger / Threshold / Count */}
           {(data.triggerCondition || data.threshold || data.occurrenceCount) && (
             <div className="pt-2 mt-1 border-t border-neutral-50 flex flex-wrap items-center gap-4 text-[10px] text-neutral-500">
                {data.triggerCondition && (
                  <div className="flex items-center gap-1">
                    <span className="font-bold text-neutral-400 uppercase tracking-wide">Trigger:</span>
                    <span className="font-mono bg-neutral-100 px-1 rounded truncate max-w-[150px]">{data.triggerCondition}</span>
                  </div>
                )}
                {data.occurrenceCount && data.occurrenceCount > 1 && (
                   <div className="flex items-center gap-1">
                    <span className="font-bold text-neutral-400 uppercase tracking-wide">Count:</span>
                    <span className="font-mono bg-neutral-100 px-1 rounded">{data.occurrenceCount}x</span>
                  </div>
                )}
             </div>
           )}

           {/* Action Bar */}
           <div className="flex items-center justify-between pt-1 flex-wrap gap-2">
             <div className="flex items-center gap-2">
                {data.assignee ? (
                   <div className="flex items-center gap-1.5 bg-neutral-100 rounded-full pl-1 pr-2 py-0.5 border border-neutral-200">
                      <div className="w-4 h-4 rounded-full bg-neutral-300 text-[8px] flex items-center justify-center font-bold text-neutral-700">
                        {data.assignee.initials}
                      </div>
                      <span className="text-[10px] font-medium text-neutral-600">Assigned</span>
                   </div>
                ) : (
                  <button 
                    onClick={(e) => { e.stopPropagation(); onAction?.(data.id, 'assign'); }}
                    className="text-[10px] font-bold text-neutral-400 hover:text-neutral-700 transition-colors uppercase tracking-wide flex items-center gap-1 px-1 py-0.5 hover:bg-neutral-50 rounded"
                  >
                    + Assign
                  </button>
                )}
             </div>

             <div className="flex items-center gap-2 ml-auto">
                {data.actions.map((action, idx) => (
                  <button
                    key={idx}
                    onClick={(e) => {
                      e.stopPropagation();
                      onAction?.(data.id, action.intent);
                    }}
                    className={`text-[10px] font-bold uppercase tracking-wide px-3 py-1.5 rounded transition-all shadow-sm active:scale-95
                      ${action.type === 'primary'
                        ? 'bg-neutral-900 text-white border border-neutral-900 hover:bg-neutral-800 hover:shadow-md'
                        : 'bg-white text-neutral-600 border border-neutral-200 hover:border-neutral-300 hover:text-neutral-900 hover:bg-neutral-50'
                      }`}
                  >
                    {action.label}
                  </button>
                ))}
                <button 
                  onClick={(e) => { e.stopPropagation(); }}
                  className="p-1.5 hover:bg-neutral-100 rounded text-neutral-400 hover:text-neutral-600 transition-colors"
                >
                   <MoreHorizontal className="w-3.5 h-3.5" />
                </button>
             </div>
           </div>
        </div>
      </div>
    </div>
  );
};


export default function ScenarioComponent({ data }) {
  if (!data) return null;
  return (
    <div className="h-full w-full flex items-start p-3 overflow-y-auto relative">
      <AlertNotification data={data.data} variant={data.variant} className="w-full" />
    </div>
  );
}
