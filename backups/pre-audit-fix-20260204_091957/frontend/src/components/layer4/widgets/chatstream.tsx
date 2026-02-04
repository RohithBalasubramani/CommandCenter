// @ts-nocheck
'use client';

import * as React from 'react';
import PropTypes from 'prop-types';
import { List as VirtualList } from 'react-window';
import { alpha, styled, useTheme } from '@mui/material/styles';
import {
  Avatar,
  Box,
  Button,
  Chip,
  Collapse,
  Divider,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography
} from '@mui/material';
import SummarizeIcon from '@mui/icons-material/Summarize';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PushPinIcon from '@mui/icons-material/PushPin';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import AddAlertIcon from '@mui/icons-material/AddAlert';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ReplayIcon from '@mui/icons-material/Replay';
import CancelIcon from '@mui/icons-material/Cancel';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import MicNoneIcon from '@mui/icons-material/MicNone';
import SendIcon from '@mui/icons-material/Send';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import MapIcon from '@mui/icons-material/Map';
import TableChartIcon from '@mui/icons-material/TableChart';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import VisibilityIcon from '@mui/icons-material/Visibility';

const ESTIMATED_ROW_HEIGHT = 176;
const VIRTUAL_OVERSCAN = 6;
const MAX_LINE_WIDTH = 720;

const Surface = styled(Paper)(({ theme }) => ({
  position: 'relative',
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  borderRadius: 16,
  padding: theme.spacing(2),
  border: `1px solid ${alpha('#FFFFFF', 0.08)}`,
  background: 'linear-gradient(135deg, rgba(16,32,72,0.92), rgba(8,16,40,0.94))',
  boxShadow: '0 28px 68px rgba(8,16,44,0.52)'
}));

const StreamViewport = styled(Box)(({ theme }) => ({
  position: 'relative',
  flex: 1,
  overflow: 'hidden',
  borderRadius: theme.spacing(2),
  background: alpha('#0a1c40', 0.35),
  border: `1px solid ${alpha('#7c8dff', 0.08)}`
}));

const StreamScroller = styled(Box)(() => ({
  height: '100%',
  overflowY: 'auto',
  overflowX: 'hidden'
}));

const MessageFrame = styled('article')(({ theme, ownerState }) => ({
  position: 'relative',
  display: 'flex',
  flexDirection: 'column',
  alignItems: ownerState.role === 'user' ? 'flex-end' : 'flex-start',
  paddingTop: ownerState.isGrouped ? theme.spacing(1) : theme.spacing(2),
  paddingBottom: theme.spacing(0.5),
  '&:hover .chat-inline-actions': {
    opacity: 1,
    pointerEvents: 'auto',
    transform: 'translateY(-50%) translateX(-4px)'
  }
}));

const Bubble = styled(Paper)(({ ownerState }) => {
  const isUser = ownerState.role === 'user';
  const isSystem = ownerState.role === 'system';
  return {
    position: 'relative',
    maxWidth: MAX_LINE_WIDTH,
    width: '100%',
    borderRadius: 24,
    border: isSystem
      ? '1px solid rgba(255,255,255,0.16)'
      : `1px solid ${alpha('#7c8dff', isUser ? 0.45 : 0.18)}`,
    background: ownerState.hasError
      ? 'linear-gradient(135deg, rgba(255,76,76,0.18), rgba(120,8,20,0.32))'
      : isUser
        ? 'linear-gradient(135deg, rgba(91,147,255,0.92), rgba(38,87,255,0.86))'
        : 'linear-gradient(135deg, rgba(12,24,52,0.88), rgba(10,20,44,0.88))',
    boxShadow: '0 20px 48px rgba(4,12,32,0.55)',
    padding: '16px 24px',
    color: isUser ? '#f8fafc' : 'rgba(226,232,240,0.92)',
    backdropFilter: 'blur(18px)'
  };
});

const ComposerSurface = styled('form')(({ theme }) => ({
  position: 'sticky',
  bottom: 0,
  marginTop: theme.spacing(2),
  borderRadius: 18,
  border: `1px solid ${alpha('#7c8dff', 0.14)}`,
  background: 'linear-gradient(135deg, rgba(12,24,56,0.88), rgba(10,20,44,0.92))',
  boxShadow: '0 18px 40px rgba(8,14,42,0.55)',
  padding: theme.spacing(1.5, 1.75),
  display: 'grid',
  gridTemplateColumns: 'minmax(0,1fr)',
  gap: theme.spacing(1),
  zIndex: 2
}));

const InlineActionBar = styled(Box)(({ theme }) => ({
  position: 'absolute',
  right: -56,
  top: '50%',
  transform: 'translateY(-50%)',
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  opacity: 0,
  pointerEvents: 'none',
  transition: 'opacity 120ms ease, transform 120ms ease',
  [theme.breakpoints.down('md')]: {
    position: 'static',
    flexDirection: 'row',
    justifyContent: 'flex-end',
    transform: 'none',
    marginTop: theme.spacing(1),
    opacity: 1,
    pointerEvents: 'auto'
  },
  '&.visible': {
    opacity: 1,
    pointerEvents: 'auto'
  }
}));

const inlineWidgetBadgeColors = {
  live: '#38bdf8',
  cached: '#facc15',
  stale: '#f97316',
  error: '#ef4444'
};

function getWidgetIcon(kind) {
  switch (kind) {
    case 'map':
    case 'lane-map':
      return <MapIcon fontSize="small" />;
    case 'table':
      return <TableChartIcon fontSize="small" />;
    case 'trend':
    case 'chart':
      return <ShowChartIcon fontSize="small" />;
    default:
      return <VisibilityIcon fontSize="small" />;
  }
}

function MessageActions({
  actions,
  onSummarize,
  onExplain,
  onPin,
  onExport,
  onOpenInLens,
  onCreateAlert,
  onCopy,
  onRegenerate
}) {
  const className = actions?.visible ? 'chat-inline-actions visible' : 'chat-inline-actions';
  return (
    <InlineActionBar className={className}>
      {actions?.canSummarize && (
        <Tooltip title="Summarize" placement="left">
          <InlineActionButton size="small" onClick={onSummarize}>
            <SummarizeIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canExplain && (
        <Tooltip title="Explain" placement="left">
          <InlineActionButton size="small" onClick={onExplain}>
            <PsychologyIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canPin && (
        <Tooltip title="Pin" placement="left">
          <InlineActionButton size="small" onClick={onPin}>
            <PushPinIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canExport && (
        <Tooltip title="Export" placement="left">
          <InlineActionButton size="small" onClick={onExport}>
            <FileDownloadIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canOpenInLens && (
        <Tooltip title="Open in Lens" placement="left">
          <InlineActionButton size="small" onClick={onOpenInLens}>
            <OpenInNewIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canCreateAlert && (
        <Tooltip title="Create alert / WO" placement="left">
          <InlineActionButton size="small" onClick={onCreateAlert}>
            <AddAlertIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canCopy && (
        <Tooltip title="Copy" placement="left">
          <InlineActionButton size="small" onClick={onCopy}>
            <ContentCopyIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
      {actions?.canRegenerate && (
        <Tooltip title="Regenerate" placement="left">
          <InlineActionButton size="small" onClick={onRegenerate}>
            <ReplayIcon fontSize="small" />
          </InlineActionButton>
        </Tooltip>
      )}
    </InlineActionBar>
  );
}

MessageActions.propTypes = {
  actions: PropTypes.object,
  onSummarize: PropTypes.func,
  onExplain: PropTypes.func,
  onPin: PropTypes.func,
  onExport: PropTypes.func,
  onOpenInLens: PropTypes.func,
  onCreateAlert: PropTypes.func,
  onCopy: PropTypes.func,
  onRegenerate: PropTypes.func
};

const InlineActionButton = styled(IconButton)(() => ({
  backgroundColor: 'rgba(10,28,64,0.65)',
  borderRadius: 999,
  color: 'rgba(226,232,240,0.88)',
  border: '1px solid rgba(124,141,255,0.22)',
  boxShadow: '0 8px 18px rgba(6,14,40,0.4)',
  '&:hover': {
    backgroundColor: 'rgba(15,42,96,0.8)'
  }
}));

const AttachmentChip = styled(Paper)(() => ({
  display: 'inline-flex',
  alignItems: 'center',
  padding: '6px 14px',
  borderRadius: 999,
  background: 'rgba(15,29,59,0.75)',
  border: '1px solid rgba(124,141,255,0.24)',
  color: 'rgba(226,232,240,0.9)',
  gap: 10
}));

function ProvenanceChip({ provenance }) {
  const theme = useTheme();
  const background = alpha('#0f1d3b', 0.74);
  return (
    <Chip
      size="small"
      label={
        <Stack direction="row" spacing={0.75} alignItems="center">
          <Typography variant="caption" sx={{ color: 'inherit', fontWeight: 500 }}>
            {provenance.domain}
          </Typography>
          <Typography variant="caption" sx={{ color: 'inherit', opacity: 0.85 }}>
            {provenance.sources} sources
          </Typography>
          <Typography variant="caption" sx={{ color: 'inherit', opacity: 0.85 }}>
            {provenance.state === 'live'
              ? `Live (${provenance.freshness_s}s)`
              : `${provenance.state.charAt(0).toUpperCase()}${provenance.state.slice(1)} (${provenance.freshness_s}s)`}
          </Typography>
        </Stack>
      }
      sx={{
        borderRadius: 999,
        paddingY: 0.25,
        paddingX: 0.5,
        backgroundColor: background,
        border: `1px solid ${alpha(theme.palette.primary.light, 0.32)}`,
        color: 'rgba(226,232,240,0.9)',
        '& .MuiChip-label': {
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing(1)
        }
      }}
    />
  );
}

ProvenanceChip.propTypes = {
  provenance: PropTypes.shape({
    domain: PropTypes.string.isRequired,
    sources: PropTypes.number,
    freshness_s: PropTypes.number,
    state: PropTypes.string
  }).isRequired
};

function MessageHeader({ role, agent, timestamp, scope, status }) {
  const theme = useTheme();
  if (role === 'system') {
    return null;
  }
  return (
    <Stack
      direction="row"
      alignItems="center"
      spacing={1}
      justifyContent="space-between"
      sx={{ mb: 1 }}
    >
      <Stack direction="row" spacing={1.25} alignItems="center">
        {role === 'agent' ? (
          <Chip
            size="small"
            label={
              <Stack direction="row" spacing={0.75} alignItems="center">
                <Avatar
                  sx={{
                    width: 20,
                    height: 20,
                    fontSize: 11,
                    backgroundColor: alpha('#7c8dff', 0.45),
                    color: '#f8fafc'
                  }}
                >
                  {agent?.acronym || 'Neuract'}
                </Avatar>
                <Typography variant="caption" sx={{ color: 'inherit', fontWeight: 600 }}>
                  {agent?.label || 'Neuract'}
                </Typography>
                {status && (
                  <Typography variant="caption" sx={{ opacity: 0.7, color: 'inherit' }}>
                    {status}
                  </Typography>
                )}
              </Stack>
            }
            sx={{
              borderRadius: 999,
              backgroundColor: alpha('#0f1d3b', 0.7),
              color: 'rgba(241,245,249,0.9)',
              border: `1px solid ${alpha(theme.palette.primary.light, 0.4)}`,
              '& .MuiChip-label': { px: 0.75, py: 0.5 }
            }}
          />
        ) : (
          <Typography variant="caption" sx={{ color: alpha('#f8fafc', 0.85) }}>
            You
          </Typography>
        )}
        {scope && (
          <Chip
            size="small"
            label={scope}
            sx={{
              backgroundColor: alpha('#0f1d3b', 0.65),
              color: 'rgba(226,232,240,0.88)',
              border: `1px solid ${alpha('#7c8dff', 0.22)}`,
              borderRadius: 999
            }}
          />
        )}
      </Stack>
      {timestamp && (
        <Tooltip title={timestamp.absolute || ''} placement="left">
          <Typography variant="caption" sx={{ color: alpha('#f8fafc', 0.72) }}>
            {timestamp.relative}
          </Typography>
        </Tooltip>
      )}
    </Stack>
  );
}

MessageHeader.propTypes = {
  role: PropTypes.string.isRequired,
  agent: PropTypes.shape({
    acronym: PropTypes.string,
    label: PropTypes.string
  }),
  timestamp: PropTypes.shape({
    relative: PropTypes.string,
    absolute: PropTypes.string
  }),
  scope: PropTypes.string,
  status: PropTypes.string
};

const TracePanel = styled(Paper)(({ theme }) => ({
  marginTop: theme.spacing(1.5),
  padding: theme.spacing(2),
  borderRadius: 18,
  border: `1px solid ${alpha('#7c8dff', 0.14)}`,
  background: alpha('#081432', 0.88),
  color: 'rgba(203,213,225,0.92)',
  display: 'grid',
  gap: theme.spacing(1)
}));

function TraceDetails({ trace, onExplain, explained, isExplaining }) {
  const [open, setOpen] = React.useState(false);

  if (!trace) {
    return null;
  }

  const handleToggle = () => {
    setOpen(value => !value);
  };

  return (
    <TracePanel role="region" aria-live="polite">
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Stack direction="row" spacing={1} alignItems="center">
          <Chip
            size="small"
            icon={<InfoOutlinedIcon fontSize="small" />}
            label="Trace"
            sx={{
              height: 24,
              borderRadius: 999,
              backgroundColor: 'rgba(12,24,54,0.76)',
              border: '1px solid rgba(124,141,255,0.32)',
              color: 'rgba(226,232,240,0.88)'
            }}
          />
          <Typography variant="body2" sx={{ color: 'rgba(226,232,240,0.88)' }}>
            {trace.summary || 'Route, tools, scopes, freshness'}
          </Typography>
        </Stack>
        <Stack direction="row" spacing={1}>
          <Button
            size="small"
            variant="text"
            onClick={() => onExplain?.(trace)}
            disabled={isExplaining}
            sx={{ color: '#7c91ff' }}
          >
            Explain
          </Button>
          <IconButton
            size="small"
            onClick={handleToggle}
            sx={{
              color: 'rgba(226,232,240,0.88)',
              border: '1px solid rgba(124,141,255,0.22)',
              borderRadius: 999,
              '&:hover': {
                backgroundColor: 'rgba(12,24,52,0.6)'
              }
            }}
          >
            {open ? <KeyboardArrowDownIcon /> : <KeyboardArrowRightIcon />}
          </IconButton>
        </Stack>
      </Stack>
      <Collapse in={open} timeout="auto" unmountOnExit>
        <Stack spacing={1.25} sx={{ mt: 1.5 }}>
          {trace.route && (
            <Stack spacing={0.5}>
              <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                Route
              </Typography>
              <Typography variant="body2">{trace.route.join(' \u2192 ')}</Typography>
            </Stack>
          )}
          {trace.tools && (
            <Stack spacing={0.5}>
              <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                Tools
              </Typography>
              <Typography variant="body2">{trace.tools.join(', ')}</Typography>
            </Stack>
          )}
          {trace.scopes && (
            <Stack spacing={0.5}>
              <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                Scopes
              </Typography>
              <Typography variant="body2">{trace.scopes.join(' \u2022 ')}</Typography>
            </Stack>
          )}
          {(trace.freshness || trace.cost) && (
            <Stack direction="row" spacing={2}>
              {trace.freshness && (
                <Stack spacing={0.5}>
                  <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                    Freshness
                  </Typography>
                  <Typography variant="body2">{trace.freshness}</Typography>
                </Stack>
              )}
              {trace.cost && (
                <Stack spacing={0.5}>
                  <Typography variant="caption" sx={{ color: 'rgba(148,163,184,0.9)' }}>
                    Cost / Time
                  </Typography>
                  <Typography variant="body2">{trace.cost}</Typography>
                </Stack>
              )}
            </Stack>
          )}
          {explained && (
            <Box
              sx={{
                mt: 1.5,
                borderRadius: 14,
                border: '1px solid rgba(124,141,255,0.22)',
                backgroundColor: 'rgba(12,24,54,0.75)',
                p: 1.5
              }}
            >
              <Typography variant="subtitle2" sx={{ color: 'rgba(226,232,240,0.92)' }}>
                Explanation
              </Typography>
              <Divider sx={{ borderColor: 'rgba(124,141,255,0.2)', my: 1 }} />
              <Stack component="ul" spacing={1} sx={{ pl: 2, color: 'rgba(226,232,240,0.88)' }}>
                {explained.map(item => (
                  <Typography key={item} component="li" variant="body2">
                    {item}
                  </Typography>
                ))}
              </Stack>
            </Box>
          )}
        </Stack>
      </Collapse>
    </TracePanel>
  );
}

TraceDetails.propTypes = {
  trace: PropTypes.object,
  onExplain: PropTypes.func,
  explained: PropTypes.arrayOf(PropTypes.string),
  isExplaining: PropTypes.bool
};

const WidgetStubSurface = styled(Paper)(({ theme, ownerState }) => ({
  marginTop: theme.spacing(1.5),
  borderRadius: 16,
  padding: theme.spacing(1.5),
  display: 'grid',
  gap: theme.spacing(1),
  border: `1px solid ${alpha(ownerState.badgeColor, 0.4)}`,
  background: alpha('#0a1c40', 0.78),
  boxShadow: '0 16px 34px rgba(6,12,32,0.44)'
}));

function InlineWidgetStub({ widget, onExpand }) {
  const badgeColor =
    inlineWidgetBadgeColors[widget.state?.toLowerCase?.() || 'cached'] || '#94a3b8';
  return (
    <WidgetStubSurface role="group" aria-label={`${widget.kind} preview`} ownerState={{ badgeColor }}>
      <Stack direction="row" alignItems="center" spacing={1.25}>
        <Chip
          size="small"
          icon={getWidgetIcon(widget.kind)}
          label={widget.title || widget.kind}
          sx={{
            borderRadius: 999,
            height: 28,
            border: `1px solid ${alpha(badgeColor, 0.6)}`,
            backgroundColor: alpha(badgeColor, 0.14),
            color: 'rgba(226,232,240,0.9)',
            '& .MuiChip-label': {
              px: 0.75
            }
          }}
        />
        <Chip
          size="small"
          label={widget.state ? widget.state.toUpperCase() : 'CACHED'}
          sx={{
            borderRadius: 999,
            backgroundColor: alpha('#0f1d3b', 0.72),
            color: 'rgba(226,232,240,0.88)',
            border: `1px solid ${alpha(badgeColor, 0.42)}`
          }}
        />
      </Stack>
      {widget.preview && (
        <Box
          sx={{
            mt: 1,
            display: 'grid',
            gap: 0.75,
            color: 'rgba(226,232,240,0.88)',
            fontSize: 13
          }}
        >
          {widget.preview.map((line, index) => (
            <Typography key={index} variant="body2" sx={{ opacity: 0.85 }}>
              {line}
            </Typography>
          ))}
        </Box>
      )}
      <Button
        variant="contained"
        size="small"
        onClick={() => onExpand?.(widget)}
        sx={{
          justifySelf: 'flex-start',
          borderRadius: 999,
          backgroundColor: alpha('#7c8dff', 0.85),
          boxShadow: '0 16px 32px rgba(124,141,255,0.38)',
          '&:hover': {
            backgroundColor: alpha('#8d9cff', 0.95)
          }
        }}
      >
        Expand in Dock
      </Button>
    </WidgetStubSurface>
  );
}

InlineWidgetStub.propTypes = {
  widget: PropTypes.shape({
    anchor_id: PropTypes.string,
    kind: PropTypes.string,
    state: PropTypes.string,
    title: PropTypes.string,
    preview: PropTypes.arrayOf(PropTypes.string)
  }).isRequired,
  onExpand: PropTypes.func
};

function AttachmentRow({ attachments }) {
  if (!attachments || attachments.length === 0) {
    return null;
  }
  return (
    <Stack direction="row" spacing={1} sx={{ mt: 1.25, flexWrap: 'wrap' }}>
      {attachments.map(file => (
        <AttachmentChip key={file.url}>
          <Typography variant="caption" sx={{ textTransform: 'uppercase', fontWeight: 600 }}>
            {file.type}
          </Typography>
          <Divider flexItem orientation="vertical" sx={{ borderColor: 'rgba(124,141,255,0.22)' }} />
          <Typography variant="body2" sx={{ color: 'rgba(226,232,240,0.9)' }}>
            {file.name}
          </Typography>
        </AttachmentChip>
      ))}
    </Stack>
  );
}

AttachmentRow.propTypes = {
  attachments: PropTypes.arrayOf(
    PropTypes.shape({
      type: PropTypes.string,
      url: PropTypes.string.isRequired,
      name: PropTypes.string
    })
  )
};

function ErrorBanner({ error, onRetry, onViewTrace }) {
  if (!error) {
    return null;
  }
  return (
    <Paper
      role="alert"
      sx={{
        mt: 1.75,
        borderRadius: 14,
        padding: 1.75,
        background: 'linear-gradient(135deg, rgba(255,76,76,0.16), rgba(92,12,22,0.4))',
        border: '1px solid rgba(239,68,68,0.55)',
        color: 'rgba(254,226,226,0.9)',
        display: 'flex',
        flexDirection: 'column',
        gap: 1
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center">
        <ErrorOutlineIcon fontSize="small" />
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {error.title || 'Error'}
        </Typography>
      </Stack>
      <Typography variant="body2" sx={{ opacity: 0.9 }}>
        {error.detail}
      </Typography>
      <Stack direction="row" spacing={1}>
        {onRetry && (
          <Button variant="contained" size="small" onClick={onRetry}>
            Retry
          </Button>
        )}
        {onViewTrace && (
          <Button variant="outlined" size="small" onClick={onViewTrace} sx={{ color: '#e87888' }}>
            View trace
          </Button>
        )}
      </Stack>
    </Paper>
  );
}

ErrorBanner.propTypes = {
  error: PropTypes.shape({
    title: PropTypes.string,
    detail: PropTypes.string
  }),
  onRetry: PropTypes.func,
  onViewTrace: PropTypes.func
};

const ReplyAnchor = styled(Button)(({ theme }) => ({
  alignSelf: 'flex-start',
  marginTop: theme.spacing(0.5),
  padding: theme.spacing(0.5, 1.5),
  borderRadius: 999,
  border: `1px solid ${alpha('#7c8dff', 0.26)}`,
  color: 'rgba(148,163,184,0.95)',
  backgroundColor: alpha('#0a1c40', 0.5),
  '&:hover': {
    backgroundColor: alpha('#0f2a60', 0.72),
    borderColor: alpha('#9ab0ff', 0.45)
  }
}));

const defaultMessages = [
  {
    message_id: 'demo-1',
    role: 'agent',
    agent: { acronym: 'Neuract', label: 'Neuract' },
    agent_status: 'Live',
    timestamp: { relative: '12s ago', absolute: '2025-10-27T14:01:18Z' },
    content:
      '<p>Line 3 OEE dips are linked to station F micro-stops. 62% of the loss stems from manual verifies triggered by the vision camera drop.</p><p><strong>Recommended:</strong></p><ul><li>Recalibrate the bin-pick arm (12 min)</li><li>Schedule lens clean before shift change</li></ul>',
    provenance: [
      { domain: 'MES', sources: 2, freshness_s: 12, state: 'live' },
      { domain: 'Vision QA', sources: 1, freshness_s: 38, state: 'cached' }
    ],
    trace: {
      route: ['Neuract Orchestrator', 'MESAgent', 'VisionAgent'],
      tools: ['sql.read', 'image.annotate'],
      scopes: ['plant:04', 'line:3'],
      freshness: '12s',
      cost: '~0.8k tokens'
    },
    actions: {
      canSummarize: true,
      canExplain: true,
      canPin: true,
      canExport: true,
      canOpenInLens: true,
      canCreateAlert: true,
      canCopy: true,
      canRegenerate: true
    },
    inline_widgets: [
      {
        anchor_id: 'w-1',
        kind: 'trend',
        state: 'live',
        title: 'Micro-stop trend',
        preview: ['Station F micro-stops +18% vs 7d', 'Top driver: Vision retry loop']
      }
    ]
  },
  {
    message_id: 'demo-2',
    role: 'user',
    timestamp: { relative: 'Just now', absolute: '2025-10-27T14:01:22Z' },
    content: 'Show me the root cause timeline and export to maintenance.'
  }
];

function MessageBubble({
  message,
  isStreaming,
  onCancel,
  onRetry,
  onSummarize,
  onExplain,
  onPin,
  onExport,
  onOpenInLens,
  onCreateAlert,
  onCopy,
  onRegenerate,
  onExpandWidget,
  onReply,
  onTraceExplain
}) {
  const [explained, setExplained] = React.useState(null);
  const [explaining, setExplaining] = React.useState(false);

  const handleExplain = async () => {
    if (!onTraceExplain || explaining) {
      return;
    }
    setExplaining(true);
    try {
      const result = await onTraceExplain(message);
      setExplained(result || null);
    } catch (error) {
      console.error('Explain trace failed', error);
    } finally {
      setExplaining(false);
    }
  };

  if (message.role === 'system') {
    return (
      <MessageFrame ownerState={{ role: 'system' }}>
        <Divider sx={{ my: 2, borderColor: 'rgba(124,141,255,0.18)' }}>
          <Chip
            size="small"
            label={message.content}
            sx={{
              borderRadius: 999,
              backgroundColor: 'rgba(12,24,52,0.72)',
              color: 'rgba(226,232,240,0.86)'
            }}
          />
        </Divider>
      </MessageFrame>
    );
  }

  return (
    <MessageFrame ownerState={{ role: message.role }}>
      <Bubble elevation={0} ownerState={{ role: message.role, hasError: Boolean(message.error) }}>
        <MessageHeader
          role={message.role}
          timestamp={message.timestamp}
          scope={message.scope}
          status={message.agent_status}
          agent={message.agent}
        />
        <Box
          sx={{
            fontSize: 15,
            lineHeight: 1.6,
            color: message.role === 'user' ? '#f8fafc' : 'rgba(226,232,240,0.88)',
            '& pre': {
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace',
              fontSize: 13,
              backgroundColor: 'rgba(8,18,48,0.82)',
              padding: 1.25,
              borderRadius: 12,
              overflowX: 'auto'
            },
            '& code': {
              fontSize: 13,
              backgroundColor: 'rgba(8,18,48,0.72)',
              padding: '2px 6px',
              borderRadius: 8
            },
            '& table': {
              width: '100%',
              borderCollapse: 'collapse',
              marginTop: 12
            },
            '& th, & td': {
              borderBottom: '1px solid rgba(124,141,255,0.12)',
              padding: '8px 12px',
              textAlign: 'left'
            },
            '& th': {
              backgroundColor: 'rgba(12,24,52,0.65)'
            }
          }}
          dangerouslySetInnerHTML={{ __html: message.markup || message.content }}
        />

        <AttachmentRow attachments={message.attachments} />

        {message.provenance && message.provenance.length > 0 && (
          <Stack
            direction="row"
            spacing={0.75}
            alignItems="center"
            sx={{ mt: 1.75, flexWrap: 'wrap', rowGap: 1 }}
          >
            {message.provenance.map(item => (
              <ProvenanceChip key={`${item.domain}-${item.state}`} provenance={item} />
            ))}
          </Stack>
        )}

        {message.inline_widgets &&
          message.inline_widgets.map(widget => (
            <InlineWidgetStub key={widget.anchor_id} widget={widget} onExpand={onExpandWidget} />
          ))}

        {message.trace && (
          <TraceDetails
            trace={message.trace}
            onExplain={handleExplain}
            explained={explained}
            isExplaining={explaining}
          />
        )}

        {message.error && (
          <ErrorBanner error={message.error} onRetry={() => onRetry?.(message)} />
        )}

        {isStreaming && (
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1.5 }}>
            <LinearProgress
              sx={{
                flex: 1,
                height: 4,
                borderRadius: 99,
                backgroundColor: 'rgba(124,141,255,0.12)',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, rgba(124,141,255,0.8), rgba(92,211,242,0.9))'
                }
              }}
            />
            <Button
              size="small"
              variant="text"
              onClick={() => onCancel?.(message)}
              startIcon={<CancelIcon fontSize="small" />}
              sx={{ color: '#fca5a5' }}
            >
              Cancel
            </Button>
          </Stack>
        )}

        {!isStreaming && message.canRetry && (
          <Button
            size="small"
            variant="outlined"
            onClick={() => onRetry?.(message)}
            startIcon={<ReplayIcon fontSize="small" />}
            sx={{
              mt: 1.5,
              color: '#bcd0ff',
              borderColor: 'rgba(124,141,255,0.32)'
            }}
          >
            Retry
          </Button>
        )}
      </Bubble>
      <ReplyAnchor size="small" onClick={() => onReply?.(message)}>
        Reply to this
      </ReplyAnchor>
      <MessageActions
        actions={message.actions}
        onSummarize={() => onSummarize?.(message)}
        onExplain={() => onExplain?.(message)}
        onPin={() => onPin?.(message)}
        onExport={() => onExport?.(message)}
        onOpenInLens={() => onOpenInLens?.(message)}
        onCreateAlert={() => onCreateAlert?.(message)}
        onCopy={() => onCopy?.(message)}
        onRegenerate={() => onRegenerate?.(message)}
      />
    </MessageFrame>
  );
}

MessageBubble.propTypes = {
  message: PropTypes.object.isRequired,
  isStreaming: PropTypes.bool,
  onCancel: PropTypes.func,
  onRetry: PropTypes.func,
  onSummarize: PropTypes.func,
  onExplain: PropTypes.func,
  onPin: PropTypes.func,
  onExport: PropTypes.func,
  onOpenInLens: PropTypes.func,
  onCreateAlert: PropTypes.func,
  onCopy: PropTypes.func,
  onRegenerate: PropTypes.func,
  onExpandWidget: PropTypes.func,
  onReply: PropTypes.func,
  onTraceExplain: PropTypes.func
};

const SCOPE_OPTIONS = ['Org', 'Plant 04', 'Line 3'];

function useDraftMemory(threadId, initialDraft = '') {
  const [draft, setDraft] = React.useState(() => {
    if (typeof window === 'undefined') {
      return initialDraft;
    }
    const key = `chat-draft::${threadId}`;
    return window.sessionStorage.getItem(key) || initialDraft;
  });

  React.useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const key = `chat-draft::${threadId}`;
    try {
      window.sessionStorage.setItem(key, draft);
    } catch (error) {
      console.warn('Unable to persist draft', error);
    }
  }, [draft, threadId]);

  return [draft, setDraft];
}

function detectAttachmentFromPaste(event, onAttachmentOffer) {
  const { clipboardData } = event;
  if (!clipboardData || !onAttachmentOffer) {
    return;
  }
  const items = clipboardData.items || [];
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    if (item.kind === 'file') {
      const file = item.getAsFile();
      if (file) {
        onAttachmentOffer({ name: file.name, type: file.type });
        break;
      }
    }
    if (item.kind === 'string') {
      item.getAsString(value => {
        if (/data:text\/csv/i.test(value) || value.includes(',') || value.includes(';')) {
          onAttachmentOffer({ name: 'clipboard.csv', type: 'text/csv' });
        }
      });
    }
  }
}

function ChatComposer({
  onSend,
  onAttach,
  onScopeChange,
  scope,
  scopeOptions,
  promptHintsEnabled,
  onTogglePromptHints,
  onOfferAttachment,
  isSending,
  threadId,
  onComposerFocus
}) {
  const [draft, setDraft] = useDraftMemory(threadId);
  const [lastSentAt, setLastSentAt] = React.useState(0);
  const inputRef = React.useRef(null);

  const handleKeyDown = event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    const now = performance.now();
    if (now - lastSentAt < 300) {
      return;
    }
    const trimmed = draft.trim();
    if (!trimmed) {
      return;
    }
    onSend?.(trimmed);
    setDraft('');
    setLastSentAt(now);
  };

  const handlePaste = event => {
    detectAttachmentFromPaste(event, onOfferAttachment);
  };

  return (
    <ComposerSurface
      role="form"
      aria-label="Compose message"
      onSubmit={event => {
        event.preventDefault();
        handleSend();
      }}
    >
      <TextField
        inputRef={inputRef}
        value={draft}
        onChange={event => setDraft(event.target.value)}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onFocus={onComposerFocus}
        placeholder="Type a prompt. Need context? Try “Summarize EMS incidents in the last 2 hours.”"
        variant="outlined"
        multiline
        minRows={1}
        maxRows={4}
        fullWidth
        aria-multiline="true"
        sx={{
          '& .MuiOutlinedInput-root': {
            borderRadius: 18,
            background: 'linear-gradient(135deg, rgba(12,24,52,0.92), rgba(8,18,40,0.92))',
            border: '1px solid rgba(124,141,255,0.22)',
            '& fieldset': { border: 'none' },
            '&.Mui-focused': {
              boxShadow: '0 0 0 2px rgba(124,141,255,0.24)'
            }
          },
          '& .MuiOutlinedInput-input': {
            color: 'rgba(226,232,240,0.92)',
            fontSize: 15
          }
        }}
      />
      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Attach (coming soon)">
            <span>
              <IconButton
                size="small"
                onClick={onAttach}
                disabled={!onAttach}
                aria-disabled={!onAttach}
                sx={{
                  borderRadius: 12,
                  backgroundColor: 'rgba(12,24,52,0.72)',
                  color: 'rgba(226,232,240,0.8)',
                  border: '1px solid rgba(124,141,255,0.24)'
                }}
              >
                <AttachFileIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          {scopeOptions && scopeOptions.length > 0 && (
            <Chip
              label={scope || scopeOptions[0]}
              onClick={() => onScopeChange?.(scope)}
              sx={{
                borderRadius: 999,
                backgroundColor: 'rgba(12,24,52,0.78)',
                color: 'rgba(226,232,240,0.88)',
                border: '1px solid rgba(124,141,255,0.24)'
              }}
            />
          )}
          <Button
            size="small"
            variant={promptHintsEnabled ? 'contained' : 'outlined'}
            onClick={onTogglePromptHints}
            sx={{
              borderRadius: 999,
              color: promptHintsEnabled ? '#0b122c' : 'rgba(226,232,240,0.88)',
              backgroundColor: promptHintsEnabled ? '#7c8dff' : 'rgba(12,24,52,0.72)',
              borderColor: 'rgba(124,141,255,0.35)'
            }}
          >
            Prompt hints
          </Button>
        </Stack>
        <Stack direction="row" spacing={1.25} alignItems="center">
          <Tooltip title="Voice (coming soon)">
            <span>
              <IconButton
                size="small"
                disabled
                sx={{
                  borderRadius: 12,
                  backgroundColor: 'rgba(12,24,52,0.64)',
                  color: 'rgba(100,116,139,0.85)',
                  border: '1px dashed rgba(148,163,184,0.25)'
                }}
              >
                <MicNoneIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Button
            type="submit"
            variant="contained"
            endIcon={<SendIcon fontSize="small" />}
            disabled={isSending}
            sx={{
              borderRadius: 999,
              px: 3,
              background: 'linear-gradient(135deg, rgba(124,141,255,0.95), rgba(96,165,250,0.9))',
              boxShadow: '0 18px 44px rgba(124,141,255,0.35)'
            }}
          >
            Send
          </Button>
        </Stack>
      </Stack>
    </ComposerSurface>
  );
}

ChatComposer.propTypes = {
  onSend: PropTypes.func,
  onAttach: PropTypes.func,
  onScopeChange: PropTypes.func,
  scope: PropTypes.string,
  scopeOptions: PropTypes.arrayOf(PropTypes.string),
  promptHintsEnabled: PropTypes.bool,
  onTogglePromptHints: PropTypes.func,
  onOfferAttachment: PropTypes.func,
  isSending: PropTypes.bool,
  threadId: PropTypes.string,
  onComposerFocus: PropTypes.func
};

function useVirtualMeasurements(count, estimated = ESTIMATED_ROW_HEIGHT) {
  const listRef = React.useRef(null);
  const sizeMapRef = React.useRef(new Map());

  const getSize = React.useCallback(
    index => sizeMapRef.current.get(index) || estimated,
    [estimated]
  );

  const setMeasuredHeight = React.useCallback(
    (index, size) => {
      const current = sizeMapRef.current.get(index);
      if (current !== size) {
        sizeMapRef.current.set(index, size);
        listRef.current?.resetAfterIndex(index);
      }
    },
    []
  );

  React.useEffect(() => {
    const map = sizeMapRef.current;
    for (let index = count; index < map.size; index += 1) {
      map.delete(index);
    }
  }, [count]);

  return { listRef, getSize, setMeasuredHeight };
}

function MessageRow({ index, style, data, setMeasuredHeight }) {
  const { messages, handlers, streamingMessageId } = data;
  const message = messages[index];
  const rowRef = React.useRef(null);

  React.useLayoutEffect(() => {
    if (!rowRef.current) {
      return undefined;
    }
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const height = entry.contentRect.height + 8;
        setMeasuredHeight(index, height);
      }
    });
    observer.observe(rowRef.current);
    return () => observer.disconnect();
  }, [index, setMeasuredHeight]);

  return (
    <Box ref={rowRef} style={style} sx={{ px: { xs: 1, md: 2 } }}>
      <MessageBubble
        message={message}
        isStreaming={streamingMessageId === message.message_id}
        onCancel={handlers.onCancel}
        onRetry={handlers.onRetry}
        onSummarize={handlers.onSummarize}
        onExplain={handlers.onExplain}
        onPin={handlers.onPin}
        onExport={handlers.onExport}
        onOpenInLens={handlers.onOpenInLens}
        onCreateAlert={handlers.onCreateAlert}
        onCopy={handlers.onCopy}
        onRegenerate={handlers.onRegenerate}
        onExpandWidget={handlers.onExpandWidget}
        onReply={handlers.onReply}
        onTraceExplain={handlers.onTraceExplain}
      />
    </Box>
  );
}

MessageRow.propTypes = {
  index: PropTypes.number.isRequired,
  style: PropTypes.object.isRequired,
  data: PropTypes.shape({
    messages: PropTypes.array.isRequired,
    handlers: PropTypes.object,
    streamingMessageId: PropTypes.string
  }).isRequired,
  setMeasuredHeight: PropTypes.func.isRequired
};

function ChatStream({
  messages: externalMessages = defaultMessages,
  threadId = 'demo-thread',
  onSendMessage,
  onCancelStream,
  onRetry,
  onSummarize,
  onExplain,
  onPin,
  onExport,
  onOpenInLens,
  onCreateAlert,
  onCopy,
  onRegenerate,
  onExpandWidget,
  onReply,
  onTraceExplain,
  scope,
  scopeOptions = SCOPE_OPTIONS,
  onScopeChange,
  promptHintsEnabled = false,
  onTogglePromptHints,
  onOfferAttachment,
  isSending = false,
  streamingMessageId,
  onComposerFocus
}) {
  const streamRef = React.useRef(null);
  const [viewportHeight, setViewportHeight] = React.useState(0);
  const [messages, setMessages] = React.useState(externalMessages);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const [isNearBottom, setIsNearBottom] = React.useState(true);

  React.useEffect(() => {
    setMessages(externalMessages);
  }, [externalMessages]);

  const totalMessages = messages.length;
  const { listRef, getSize, setMeasuredHeight } = useVirtualMeasurements(totalMessages);

  const handleSend = React.useCallback(
    text => {
      if (onSendMessage) {
        onSendMessage(text);
        return;
      }
      const userMessage = {
        message_id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: { relative: 'Just now', absolute: new Date().toISOString() },
        actions: {}
      };
      setMessages(current => [...current, userMessage]);
    },
    [onSendMessage]
  );

  const handleScroll = React.useCallback(() => {
    const node = streamRef.current;
    if (!node) {
      return;
    }
    const { scrollTop, scrollHeight, clientHeight } = node;
    const atBottom = scrollHeight - (scrollTop + clientHeight) < 32;
    setIsNearBottom(atBottom);
    setAutoScroll(atBottom);
  }, []);

  React.useEffect(() => {
    const node = streamRef.current;
    if (!node) {
      return undefined;
    }
    node.addEventListener('scroll', handleScroll, { passive: true });
    return () => node.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  React.useLayoutEffect(() => {
    const node = streamRef.current;
    if (!node) {
      return undefined;
    }
    setViewportHeight(node.clientHeight);
    const observer = new ResizeObserver(entries => {
      const entry = entries[0];
      if (entry) {
        setViewportHeight(entry.contentRect.height);
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    if (!viewportHeight) {
      return;
    }
    listRef.current?.resetAfterIndex(0, true);
  }, [viewportHeight, listRef]);

  React.useEffect(() => {
    if (!autoScroll) {
      return;
    }
    listRef.current?.scrollToItem(totalMessages - 1);
  }, [autoScroll, totalMessages, listRef, messages]);

  const handlers = React.useMemo(
    () => ({
      onCancel: message => onCancelStream?.(message),
      onRetry: message => onRetry?.(message),
      onSummarize: message => onSummarize?.(message),
      onExplain: message => onExplain?.(message),
      onPin: message => onPin?.(message),
      onExport: message => onExport?.(message),
      onOpenInLens: message => onOpenInLens?.(message),
      onCreateAlert: message => onCreateAlert?.(message),
      onCopy: message => onCopy?.(message),
      onRegenerate: message => onRegenerate?.(message),
      onExpandWidget: widget => onExpandWidget?.(widget),
      onReply: message => onReply?.(message),
      onTraceExplain: message => onTraceExplain?.(message)
    }),
    [
      onCancelStream,
      onRetry,
      onSummarize,
      onExplain,
      onPin,
      onExport,
      onOpenInLens,
      onCreateAlert,
      onCopy,
      onRegenerate,
      onExpandWidget,
      onReply,
      onTraceExplain
    ]
  );

  return (
    <Surface elevation={0}>
      <StreamViewport>
        <StreamScroller ref={streamRef} role="log" aria-live="polite">
          {totalMessages === 0 ? (
            <Stack
              alignItems="center"
              justifyContent="center"
              sx={{
                height: '100%',
                color: 'rgba(148,163,184,0.85)'
              }}
            >
              <Typography variant="body1">
                Ask anything about your ops data. Neuract keeps provenance live.
              </Typography>
            </Stack>
          ) : (
            <VirtualList
              height={viewportHeight || 680}
              width="100%"
              itemCount={totalMessages}
              itemData={{ messages, handlers, streamingMessageId }}
              itemSize={getSize}
              ref={listRef}
              overscanCount={VIRTUAL_OVERSCAN}
              style={{ padding: '24px 0' }}
            >
              {props => <MessageRow {...props} setMeasuredHeight={setMeasuredHeight} />}
            </VirtualList>
          )}
        </StreamScroller>
      </StreamViewport>

      {!isNearBottom && (
        <Button
          variant="contained"
          size="small"
          onClick={() => {
            listRef.current?.scrollToItem(totalMessages - 1, 'smart');
            setAutoScroll(true);
          }}
          sx={{
            position: 'absolute',
            bottom: 112,
            right: 32,
            zIndex: 3,
            borderRadius: 999,
            backgroundColor: '#7c8dff',
            boxShadow: '0 12px 30px rgba(124,141,255,0.45)'
          }}
        >
          New messages
        </Button>
      )}

      <ChatComposer
        onSend={handleSend}
        onScopeChange={onScopeChange}
        scope={scope}
        scopeOptions={scopeOptions}
        onAttach={null}
        promptHintsEnabled={promptHintsEnabled}
        onTogglePromptHints={onTogglePromptHints}
        onOfferAttachment={onOfferAttachment}
        isSending={isSending}
        threadId={threadId}
        onComposerFocus={onComposerFocus}
      />
    </Surface>
  );
}

ChatStream.propTypes = {
  messages: PropTypes.arrayOf(PropTypes.object),
  threadId: PropTypes.string,
  onSendMessage: PropTypes.func,
  onCancelStream: PropTypes.func,
  onRetry: PropTypes.func,
  onSummarize: PropTypes.func,
  onExplain: PropTypes.func,
  onPin: PropTypes.func,
  onExport: PropTypes.func,
  onOpenInLens: PropTypes.func,
  onCreateAlert: PropTypes.func,
  onCopy: PropTypes.func,
  onRegenerate: PropTypes.func,
  onExpandWidget: PropTypes.func,
  onReply: PropTypes.func,
  onTraceExplain: PropTypes.func,
  scope: PropTypes.string,
  scopeOptions: PropTypes.arrayOf(PropTypes.string),
  onScopeChange: PropTypes.func,
  promptHintsEnabled: PropTypes.bool,
  onTogglePromptHints: PropTypes.func,
  onOfferAttachment: PropTypes.func,
  isSending: PropTypes.bool,
  streamingMessageId: PropTypes.string,
  onComposerFocus: PropTypes.func
};

export default function ScenarioComponent({ data }) {
  return <ChatStream messages={data?.messages} />;
}
