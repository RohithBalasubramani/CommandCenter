// @ts-nocheck
'use client';

import * as React from 'react';
import { alpha, useTheme } from '@mui/material/styles';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  IconButton,
  InputAdornment,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
  useMediaQuery
} from '@mui/material';
import SummarizeIcon from '@mui/icons-material/Summarize';
import RefreshIcon from '@mui/icons-material/Refresh';
import SearchIcon from '@mui/icons-material/Search';
import LayersIcon from '@mui/icons-material/Layers';
import DeviceHubIcon from '@mui/icons-material/DeviceHub';
import SensorsIcon from '@mui/icons-material/Sensors';
import InsightsIcon from '@mui/icons-material/Insights';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import LanIcon from '@mui/icons-material/Lan';
import SpeedIcon from '@mui/icons-material/Speed';

const STATUS_COLORS = {
  online: { color: '#00A30E', halo: 'rgba(0,163,14,0.45)' },
  degraded: { color: '#f59e0b', halo: 'rgba(245,158,11,0.45)' },
  offline: { color: '#ef4444', halo: 'rgba(239,68,68,0.55)' },
  unknown: { color: '#94a3b8', halo: 'rgba(148,163,184,0.45)' }
};

const PROTOCOL_COLORS = {
  MQTT: '#38bdf8',
  OPC_UA: '#f97316',
  Modbus: '#22d3ee',
  HTTP: '#8b5cf6',
  RTSP: '#34d399'
};

const LAYER_DEFAULT = ['traffic', 'protocol', 'alarm', 'labels'];

const FILTER_OPTIONS = {
  plant: ['All Plants', 'Hyderabad Campus', 'Solar Fab 2'],
  line: ['All Lines', 'Module Line L1', 'Module Line L2'],
  status: ['All Status', 'online', 'degraded', 'offline', 'unknown'],
  protocol: ['All Protocols', 'MQTT', 'OPC_UA', 'Modbus', 'HTTP']
};

const PANEL_SURFACE = {
  borderRadius: 12,
  border: '1px solid rgba(255,255,255,0.08)',
  background: 'linear-gradient(135deg, rgba(10,28,68,0.92), rgba(6,16,40,0.92))',
  boxShadow: '0 26px 60px rgba(6,12,38,0.55)'
};

const now = Date.now();

function createSeries(base, variance = 5, points = 20) {
  return Array.from({ length: points }, (_, index) => {
    const offset = Math.sin(index / 3) * variance + (Math.random() - 0.5) * variance * 0.6;
    const value = Number((base + offset).toFixed(1));
    const ts = now - (points - index) * 60000;
    return [ts, value];
  });
}

const TOPOLOGY = {
  gateways: [
    {
      id: 'gw-main',
      name: 'Hyderabad Edge Gateway',
      type: 'gateway',
      status: 'online',
      ip: '10.0.0.10',
      plant: 'Hyderabad Campus',
      line: 'Module Line L1',
      rtt_ms: 7,
      loss_pct: 0.2
    }
  ],
  edges: [
    {
      id: 'edge-a',
      name: 'Stringer-Edge-1',
      type: 'edge',
      status: 'degraded',
      ip: '10.0.1.12',
      plant: 'Hyderabad Campus',
      line: 'Module Line L1',
      rtt_ms: 14,
      loss_pct: 1.9
    },
    {
      id: 'edge-b',
      name: 'Lamination-Edge-2',
      type: 'edge',
      status: 'online',
      ip: '10.0.1.13',
      plant: 'Hyderabad Campus',
      line: 'Module Line L2',
      rtt_ms: 9,
      loss_pct: 0.3
    }
  ],
  nodes: [
    {
      id: 'plc-17',
      name: 'PLC-Stringer-17',
      type: 'plc',
      status: 'online',
      ip: '10.0.2.21',
      plant: 'Hyderabad Campus',
      line: 'Module Line L1',
      protocols: ['OPC_UA']
    },
    {
      id: 'cam-3',
      name: 'EL-Cam-3',
      type: 'camera',
      status: 'offline',
      ip: '10.0.3.45',
      plant: 'Hyderabad Campus',
      line: 'Module Line L2',
      protocols: ['RTSP']
    },
    {
      id: 'meter-a13',
      name: 'Meter-Inverter-A13',
      type: 'meter',
      status: 'online',
      ip: '10.0.4.90',
      plant: 'Solar Fab 2',
      line: 'Module Line L2',
      protocols: ['Modbus']
    }
  ],
  links: [
    { from: 'gw-main', to: 'edge-a', protocol: 'MQTT', bw_mbps: 22.4, rtt_ms: 8, errors: 0 },
    { from: 'gw-main', to: 'edge-b', protocol: 'MQTT', bw_mbps: 15.2, rtt_ms: 7, errors: 2 },
    { from: 'edge-a', to: 'plc-17', protocol: 'OPC_UA', bw_mbps: 2.3, rtt_ms: 12, errors: 3 },
    { from: 'edge-b', to: 'meter-a13', protocol: 'Modbus', bw_mbps: 0.4, rtt_ms: 11, errors: 0 }
  ]
};

const METRIC_BLUEPRINTS = {
  'gw-main': {
    vitals: {
      cpu: 42,
      gpu: 8,
      mem: 54,
      disk: 48,
      temp: 44,
      uptime_s: 172800,
      rtt_ms: 7,
      loss_pct: 0.2
    },
    connectors: [
      { type: 'MQTT', status: 'online', msgs_per_s: 420, topics: 118, errors: 0 },
      { type: 'HTTP', status: 'online', msgs_per_s: 12, endpoints: 8, errors: 0 }
    ],
    workloads: [{ name: 'routing-orchestrator', version: '2.4.0', qps: 6.8, p95_ms: 72, last_failure: null }],
    ingestion: { events_per_s: 280, queue_depth: 12, backpressure: 'Normal', drop_rate: 0.3 },
    logs: [
      { ts: 't-60', level: 'INFO', msg: 'MQTT flush 120 msgs' },
      { ts: 't-30', level: 'DEBUG', msg: 'Heartbeat sync edge-b 7ms' }
    ],
    dependencies: {
      upstream: ['Core Data Mesh'],
      downstream: ['Edge-A', 'Edge-B', 'Vault ingest'],
      blast_radius: 'Impacts 5 lines if offline'
    },
    alerts: { critical: 0, high: 1 },
    network: {
      peers: [
        { id: 'edge-a', protocol: 'MQTT', rtt_ms: 8, bw_mbps: 22.4, errors: 0 },
        { id: 'edge-b', protocol: 'MQTT', rtt_ms: 7, bw_mbps: 15.2, errors: 2 }
      ]
    },
    series: {
      cpu: [46, 5],
      gpu: [10, 2],
      mem: [52, 4],
      disk: [48, 3],
      temp: [44, 2],
      msgs: [420, 28],
      throughputIn: [6.4, 1.2],
      throughputOut: [5.1, 1.1],
      errors: [0.2, 0.4]
    }
  },
  'edge-a': {
    vitals: {
      cpu: 63,
      gpu: 22,
      mem: 71,
      disk: 54,
      temp: 56,
      uptime_s: 86400,
      rtt_ms: 14,
      loss_pct: 1.9
    },
    connectors: [
      { type: 'MQTT', status: 'online', msgs_per_s: 180, topics: 42, errors: 0 },
      { type: 'OPC_UA', status: 'degraded', msgs_per_s: 22, endpoints: 3, errors: 5, last_failure: 'Bad_SecurityChecksFailed' }
    ],
    workloads: [
      { name: 'anomaly-detector', version: '1.3.2', qps: 3.2, p95_ms: 85, last_failure: null },
      { name: 'forecast-lite', version: '0.9.5', qps: 1.1, p95_ms: 132, last_failure: 't-1800' }
    ],
    ingestion: { events_per_s: 210, queue_depth: 34, backpressure: 'Moderate', drop_rate: 2.1 },
    logs: [
      { ts: 't-20', level: 'WARN', msg: 'OPC UA handshake retry 2/3' },
      { ts: 't-5', level: 'INFO', msg: 'MQTT batch flush 120 msgs' }
    ],
    dependencies: {
      upstream: ['Main Edge Gateway'],
      downstream: ['PLC-17', 'Line A historian'],
      blast_radius: 'Impacts 3 lines if offline'
    },
    alerts: { critical: 1, high: 2 },
    network: {
      peers: [
        { id: 'gw-main', protocol: 'MQTT', rtt_ms: 8, bw_mbps: 22.4, errors: 0 },
        { id: 'plc-17', protocol: 'OPC_UA', rtt_ms: 12, bw_mbps: 2.3, errors: 3 }
      ]
    },
    series: {
      cpu: [64, 8],
      gpu: [28, 5],
      mem: [72, 5],
      disk: [57, 4],
      temp: [58, 3],
      msgs: [180, 24],
      throughputIn: [3.2, 0.7],
      throughputOut: [2.7, 0.6],
      errors: [1.2, 1.5]
    }
  },
  'edge-b': {
    vitals: {
      cpu: 38,
      gpu: 12,
      mem: 48,
      disk: 43,
      temp: 48,
      uptime_s: 96400,
      rtt_ms: 9,
      loss_pct: 0.3
    },
    connectors: [
      { type: 'MQTT', status: 'online', msgs_per_s: 142, topics: 38, errors: 1 },
      { type: 'Modbus', status: 'online', msgs_per_s: 12, devices: 18, errors: 0 }
    ],
    workloads: [{ name: 'vision-prep', version: '0.7.2', qps: 2.6, p95_ms: 102, last_failure: null }],
    ingestion: { events_per_s: 140, queue_depth: 9, backpressure: 'None', drop_rate: 0.4 },
    logs: [
      { ts: 't-12', level: 'INFO', msg: 'Modbus poll cycle 6 completed' },
      { ts: 't-2', level: 'INFO', msg: 'MQTT flush 88 msgs' }
    ],
    dependencies: {
      upstream: ['Main Edge Gateway'],
      downstream: ['Meter-A13', 'Camera cluster'],
      blast_radius: 'Impacts packaging line if offline'
    },
    alerts: { critical: 0, high: 0 },
    network: {
      peers: [
        { id: 'gw-main', protocol: 'MQTT', rtt_ms: 7, bw_mbps: 15.2, errors: 2 },
        { id: 'meter-a13', protocol: 'Modbus', rtt_ms: 11, bw_mbps: 0.4, errors: 0 }
      ]
    },
    series: {
      cpu: [40, 6],
      gpu: [12, 3],
      mem: [50, 4],
      disk: [44, 3],
      temp: [48, 2],
      msgs: [142, 18],
      throughputIn: [2.1, 0.4],
      throughputOut: [2.4, 0.5],
      errors: [0.4, 0.6]
    }
  },
  'plc-17': {
    vitals: {
      cpu: 28,
      gpu: 0,
      mem: 36,
      disk: 22,
      temp: 40,
      uptime_s: 302400,
      rtt_ms: 12,
      loss_pct: 0.6
    },
    connectors: [
      { type: 'OPC_UA', status: 'degraded', msgs_per_s: 22, endpoints: 3, errors: 5, last_failure: 'Bad_SecurityChecksFailed' }
    ],
    workloads: [],
    ingestion: { events_per_s: 42, queue_depth: 6, backpressure: 'None', drop_rate: 1.2 },
    logs: [
      { ts: 't-18', level: 'WARN', msg: 'Write rejected (Bad_SecurityChecksFailed)' },
      { ts: 't-4', level: 'INFO', msg: 'Subscription keepalive acknowledged' }
    ],
    dependencies: {
      upstream: ['Edge-A'],
      downstream: ['Line A historian'],
      blast_radius: 'Impacts 1 line if offline'
    },
    alerts: { critical: 0, high: 1 },
    network: {
      peers: [{ id: 'edge-a', protocol: 'OPC_UA', rtt_ms: 12, bw_mbps: 2.3, errors: 3 }]
    },
    series: {
      cpu: [30, 4],
      mem: [38, 4],
      disk: [24, 3],
      temp: [41, 3],
      msgs: [40, 8],
      throughputIn: [0.9, 0.2],
      throughputOut: [1.1, 0.2],
      errors: [0.6, 0.8]
    }
  },
  'cam-3': {
    vitals: {
      cpu: 0,
      gpu: 0,
      mem: 0,
      disk: 0,
      temp: 0,
      uptime_s: 0,
      rtt_ms: null,
      loss_pct: null
    },
    connectors: [{ type: 'RTSP', status: 'offline', errors: 4, last_failure: 'No signal 09:42' }],
    workloads: [],
    ingestion: { events_per_s: 0, queue_depth: 0, backpressure: 'NA', drop_rate: 100 },
    logs: [
      { ts: 't-50', level: 'ERROR', msg: 'Stream lost (No signal)' },
      { ts: 't-5', level: 'WARN', msg: 'Reconnect attempt queued' }
    ],
    dependencies: {
      upstream: ['Edge-B'],
      downstream: ['Vision analytics'],
      blast_radius: 'Impacts defect detection stream'
    },
    alerts: { critical: 1, high: 0 },
    network: {
      peers: [{ id: 'edge-b', protocol: 'RTSP', rtt_ms: null, bw_mbps: 0, errors: 4 }]
    },
    series: {
      cpu: [0, 0],
      mem: [0, 0],
      disk: [0, 0],
      temp: [0, 0],
      msgs: [0, 0],
      throughputIn: [0, 0],
      throughputOut: [0, 0],
      errors: [4, 0]
    }
  },
  'meter-a13': {
    vitals: {
      cpu: 12,
      gpu: 0,
      mem: 18,
      disk: 14,
      temp: 37,
      uptime_s: 512000,
      rtt_ms: 11,
      loss_pct: 0.4
    },
    connectors: [{ type: 'Modbus', status: 'online', msgs_per_s: 18, devices: 6, errors: 0 }],
    workloads: [],
    ingestion: { events_per_s: 18, queue_depth: 2, backpressure: 'None', drop_rate: 0.1 },
    logs: [
      { ts: 't-15', level: 'INFO', msg: 'Meter read stable 418 kWh' },
      { ts: 't-2', level: 'DEBUG', msg: 'Polling cycle completed' }
    ],
    dependencies: {
      upstream: ['Edge-B'],
      downstream: ['Energy dashboard'],
      blast_radius: 'Impacts energy reports'
    },
    alerts: { critical: 0, high: 0 },
    network: {
      peers: [{ id: 'edge-b', protocol: 'Modbus', rtt_ms: 11, bw_mbps: 0.4, errors: 0 }]
    },
    series: {
      cpu: [14, 2],
      mem: [20, 3],
      disk: [16, 2],
      temp: [38, 2],
      msgs: [18, 3],
      throughputIn: [0.5, 0.1],
      throughputOut: [0.4, 0.1],
      errors: [0.1, 0.2]
    }
  }
};
const METRICS = Object.fromEntries(
  Object.entries(METRIC_BLUEPRINTS).map(([id, blueprint]) => {
    const { series = {}, ...rest } = blueprint;
    const timeseries = {
      cpu: createSeries(series.cpu?.[0] ?? 0, series.cpu?.[1] ?? 0),
      gpu: series.gpu ? createSeries(series.gpu[0], series.gpu[1]) : [],
      mem: createSeries(series.mem?.[0] ?? 0, series.mem?.[1] ?? 0),
      disk: createSeries(series.disk?.[0] ?? 0, series.disk?.[1] ?? 0),
      temp: createSeries(series.temp?.[0] ?? 0, series.temp?.[1] ?? 0),
      msgs_per_s: createSeries(series.msgs?.[0] ?? 0, series.msgs?.[1] ?? 0),
      throughput_in_mbps: createSeries(series.throughputIn?.[0] ?? 0, series.throughputIn?.[1] ?? 0),
      throughput_out_mbps: createSeries(series.throughputOut?.[0] ?? 0, series.throughputOut?.[1] ?? 0),
      errors_per_min: createSeries(series.errors?.[0] ?? 0, series.errors?.[1] ?? 0)
    };
    return [id, { id, ...rest, timeseries }];
  })
);

const SUMMARY_SAMPLE = {
  selection: ['edge-a', 'plc-17'],
  summary:
    'Edge-A degraded: RTT +42%, OPC UA handshake errors persisting. PLC-17 online with stable throughput. Recommend connector restart and QoS bump.',
  confidence: 0.81,
  sources: [
    { id: 'metrics/edge-a', age_s: 12 },
    { id: 'logs/edge-a', age_s: 45 }
  ]
};

function computeSummaryTiles(topology, metrics) {
  const population = [...topology.gateways, ...topology.edges, ...topology.nodes];
  const statusCounts = population.reduce(
    (accumulator, device) => {
      accumulator[device.status] = (accumulator[device.status] || 0) + 1;
      return accumulator;
    },
    {}
  );
  const gatewayStatuses = topology.gateways.reduce(
    (accumulator, device) => {
      accumulator[device.status] = (accumulator[device.status] || 0) + 1;
      return accumulator;
    },
    {}
  );
  const connectors = Object.values(metrics).flatMap(entry => entry.connectors || []);
  const connectorCounts = connectors.reduce(
    (accumulator, connector) => {
      const key = connector.type || 'other';
      accumulator[key] = (accumulator[key] || 0) + (connector.status === 'online' ? 1 : 0);
      return accumulator;
    },
    {}
  );
  const connectorsUp = Object.values(connectorCounts).reduce((sum, value) => sum + value, 0);
  const avgRtt =
    population.reduce((sum, item) => sum + (item.rtt_ms || 0), 0) / Math.max(population.length, 1);
  const avgLoss =
    population.reduce((sum, item) => sum + (item.loss_pct || 0), 0) / Math.max(population.length, 1);
  const linksWithErrors = topology.links.filter(link => link.errors > 0).length;

  const inferenceLoads = Object.values(metrics)
    .flatMap(entry => entry.workloads || [])
    .filter(workload => workload.name.includes('anomaly') || workload.name.includes('vision'));
  const inferenceQps = inferenceLoads.reduce((sum, workload) => sum + workload.qps, 0);
  const inferenceP95 =
    inferenceLoads.reduce((sum, workload) => sum + workload.p95_ms, 0) / Math.max(inferenceLoads.length || 1, 1);

  const alerts = Object.values(metrics).reduce(
    (accumulator, entry) => {
      accumulator.critical += entry.alerts?.critical || 0;
      accumulator.high += entry.alerts?.high || 0;
      return accumulator;
    },
    { critical: 0, high: 0 }
  );

  return [
    {
      key: 'gateways',
      label: 'Gateways',
      icon: DeviceHubIcon,
      accent: '#5b8cff',
      headline: (gatewayStatuses.online || 0).toString(),
      caption: 'online',
      secondary: `Total ${topology.gateways.length}`,
      meta: [
        { label: 'Degraded', value: gatewayStatuses.degraded || 0 },
        { label: 'Offline', value: gatewayStatuses.offline || 0 }
      ]
    },
    {
      key: 'nodes',
      label: 'Edge estate',
      icon: SensorsIcon,
      accent: '#3cc7d3',
      headline: (statusCounts.online || 0).toString(),
      caption: 'devices online',
      secondary: `${population.length} total nodes`,
      meta: [
        { label: 'Degraded', value: statusCounts.degraded || 0 },
        { label: 'Offline', value: statusCounts.offline || 0 }
      ]
    },
    {
      key: 'connectors',
      label: 'Connectors',
      icon: LanIcon,
      accent: '#7c8dff',
      headline: connectorsUp.toString(),
      caption: 'links up',
      secondary: 'Live protocol uplinks',
      meta: Object.entries({
        MQTT: connectorCounts.MQTT || 0,
        OPC: connectorCounts.OPC_UA || 0,
        Modbus: connectorCounts.Modbus || 0
      })
        .filter(([, value]) => value != null)
        .map(([label, value]) => ({ label, value }))
    },
    {
      key: 'network',
      label: 'Network health',
      icon: SpeedIcon,
      accent: '#f89d5c',
      headline: avgRtt.toFixed(1),
      caption: 'ms avg RTT',
      secondary: `Loss ${avgLoss.toFixed(1)}%`,
      meta: [{ label: 'Links w/ errors', value: linksWithErrors }]
    },
    {
      key: 'inference',
      label: 'Inference load',
      icon: InsightsIcon,
      accent: '#9c7cff',
      headline: inferenceQps.toFixed(1),
      caption: 'QPS',
      secondary: `p95 ${inferenceP95.toFixed(0)} ms`,
      meta: [{ label: 'Workloads', value: inferenceLoads.length }]
    },
    {
      key: 'alerts',
      label: 'Active alerts',
      icon: WarningAmberIcon,
      accent: '#fb7185',
      headline: alerts.critical.toString(),
      caption: 'critical',
      secondary: `${alerts.high} high priority`,
      meta: [
        { label: 'High', value: alerts.high },
        { label: 'Total', value: alerts.critical + alerts.high }
      ]
    }
  ];
}

function computeRadialLayout(topology, overrides = {}) {
  const center = 500;
  const layout = { nodes: [], links: [], map: new Map() };

  function register(node, position, meta) {
    const override = overrides[node.id];
    const finalPosition = override ? { x: override.x, y: override.y } : position;
    const entry = {
      id: node.id,
      name: node.name,
      type: node.type,
      status: node.status,
      plant: node.plant,
      line: node.line,
      protocols: node.protocols || [],
      data: node,
      position: finalPosition,
      base: position,
      meta: meta || {}
    };
    layout.nodes.push(entry);
    layout.map.set(node.id, entry);
  }

  const gateway = topology.gateways[0];
  if (gateway) {
    register(gateway, { x: center, y: center }, { radius: 0, size: 110 });
  }

  const edgeRadius = 260;
  topology.edges.forEach((edge, index) => {
    const angle = (index / Math.max(topology.edges.length, 1)) * Math.PI * 2;
    register(
      edge,
      { x: center + Math.cos(angle) * edgeRadius, y: center + Math.sin(angle) * edgeRadius },
      { radius: 1, size: 86 }
    );
  });

  const nodeRadius = 380;
  const linksBySource = topology.links.reduce((accumulator, link) => {
    accumulator[link.from] = accumulator[link.from] || [];
    accumulator[link.from].push(link.to);
    return accumulator;
  }, {});

  topology.nodes.forEach(node => {
    const parentId = Object.entries(linksBySource).find(([, children]) => children.includes(node.id))?.[0] || 'gw-main';
    const siblings = topology.nodes.filter(candidate => {
      const candidateParent = Object.entries(linksBySource).find(([, children]) =>
        children.includes(candidate.id)
      )?.[0];
      return (candidateParent || 'gw-main') === parentId;
    });
    const index = siblings.findIndex(item => item.id === node.id);
    const angleOffset = layout.map.get(parentId)
      ? Math.atan2(layout.map.get(parentId).position.y - center, layout.map.get(parentId).position.x - center)
      : 0;
    const angle = (index / Math.max(siblings.length, 1)) * Math.PI * 2 + angleOffset;
    register(
      node,
      { x: center + Math.cos(angle) * nodeRadius, y: center + Math.sin(angle) * nodeRadius },
      { radius: 2, size: 72, parentId }
    );
  });

  layout.links = topology.links
    .map(link => {
      const source = layout.map.get(link.from);
      const target = layout.map.get(link.to);
      if (!source || !target) {
        return null;
      }
      const midpoint = {
        x: (source.position.x + target.position.x) / 2,
        y: (source.position.y + target.position.y) / 2
      };
      const curvature = 0.12;
      const control = {
        x: midpoint.x + (target.position.y - source.position.y) * curvature,
        y: midpoint.y - (target.position.x - source.position.x) * curvature
      };
      return {
        id: `${link.from}-${link.to}`,
        ...link,
        path: `M ${source.position.x} ${source.position.y} Q ${control.x} ${control.y} ${target.position.x} ${target.position.y}`
      };
    })
    .filter(Boolean);

  return layout;
}

function formatUptime(seconds) {
  if (!seconds) {
    return '--';
  }
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  return days ? `${days}d ${hours}h` : `${hours}h`;
}

function Sparkline({ data, color }) {
  if (!data || !data.length) {
    return null;
  }
  const values = data.map(([, value]) => value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min || 1;
  const path = data
    .map(([, value], index) => {
      const x = (index / Math.max(data.length - 1, 1)) * 100;
      const y = 100 - ((value - min) / spread) * 100;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
      <path d={path} stroke={color} strokeWidth={2} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function DetailsPane({
  selected,
  metricsMap,
  tab,
  onTabChange,
  onSummarize,
  summary,
  onClearSummary,
  layout
}) {
  if (!selected.length) {
    return (
      <Paper elevation={0} sx={{ ...PANEL_SURFACE, minHeight: 0, p: 3 }}>
        <Stack spacing={1.5} alignItems="center" justifyContent="center" sx={{ height: 1 }}>
          <Typography variant="subtitle1" fontWeight={600}>
            Select a device to inspect
          </Typography>
          <Typography variant="body2" color="text.secondary" textAlign="center">
            Click nodes on the graph or shift-select to compare up to four devices.
          </Typography>
        </Stack>
      </Paper>
    );
  }

  const primary = selected[0];
  const metrics = metricsMap[primary.id] || {};
  const tabs = [
    'overview',
    'health',
    'network',
    'connectors',
    'ingestion',
    'workloads',
    'logs',
    'config',
    'dependencies'
  ];
  if (selected.length > 1) {
    tabs.push('compare');
  }

  const renderVitals = (label, value) => (
    <Stack key={label} spacing={0.5}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center">
        <LinearProgress
          variant="determinate"
          value={Math.min(Math.max(value ?? 0, 0), 100)}
          sx={{ flexGrow: 1, height: 6, borderRadius: 999 }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ width: 40, textAlign: 'right' }}>
          {value ?? '--'}%
        </Typography>
      </Stack>
    </Stack>
  );

  const sectionSurface = {
    borderRadius: 3,
    border: '1px solid rgba(120,140,255,0.16)',
    background: 'linear-gradient(135deg, rgba(12,24,52,0.82), rgba(7,16,34,0.92))',
    boxShadow: '0 18px 34px rgba(6,12,32,0.42)'
  };

  const pillChipSx = {
    borderRadius: 999,
    border: 'none',
    px: 1.6,
    py: 0.2,
    fontSize: 12,
    color: 'rgba(226,232,240,0.88)',
    backgroundColor: 'rgba(125,142,255,0.22)',
    '& .MuiChip-label': {
      px: 0
    }
  };

  const renderSeries = (label, key, color) => (
    <Paper key={key} elevation={0} sx={{ ...sectionSurface, p: 2 }}>
      <Stack spacing={1.2}>
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Typography variant="subtitle2" fontWeight={600}>
            {label}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {metrics.timeseries?.[key]?.slice(-1)[0]?.[1] ?? '--'}
          </Typography>
        </Stack>
        <Box sx={{ height: 80 }}>
          <Sparkline data={metrics.timeseries?.[key]} color={color} />
        </Box>
      </Stack>
    </Paper>
  );

  return (
    <Paper
      elevation={0}
      sx={{
        ...PANEL_SURFACE,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0
      }}
    >
      <Box
        sx={{
          px: { xs: 2, md: 2.25 },
          pt: { xs: 2, md: 2.4 },
          pb: { xs: 1.6, md: 1.8 },
          borderBottom: '1px solid rgba(148,163,184,0.16)'
        }}
      >
        <Stack spacing={1.2}>
          <Stack direction="row" spacing={1.2} alignItems="center" flexWrap="wrap">
            <Typography variant="h6" fontWeight={600}>
              {primary.name}
            </Typography>
            <Chip
              size="small"
              label={primary.type.toUpperCase()}
              sx={{ borderRadius: 1.5, bgcolor: 'rgba(148,163,184,0.16)', color: 'rgba(226,232,240,0.9)' }}
            />
            <Chip
              size="small"
              label={primary.status}
              sx={{
                borderRadius: 1.5,
                bgcolor: alpha(STATUS_COLORS[primary.status]?.color || '#94a3b8', 0.18),
                color: STATUS_COLORS[primary.status]?.color || '#94a3b8',
                textTransform: 'capitalize'
              }}
            />
            {primary.plant && (
              <Chip
                size="small"
                label={`${primary.plant}  -  ${primary.line}`}
                sx={{ borderRadius: 1.5, bgcolor: 'rgba(148,163,184,0.12)', color: 'rgba(226,232,240,0.75)' }}
              />
            )}
          </Stack>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.2}>
            <Button
              size="small"
              variant="contained"
              color="secondary"
              startIcon={<SummarizeIcon fontSize="inherit" />}
              sx={{ textTransform: 'none', borderRadius: 999, boxShadow: 'none' }}
              onClick={onSummarize}
            >
              Summarize selection
            </Button>
            <Stack direction="row" spacing={1}>
              <Button size="small" variant="outlined" color="secondary" disabled sx={{ borderRadius: 999 }}>
                Restart connector (Preview)
              </Button>
              <Button size="small" variant="outlined" color="secondary" disabled sx={{ borderRadius: 999 }}>
                Restart service (Preview)
              </Button>
            </Stack>
          </Stack>
        </Stack>
        {summary && (
          <Alert
            severity="info"
            sx={{
              mt: 2,
              borderRadius: 2,
              border: '1px solid rgba(148,163,184,0.24)',
              bgcolor: 'rgba(15,118,110,0.08)'
            }}
            action={
              <Button color="inherit" size="small" onClick={onClearSummary}>
                Dismiss
              </Button>
            }
          >
            <Stack spacing={0.75}>
              <Typography variant="body2">{summary.summary}</Typography>
              <Typography variant="caption" color="text.secondary">
                Confidence {(summary.confidence * 100).toFixed(0)}%  -  Sources {summary.sources.map(source => source.id).join(', ')}
              </Typography>
            </Stack>
          </Alert>
        )}
      </Box>
      <Tabs
        value={tab}
        onChange={onTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ px: { xs: 2, md: 2.25 }, '& .MuiTab-root': { textTransform: 'none' } }}
      >
        {tabs.map(value => (
          <Tab key={value} value={value} label={value === 'overview' ? 'Overview' : value.charAt(0).toUpperCase() + value.slice(1)} />
        ))}
      </Tabs>
      <Divider sx={{ borderColor: 'rgba(148,163,184,0.12)' }} />
      <Box sx={{ flexGrow: 1, overflowY: 'auto', px: { xs: 2, md: 2.25 }, py: { xs: 2, md: 2.4 } }}>
        {tab === 'overview' && (
          <Stack spacing={2.4}>
            <Paper elevation={0} sx={{ ...sectionSurface, p: { xs: 1.8, md: 2.2 } }}>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2.4}>
                <Stack spacing={1.1} sx={{ flex: 1 }}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Quick vitals
                  </Typography>
                  <Stack spacing={1.1}>
                    {['cpu', 'gpu', 'mem', 'disk'].map(key => renderVitals(key.toUpperCase(), metrics.vitals?.[key]))}
                  </Stack>
                </Stack>
                <Stack spacing={0.7} sx={{ minWidth: 200 }}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Status
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Temp {metrics.vitals?.temp ?? '--'} deg C
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    RTT {metrics.vitals?.rtt_ms ?? '--'} ms  -  Loss {metrics.vitals?.loss_pct ?? '--'}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Uptime {formatUptime(metrics.vitals?.uptime_s)}
                  </Typography>
                </Stack>
              </Stack>
            </Paper>
            <Paper elevation={0} sx={{ ...sectionSurface, p: { xs: 1.8, md: 2.1 } }}>
              <Stack spacing={1.6}>
                <Stack spacing={1}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Connectors
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {(metrics.connectors || []).map(connector => {
                      const statusKey = connector.status === 'online' ? 'online' : connector.status === 'offline' ? 'offline' : 'degraded';
                      const statusColor = STATUS_COLORS[statusKey]?.color || '#94a3b8';
                      return (
                        <Chip
                          key={connector.type}
                          label={`${connector.type}  -  ${connector.status}`}
                          sx={{
                            ...pillChipSx,
                            bgcolor: alpha(statusColor, 0.22),
                            color: statusColor
                          }}
                        />
                      );
                    })}
                  </Stack>
                  {!metrics.connectors?.length && (
                    <Typography variant="body2" color="text.secondary">
                      No connectors configured.
                    </Typography>
                  )}
                </Stack>
                <Divider sx={{ borderColor: 'rgba(120,140,255,0.14)' }} />
                <Stack spacing={1}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    Workloads snapshot
                  </Typography>
                  {(metrics.workloads || []).map(workload => (
                    <Box
                      key={workload.name}
                      sx={{
                        borderRadius: 2,
                        border: '1px solid rgba(120,140,255,0.16)',
                        backgroundColor: 'rgba(12,24,52,0.6)',
                        px: 1.6,
                        py: 1.2,
                        display: 'flex',
                        alignItems: 'flex-start',
                        justifyContent: 'space-between',
                        gap: 2
                      }}
                    >
                      <Stack spacing={0.4}>
                        <Typography variant="body2" fontWeight={600}>
                          {workload.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          v{workload.version}
                        </Typography>
                      </Stack>
                      <Stack spacing={0.3} alignItems="flex-end">
                        <Typography variant="body2" color="text.secondary">
                          QPS {workload.qps}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          p95 {workload.p95_ms} ms
                        </Typography>
                        {workload.last_failure && (
                          <Typography variant="caption" color="warning.main">
                            Last failure {workload.last_failure}
                          </Typography>
                        )}
                      </Stack>
                    </Box>
                  ))}
                  {!metrics.workloads?.length && (
                    <Typography variant="body2" color="text.secondary">
                      No workloads deployed on this device.
                    </Typography>
                  )}
                </Stack>
              </Stack>
            </Paper>
          </Stack>
        )}
        {tab === 'health' && (
          <Stack spacing={2.4}>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.4}>
              {renderSeries('CPU %', 'cpu', '#38bdf8')}
              {renderSeries('GPU %', 'gpu', '#8b5cf6')}
            </Stack>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.4}>
              {renderSeries('Memory %', 'mem', '#0ea5e9')}
              {renderSeries('Disk %', 'disk', '#14b8a6')}
            </Stack>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.4}>
              {renderSeries('Temperature deg C', 'temp', '#f97316')}
              {renderSeries('Errors / min', 'errors_per_min', '#ef4444')}
            </Stack>
          </Stack>
        )}
        {tab === 'network' && (
          <Stack spacing={2.2}>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.4}>
              {renderSeries('Throughput in Mbps', 'throughput_in_mbps', '#38bdf8')}
              {renderSeries('Throughput out Mbps', 'throughput_out_mbps', '#34d399')}
            </Stack>
            <Paper elevation={0} sx={{ ...sectionSurface, p: 2 }}>
              <Stack spacing={1.4}>
                <Typography variant="subtitle2" fontWeight={600}>
                  Peers
                </Typography>
                {(layout.links || [])
                  .filter(link => link.from === primary.id || link.to === primary.id)
                  .map(link => {
                    const peerId = link.from === primary.id ? link.to : link.from;
                    const peer = layout.map.get(peerId);
                    return (
                      <Box
                        key={link.id}
                        sx={{
                          borderRadius: 2,
                          border: '1px solid rgba(120,140,255,0.12)',
                          backgroundColor: 'rgba(12,24,52,0.62)',
                          px: 1.6,
                          py: 1.1,
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          gap: 2
                        }}
                      >
                        <Stack spacing={0.35}>
                          <Typography variant="body2" fontWeight={500}>
                            {peer?.name || peerId}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {link.protocol}  -  RTT {link.rtt_ms} ms  -  BW {link.bw_mbps} Mbps
                          </Typography>
                        </Stack>
                        <Chip
                          size="small"
                          label={`Errors ${link.errors}`}
                          sx={{
                            ...pillChipSx,
                            bgcolor: alpha(link.errors ? '#ef4444' : '#34d399', 0.22),
                            color: link.errors ? '#ef4444' : '#34d399'
                          }}
                        />
                      </Box>
                    );
                  })}
              </Stack>
            </Paper>
          </Stack>
        )}
        {tab === 'connectors' && (
          <Stack spacing={1.8}>
            {(metrics.connectors || []).map(connector => {
              const statusKey = connector.status === 'online' ? 'online' : connector.status === 'offline' ? 'offline' : 'degraded';
              const statusColor = STATUS_COLORS[statusKey]?.color || '#94a3b8';
              return (
                <Paper key={connector.type} elevation={0} sx={{ ...sectionSurface, p: 2 }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack spacing={0.5}>
                      <Typography variant="subtitle2" fontWeight={600}>
                        {connector.type}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Status {connector.status}
                      </Typography>
                    </Stack>
                    <Chip
                      size="small"
                      label={`${connector.msgs_per_s || connector.devices || connector.endpoints || 0} msgs/s`}
                      sx={{ ...pillChipSx, bgcolor: alpha(statusColor, 0.18), color: statusColor }}
                    />
                  </Stack>
                  <Stack direction="row" spacing={1.5} sx={{ mt: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      Errors {connector.errors ?? '--'}
                    </Typography>
                    {connector.last_failure && (
                      <Typography variant="caption" color="warning.main">
                        Last failure: {connector.last_failure}
                      </Typography>
                    )}
                  </Stack>
                </Paper>
              );
            })}
            {!metrics.connectors?.length && (
              <Typography variant="body2" color="text.secondary">
                No connectors registered.
              </Typography>
            )}
          </Stack>
        )}
        {tab === 'ingestion' && (
          <Stack spacing={2.2}>
            <Paper elevation={0} sx={{ ...sectionSurface, p: 2 }}>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                {[
                  ['Events/s', metrics.ingestion?.events_per_s],
                  ['Queue depth', metrics.ingestion?.queue_depth],
                  ['Backpressure', metrics.ingestion?.backpressure],
                  ['Drop rate', metrics.ingestion?.drop_rate != null ? `${metrics.ingestion.drop_rate}%` : '--']
                ].map(([label, value]) => (
                  <Stack key={label} spacing={0.35} sx={{ flex: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      {label}
                    </Typography>
                    <Typography variant="h6">{value ?? '--'}</Typography>
                  </Stack>
                ))}
              </Stack>
            </Paper>
            <Paper elevation={0} sx={{ ...sectionSurface, p: 2 }}>
              <Stack spacing={1.2}>
                <Typography variant="subtitle2" fontWeight={600}>
                  Recent parse issues
                </Typography>
                {(metrics.ingestion?.parse_errors || []).map(error => (
                  <Box
                    key={error.ts}
                    sx={{
                      borderRadius: 2,
                      border: '1px solid rgba(239,68,68,0.25)',
                      backgroundColor: 'rgba(239,68,68,0.12)',
                      px: 1.6,
                      py: 1.2
                    }}
                  >
                    <Typography variant="body2" fontWeight={600}>
                      {error.ts}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {error.snippet}
                    </Typography>
                  </Box>
                ))}
                {!metrics.ingestion?.parse_errors?.length && (
                  <Typography variant="body2" color="text.secondary">
                    No parse errors recorded.
                  </Typography>
                )}
              </Stack>
            </Paper>
          </Stack>
        )}
        {tab === 'workloads' && (
          <Stack spacing={1.6}>
            {(metrics.workloads || []).map(workload => (
              <Paper key={workload.name} elevation={0} sx={{ ...sectionSurface, p: 2 }}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                  <Stack spacing={0.4}>
                    <Typography variant="subtitle2" fontWeight={600}>
                      {workload.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Version {workload.version}
                    </Typography>
                  </Stack>
                  <Stack spacing={0.3} alignItems="flex-end">
                    <Typography variant="body2">QPS {workload.qps}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      p95 {workload.p95_ms} ms
                    </Typography>
                    {workload.last_failure && (
                      <Typography variant="caption" color="warning.main">
                        Last failure {workload.last_failure}
                      </Typography>
                    )}
                  </Stack>
                </Stack>
              </Paper>
            ))}
            {!metrics.workloads?.length && (
              <Typography variant="body2" color="text.secondary">
                No workloads deployed on this device.
              </Typography>
            )}
          </Stack>
        )}
        {tab === 'logs' && (
          <Stack spacing={1.2}>
            {(metrics.logs || []).map(log => (
              <Paper key={`${log.ts}-${log.msg}`} elevation={0} sx={{ ...sectionSurface, p: 1.6 }}>
                <Stack direction="row" spacing={1.2} alignItems="center">
                  <Chip
                    size="small"
                    label={log.level}
                    sx={{
                      ...pillChipSx,
                      bgcolor:
                        log.level === 'ERROR'
                          ? 'rgba(239,68,68,0.18)'
                          : log.level === 'WARN'
                          ? 'rgba(245,158,11,0.18)'
                          : 'rgba(148,163,184,0.14)',
                      color:
                        log.level === 'ERROR'
                          ? '#ef4444'
                          : log.level === 'WARN'
                          ? '#f59e0b'
                          : 'rgba(226,232,240,0.88)'
                    }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {log.ts}
                  </Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 0.6 }}>
                  {log.msg}
                </Typography>
              </Paper>
            ))}
          </Stack>
        )}
        {tab === 'config' && (
          <Paper elevation={0} sx={{ ...sectionSurface, p: 2 }}>
            <Stack spacing={1.1}>
              {[
                ['Device ID', primary.id],
                ['IP address', primary.data.ip],
                ['Plant', primary.plant],
                ['Line', primary.line],
                ['Protocols', primary.protocols.join(', ') || '--'],
                ['Uptime', formatUptime(metrics.vitals?.uptime_s)]
              ].map(([label, value]) => (
                <Stack key={label} direction="row" justifyContent="space-between">
                  <Typography variant="body2" color="text.secondary">
                    {label}
                  </Typography>
                  <Typography variant="body2">{value ?? '--'}</Typography>
                </Stack>
              ))}
            </Stack>
          </Paper>
        )}
        {tab === 'dependencies' && (
          <Paper elevation={0} sx={{ ...sectionSurface, p: 2 }}>
            <Stack spacing={1.2}>
              <Typography variant="subtitle2" fontWeight={600}>Upstream</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {(metrics.dependencies?.upstream || []).map(item => (
                  <Chip key={item} label={item} sx={{ ...pillChipSx, bgcolor: 'rgba(124,141,255,0.18)' }} />
                ))}
              </Stack>
              <Typography variant="subtitle2" fontWeight={600}>Downstream</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {(metrics.dependencies?.downstream || []).map(item => (
                  <Chip key={item} label={item} sx={{ ...pillChipSx, bgcolor: 'rgba(59,201,173,0.2)', color: '#5eead4' }} />
                ))}
              </Stack>
              <Typography variant="body2" color="text.secondary">
                {metrics.dependencies?.blast_radius || 'Blast radius data unavailable.'}
              </Typography>
            </Stack>
          </Paper>
        )}
        {tab === 'compare' && (
          <Stack spacing={1.6}>
            {['cpu', 'mem', 'rtt_ms', 'loss_pct', 'msgs_per_s'].map(key => (
              <Paper key={key} elevation={0} sx={{ ...sectionSurface, p: 1.8 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {key}
                </Typography>
                <Stack spacing={0.4}>
                  {selected.map(entry => (
                    <Stack key={entry.id} direction="row" justifyContent="space-between">
                      <Typography variant="body2" fontWeight={500}>
                        {entry.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {key === 'msgs_per_s'
                          ? metricsMap[entry.id]?.timeseries?.msgs_per_s?.slice(-1)[0]?.[1] ?? '--'
                          : metricsMap[entry.id]?.vitals?.[key] ?? '--'}
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
              </Paper>
            ))}
          </Stack>
        )}
      </Box>
      <Divider sx={{ borderColor: 'rgba(148,163,184,0.12)' }} />
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={1.2}
        sx={{ px: { xs: 2, md: 2.25 }, py: 2 }}
      >
        {['Open in MES Line 2', 'Open EMS Meter A-13', 'Open Asset Page', 'Open Edge Sentinel'].map(link => (
          <Chip
            key={link}
            label={link}
            clickable
            sx={{
              ...pillChipSx,
              borderRadius: 999,
              bgcolor: 'rgba(124,141,255,0.16)',
              color: '#cbd5ff',
              textTransform: 'none',
              fontWeight: 500
            }}
          />
        ))}
      </Stack>
    </Paper>
  );
}
function GraphCanvas({ layout, selectedIds, onSelect, filters, layers, searchTerm, onPositionChange }) {
  const svgRef = React.useRef(null);
  const [dragging, setDragging] = React.useState(null);

  const handlePointerMove = React.useCallback(
    event => {
      if (!dragging || !svgRef.current) {
        return;
      }
      const rect = svgRef.current.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * 1000;
      const y = ((event.clientY - rect.top) / rect.height) * 1000;
      onPositionChange(dragging.id, {
        x: Math.max(80, Math.min(920, x)),
        y: Math.max(80, Math.min(920, y))
      });
    },
    [dragging, onPositionChange]
  );

  const handlePointerUp = React.useCallback(() => setDragging(null), []);

  const matchesFilters = React.useCallback(
    node => {
      if (filters.status !== 'All Status' && filters.status !== node.status) {
        return false;
      }
      if (filters.protocol !== 'All Protocols' && !(node.protocols || []).includes(filters.protocol)) {
        return false;
      }
      if (filters.plant !== 'All Plants' && node.plant !== filters.plant) {
        return false;
      }
      if (filters.line !== 'All Lines' && node.line !== filters.line) {
        return false;
      }
      return true;
    },
    [filters]
  );

  const matchesSearch = React.useCallback(
    node => {
      if (!searchTerm) {
        return true;
      }
      const term = searchTerm.toLowerCase();
      return (
        node.name.toLowerCase().includes(term) ||
        node.id.toLowerCase().includes(term) ||
        (node.data.ip || '').toLowerCase().includes(term)
      );
    },
    [searchTerm]
  );

  return (
    <Paper
      elevation={0}
      sx={{ ...PANEL_SURFACE, position: 'relative', minHeight: 0, overflow: 'hidden', minWidth: 0 }}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      <svg ref={svgRef} viewBox="0 0 1000 1000" style={{ width: '100%', height: '100%', position: 'absolute', inset: 0 }}>
        {layout.links.map(link => {
          const visible =
            filters.protocol === 'All Protocols' || filters.protocol === link.protocol || !link.protocol;
          const color = layers.includes('protocol')
            ? PROTOCOL_COLORS[link.protocol] || 'rgba(148,163,184,0.35)'
            : 'rgba(148,163,184,0.25)';
          const strokeWidth = layers.includes('traffic')
            ? Math.max(1.5, Math.min(5.5, link.bw_mbps / 4))
            : 1.4;
          const dash = link.errors && layers.includes('alarm') ? '6 4' : '1';
          return (
            <path
              key={link.id}
              d={link.path}
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={dash}
              strokeOpacity={visible ? 0.85 : 0.12}
              fill="none"
            />
          );
        })}
      </svg>
      {layout.nodes.map(node => {
        const palette = STATUS_COLORS[node.status] || STATUS_COLORS.unknown;
        const selected = selectedIds.includes(node.id);
        const dim = !matchesFilters(node) || !matchesSearch(node);
        const size = node.meta.size || (node.type === 'gateway' ? 110 : node.type === 'edge' ? 86 : 72);

        const handleClick = event => {
          onSelect(node.id, event.shiftKey);
        };

        const handlePointerDown = event => {
          if (event.button !== 0) {
            return;
          }
          setDragging({ id: node.id });
        };

        return (
          <Tooltip
            key={node.id}
            title={
              <Stack spacing={0.4}>
                <Typography variant="subtitle2">{node.name}</Typography>
                <Typography variant="caption">{node.data.ip}</Typography>
                <Typography variant="caption">
                  RTT {node.data.rtt_ms ?? '--'} ms  -  Loss {node.data.loss_pct ?? '--'}%
                </Typography>
              </Stack>
            }
          >
            <Box
              onClick={handleClick}
              onPointerDown={handlePointerDown}
              sx={{
                position: 'absolute',
                left: `${node.position.x / 10}%`,
                top: `${node.position.y / 10}%`,
                width: size,
                height: size,
                transform: 'translate(-50%, -50%)',
                borderRadius: size,
                border: '1px solid rgba(255,255,255,0.08)',
                background: 'linear-gradient(135deg, rgba(17,24,39,0.95), rgba(24,39,70,0.98))',
                boxShadow: selected
                  ? '0 0 0 4px rgba(16,98,185,0.45), 0 20px 40px rgba(6,12,38,0.45)'
                  : '0 16px 32px rgba(6,12,38,0.32)',
                cursor: 'pointer',
                opacity: dim ? 0.25 : 1,
                filter: dim ? 'grayscale(0.4)' : 'none',
                transition: 'transform 160ms ease, box-shadow 160ms ease'
              }}
            >
              {layers.includes('alarm') && (METRICS[node.id]?.alerts?.critical || METRICS[node.id]?.alerts?.high) && (
                <Box
                  sx={{
                    position: 'absolute',
                    inset: -18,
                    borderRadius: size + 36,
                    background: `radial-gradient(circle, ${palette.halo}, transparent)`,
                    animation: 'haloPulse 1.2s ease-in-out infinite'
                  }}
                />
              )}
              <Stack spacing={0.6} alignItems="center" justifyContent="center" sx={{ position: 'relative', zIndex: 1, height: 1 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: 12, bgcolor: palette.color, boxShadow: `0 0 12px ${palette.halo}` }} />
                {layers.includes('labels') && (
                  <Typography variant="caption" color="rgba(226,232,240,0.9)" textAlign="center">
                    {node.name}
                  </Typography>
                )}
                <Box sx={{ width: '70%', height: 24 }}>
                  <Sparkline data={METRICS[node.id]?.timeseries?.msgs_per_s} color={palette.color} />
                </Box>
              </Stack>
            </Box>
          </Tooltip>
        );
      })}
      <style jsx>{`
        @keyframes haloPulse {
          0% {
            opacity: 0.2;
            transform: scale(0.94);
          }
          50% {
            opacity: 1;
            transform: scale(1);
          }
          100% {
            opacity: 0.2;
            transform: scale(0.94);
          }
        }
      `}</style>
    </Paper>
  );
}
function ListMode({ nodes, selectedIds, onSelect }) {
  return (
    <Stack spacing={1.5}>
      {nodes.map(node => {
        const palette = STATUS_COLORS[node.status] || STATUS_COLORS.unknown;
        const selected = selectedIds.includes(node.id);
        return (
          <Paper
            key={node.id}
            elevation={0}
            onClick={event => onSelect(node.id, event.shiftKey)}
            sx={{
              ...PANEL_SURFACE,
              p: 2,
              cursor: 'pointer',
              border: selected ? '1px solid rgba(16,98,185,0.65)' : '1px solid rgba(255,255,255,0.06)',
              boxShadow: selected ? '0 0 0 4px rgba(16,98,185,0.25)' : PANEL_SURFACE.boxShadow
            }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Stack spacing={0.4}>
                <Typography variant="body1" fontWeight={600}>
                  {node.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {node.type.toUpperCase()}  -  {node.data.ip}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {node.plant}  -  {node.line}
                </Typography>
              </Stack>
              <Chip
                label={node.status}
                size="small"
                sx={{
                  borderRadius: 1.5,
                  bgcolor: alpha(palette.color, 0.18),
                  color: palette.color,
                  textTransform: 'capitalize'
                }}
              />
            </Stack>
          </Paper>
        );
      })}
    </Stack>
  );
}
function EdgeDevicePanel() {
  const theme = useTheme();
  const summaryTiles = React.useMemo(() => computeSummaryTiles(TOPOLOGY, METRICS), []);
  const isDrawer = useMediaQuery('(max-width:1279px)');
  const isListMode = useMediaQuery('(max-width:1023px)');
  const [filters, setFilters] = React.useState({
    plant: FILTER_OPTIONS.plant[0],
    line: FILTER_OPTIONS.line[0],
    status: FILTER_OPTIONS.status[0],
    protocol: FILTER_OPTIONS.protocol[0],
    search: ''
  });
  const [layers, setLayers] = React.useState(LAYER_DEFAULT);
  const [selectedIds, setSelectedIds] = React.useState(['gw-main', 'edge-a']);
  const [activeTab, setActiveTab] = React.useState('overview');
  const [summary, setSummary] = React.useState(null);
  const [layoutOverrides, setLayoutOverrides] = React.useState({});

  React.useEffect(() => {
    try {
      const stored = window.localStorage.getItem('edge-panel-layout');
      if (stored) {
        setLayoutOverrides(JSON.parse(stored));
      }
    } catch (error) {
      // ignore hydration issues in demo mode
    }
  }, []);

  React.useEffect(() => {
    window.localStorage.setItem('edge-panel-layout', JSON.stringify(layoutOverrides));
  }, [layoutOverrides]);

  const layout = React.useMemo(() => computeRadialLayout(TOPOLOGY, layoutOverrides), [layoutOverrides]);

  const filteredNodes = React.useMemo(() => {
    return layout.nodes.filter(node => {
      if (filters.status !== 'All Status' && filters.status !== node.status) {
        return false;
      }
      if (filters.protocol !== 'All Protocols' && !(node.protocols || []).includes(filters.protocol)) {
        return false;
      }
      if (filters.plant !== 'All Plants' && node.plant !== filters.plant) {
        return false;
      }
      if (filters.line !== 'All Lines' && node.line !== filters.line) {
        return false;
      }
      if (filters.search) {
        const term = filters.search.toLowerCase();
        if (
          !node.name.toLowerCase().includes(term) &&
          !node.id.toLowerCase().includes(term) &&
          !(node.data.ip || '').toLowerCase().includes(term)
        ) {
          return false;
        }
      }
      return true;
    });
  }, [layout.nodes, filters]);

  const selectedNodes = React.useMemo(() => selectedIds.map(id => layout.map.get(id)).filter(Boolean), [layout.map, selectedIds]);

  React.useEffect(() => {
    if (selectedNodes.length > 1 && activeTab === 'overview') {
      setActiveTab('compare');
    }
    if (selectedNodes.length === 1 && activeTab === 'compare') {
      setActiveTab('overview');
    }
  }, [activeTab, selectedNodes.length]);

  const handleSelect = React.useCallback((id, additive) => {
    setSelectedIds(previous => {
      if (additive) {
        if (previous.includes(id)) {
          return previous.filter(item => item !== id);
        }
        return [...previous.slice(-3), id];
      }
      return [id];
    });
  }, []);

  const handleSummarize = React.useCallback(() => {
    const selection = selectedIds.length ? selectedIds : ['gw-main'];
    setSummary({ ...SUMMARY_SAMPLE, selection, generatedAt: new Date().toISOString() });
  }, [selectedIds]);

  const handleLayerChange = React.useCallback((event, values) => {
    if (values.length) {
      setLayers(values);
    }
  }, []);

  const handlePositionChange = React.useCallback((id, position) => {
    setLayoutOverrides(previous => ({ ...previous, [id]: position }));
  }, []);

  const handleFilterChange = React.useCallback(field => event => {
    setFilters(previous => ({ ...previous, [field]: event.target.value }));
  }, []);

  const handleSearchChange = event => setFilters(previous => ({ ...previous, search: event.target.value }));

  const handleResetLayout = React.useCallback(() => setLayoutOverrides({}), []);

  const selectFieldSx = {
    minWidth: { xs: '100%', sm: 150 },
    flexGrow: { xs: 1, sm: 0 },
    '& .MuiOutlinedInput-root': {
      borderRadius: 999,
      backgroundColor: 'rgba(12,24,52,0.72)',
      transition: 'all 0.18s ease',
      '& fieldset': {
        border: '1px solid rgba(120,140,255,0.18)'
      },
      '&:hover fieldset': {
        borderColor: 'rgba(120,140,255,0.32)'
      },
      '&.Mui-focused fieldset': {
        borderColor: '#7c8dff',
        boxShadow: '0 0 0 2px rgba(124,141,255,0.24)'
      }
    },
    '& .MuiOutlinedInput-input': {
      padding: '9px 20px',
      borderRadius: 999,
      color: 'rgba(226,232,240,0.9)'
    },
    '& .MuiSelect-icon': {
      color: 'rgba(226,232,240,0.72)'
    }
  };

  return (
    <Paper elevation={0} sx={{ ...PANEL_SURFACE, p: { xs: 2.5, md: 3.5 }, height: '100%', overflow: 'auto' }}>
      <Stack spacing={3}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2.2} justifyContent="space-between">
          <Stack spacing={1}>
            <Typography variant="h6" fontWeight={600}>
              Edge estate
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Live topology beside the chat stream. Hover for stats, click to pin, Shift+Click to compare.
            </Typography>
          </Stack>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.4} alignItems={{ sm: 'center' }}>
            <ToggleButtonGroup
              value={layers}
              onChange={handleLayerChange}
              size="small"
              aria-label="Layer toggles"
              sx={{
                borderRadius: 999,
                border: '1px solid rgba(120,140,255,0.2)',
                background: 'linear-gradient(135deg, rgba(12,24,52,0.92), rgba(8,16,42,0.92))',
                p: 0.4,
                display: 'inline-flex',
                gap: 0.5,
                '& .MuiToggleButton-root': {
                  border: 'none',
                  borderRadius: 999,
                  textTransform: 'none',
                  px: 1.8,
                  fontSize: 12,
                  color: 'rgba(226,232,240,0.76)',
                  transition: 'all 0.18s ease',
                  '&.Mui-selected': {
                    background: 'linear-gradient(135deg, rgba(94,130,255,0.55), rgba(139,92,246,0.6))',
                    color: '#f8fafc',
                    boxShadow: '0 12px 24px rgba(15,32,72,0.42)'
                  },
                  '&:not(.Mui-selected):hover': {
                    backgroundColor: 'rgba(148,163,184,0.14)'
                  }
                }
              }}
            >
              <ToggleButton value="traffic">
                <LayersIcon fontSize="inherit" sx={{ mr: 0.6 }} /> Traffic
              </ToggleButton>
              <ToggleButton value="alarm">Alerts</ToggleButton>
              <ToggleButton value="protocol">Protocol</ToggleButton>
              <ToggleButton value="labels">Labels</ToggleButton>
            </ToggleButtonGroup>
            <Tooltip title="Reset layout">
              <IconButton
                size="small"
                onClick={handleResetLayout}
                sx={{
                  borderRadius: 999,
                  border: '1px solid rgba(120,140,255,0.24)',
                  backgroundColor: 'rgba(12,22,46,0.9)',
                  color: '#7c8dff',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: 'rgba(91,140,255,0.2)',
                    color: '#c7d2ff'
                  }
                }}
              >
                <RefreshIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>

        <Box
          sx={{
            display: 'grid',
            gap: { xs: 1.8, md: 2.2 },
            gridTemplateColumns: {
              xs: '1fr',
              sm: 'repeat(2, minmax(0, 1fr))',
              xl: 'repeat(3, minmax(0, 1fr))'
            }
          }}
        >
          {summaryTiles.map(tile => {
            const IconComponent = tile.icon;
            return (
              <Paper
                key={tile.key}
                elevation={0}
                sx={{
                  position: 'relative',
                  overflow: 'hidden',
                  borderRadius: 3,
                  border: '1px solid rgba(120,140,255,0.16)',
                  background: `linear-gradient(135deg, ${alpha(tile.accent, 0.18)} 0%, rgba(9,18,42,0.92) 100%)`,
                  p: { xs: 2, md: 2.4 },
                  minHeight: 148
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    inset: 0,
                    pointerEvents: 'none',
                    background: `radial-gradient(circle at 92% -18%, ${alpha(tile.accent, 0.4)} 0%, transparent 55%)`,
                    opacity: 0.6
                  }}
                />
                <Stack spacing={1.4} sx={{ position: 'relative', zIndex: 1 }}>
                  <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1.5}>
                    <Stack direction="row" spacing={1.2} alignItems="center">
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: tile.accent,
                          backgroundColor: alpha(tile.accent, 0.22),
                          boxShadow: `0 12px 24px ${alpha(tile.accent, 0.25)}`
                        }}
                      >
                        {IconComponent && <IconComponent fontSize="small" />}
                      </Box>
                      <Typography variant="overline" color="rgba(226,232,240,0.76)" sx={{ letterSpacing: 1.6 }}>
                        {tile.label}
                      </Typography>
                    </Stack>
                    {tile.secondary && (
                      <Typography variant="caption" color="rgba(226,232,240,0.7)" sx={{ textAlign: 'right' }}>
                        {tile.secondary}
                      </Typography>
                    )}
                  </Stack>
                  <Box>
                    <Typography
                      variant="h4"
                      sx={{
                        fontFamily: 'var(--font-mono, "IBM Plex Mono", "SFMono-Regular", monospace)',
                        color: tile.accent,
                        letterSpacing: -1
                      }}
                    >
                      {tile.headline}
                      {tile.caption && (
                        <Typography
                          component="span"
                          variant="subtitle2"
                          sx={{
                            ml: 1,
                            color: 'rgba(226,232,240,0.82)',
                            fontWeight: 500
                          }}
                        >
                          {tile.caption}
                        </Typography>
                      )}
                    </Typography>
                  </Box>
                  {tile.meta && tile.meta.length > 0 && (
                    <Stack direction="row" spacing={1} flexWrap="wrap">
                      {tile.meta.map(({ label, value }) => (
                        <Chip
                          key={`${tile.key}-${label}`}
                          size="small"
                          label={`${label} ${value}`}
                          sx={{
                            borderRadius: 2,
                            backgroundColor: alpha(tile.accent, 0.16),
                            color: 'rgba(226,232,240,0.88)',
                            border: 'none',
                            fontWeight: 500
                          }}
                        />
                      ))}
                    </Stack>
                  )}
                </Stack>
              </Paper>
            );
          })}
        </Box>

        <Paper
          elevation={0}
          sx={{
            borderRadius: 3,
            border: '1px solid rgba(120,140,255,0.18)',
            background: 'linear-gradient(135deg, rgba(10,26,60,0.9), rgba(7,16,36,0.92))',
            p: { xs: 1.4, md: 2 },
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            gap: { xs: 1.2, md: 1.8 },
            alignItems: { md: 'center' }
          }}
        >
          <Stack direction="row" spacing={1.2} flexWrap="wrap" sx={{ flexShrink: 0 }}>
            <FormControl size="small" sx={selectFieldSx}>
              <Select value={filters.plant} onChange={handleFilterChange('plant')}>
                {FILTER_OPTIONS.plant.map(option => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={selectFieldSx}>
              <Select value={filters.line} onChange={handleFilterChange('line')}>
                {FILTER_OPTIONS.line.map(option => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={selectFieldSx}>
              <Select value={filters.protocol} onChange={handleFilterChange('protocol')}>
                {FILTER_OPTIONS.protocol.map(option => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={selectFieldSx}>
              <Select value={filters.status} onChange={handleFilterChange('status')}>
                {FILTER_OPTIONS.status.map(option => (
                  <MenuItem key={option} value={option}>
                    {option === 'All Status' ? option : option.charAt(0).toUpperCase() + option.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Stack>
          <TextField
            size="small"
            value={filters.search}
            onChange={handleSearchChange}
            placeholder="Search device ID, asset, IP, or zone"
            sx={{
              minWidth: { xs: '100%', md: 260 },
              flexGrow: 1,
              '& .MuiOutlinedInput-root': {
                borderRadius: 999,
                backgroundColor: 'rgba(12,24,52,0.72)',
                '& fieldset': { border: '1px solid rgba(120,140,255,0.18)' },
                '&:hover fieldset': { borderColor: 'rgba(120,140,255,0.32)' },
                '&.Mui-focused fieldset': { borderColor: '#7c8dff', boxShadow: '0 0 0 2px rgba(124,141,255,0.25)' }
              },
              '& .MuiOutlinedInput-input': { py: 0.9, color: 'rgba(226,232,240,0.9)' }
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              )
            }}
          />
        </Paper>

        {isListMode ? (
          <Stack spacing={2.4}>
            <ListMode nodes={filteredNodes} selectedIds={selectedIds} onSelect={handleSelect} />
            <DetailsPane
              selected={selectedNodes}
              metricsMap={METRICS}
              tab={activeTab}
              onTabChange={(event, value) => setActiveTab(value)}
              onSummarize={handleSummarize}
              summary={summary}
              onClearSummary={() => setSummary(null)}
              layout={layout}
            />
          </Stack>
        ) : (
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { md: 'minmax(0, 0.4fr) minmax(0, 0.6fr)' },
              gap: 2.4
            }}
          >
            <GraphCanvas
              layout={layout}
              selectedIds={selectedIds}
              onSelect={handleSelect}
              filters={filters}
              layers={layers}
              searchTerm={filters.search}
              onPositionChange={handlePositionChange}
            />
            <DetailsPane
              selected={selectedNodes}
              metricsMap={METRICS}
              tab={activeTab}
              onTabChange={(event, value) => setActiveTab(value)}
              onSummarize={handleSummarize}
              summary={summary}
              onClearSummary={() => setSummary(null)}
              layout={layout}
            />
          </Box>
        )}
      </Stack>
    </Paper>
  );
}

export default function ScenarioComponent({ data }) {
  return <EdgeDevicePanel />;
}
