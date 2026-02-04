// @ts-nocheck
'use client';

import * as React from 'react';
import {
  Avatar,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography
} from '@mui/material';
import HealthAndSafetyRoundedIcon from '@mui/icons-material/HealthAndSafetyRounded';
import CachedRoundedIcon from '@mui/icons-material/CachedRounded';
import RouteRoundedIcon from '@mui/icons-material/RouteRounded';
import SmartToyRoundedIcon from '@mui/icons-material/SmartToyRounded';
import TuneRoundedIcon from '@mui/icons-material/TuneRounded';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import InfoRoundedIcon from '@mui/icons-material/InfoRounded';
import ChatBubbleOutlineRoundedIcon from '@mui/icons-material/ChatBubbleOutlineRounded';

const agentRegistry = [
  {
    name: 'Neuract Core',
    description: 'Primary orchestrator bridging MES, EMS, BMS, Inventory, and People with provenance-first responses.',
    model: 'Neuract-edge 12B',
    health: 99,
    state: 'Healthy',
    routes: ['Command Center', 'Signal Panel'],
    slashCommands: ['/neuract explain', '/neuract trace', '/neuract recap']
  },
  {
    name: 'MES Sentinel',
    description: 'Line diagnostics, OEE analytics, and predictive maintenance recommendations.',
    model: 'Forge-8',
    health: 95,
    state: 'Healthy',
    routes: ['Command Center', 'MES Lens'],
    slashCommands: ['/mes downtime', '/mes quality', '/mes workorder']
  },
  {
    name: 'EMS Optimizer',
    description: 'Balances load curves, tariffs, and demand response automation for peak shaving.',
    model: 'Edge-DR 6B',
    health: 94,
    state: 'Monitoring',
    routes: ['Command Center', 'EMS Lens'],
    slashCommands: ['/ems peak', '/ems cost', '/ems policy']
  },
  {
    name: 'BMS Steward',
    description: 'Controls environment schedules, anomaly detection, and comfort guardrails.',
    model: 'Therma-5',
    health: 91,
    state: 'Healthy',
    routes: ['Command Center', 'BMS Lens'],
    slashCommands: ['/bms zone', '/bms anomaly', '/bms override']
  },
  {
    name: 'Inventory Sentinel',
    description: 'Tracks critical spares, asset health, and work order readiness.',
    model: 'Stock-3',
    health: 88,
    state: 'Needs attention',
    routes: ['Command Center', 'Inventory & Assets Lens'],
    slashCommands: ['/inventory critical', '/inventory reorder', '/inventory asset']
  }
];

const routingTable = [
  { intent: 'line downtime', route: 'MES Sentinel', fallback: 'Neuract Core', sla: '< 1.0s' },
  { intent: 'peak demand forecast', route: 'EMS Optimizer', fallback: 'Neuract Core', sla: '< 1.2s' },
  { intent: 'zone humidity alert', route: 'BMS Steward', fallback: 'Neuract Core', sla: '< 0.9s' },
  { intent: 'critical spare coverage', route: 'Inventory Sentinel', fallback: 'Neuract Core', sla: '< 1.1s' },
  { intent: 'cross-domain narrative', route: 'Neuract Core', fallback: 'Operator review', sla: '< 1.4s' }
];

function AgentsView() {
  const [chatAgent, setChatAgent] = React.useState(null);
  const [chatMessage, setChatMessage] = React.useState('');

  const handleOpenChat = React.useCallback(agent => {
    setChatAgent(agent);
    setChatMessage('');
  }, []);

  const handleCloseChat = React.useCallback(() => {
    setChatAgent(null);
    setChatMessage('');
  }, []);

  const handleSubmitChat = React.useCallback(() => {
    // Placeholder for integration with scoped agent chat flow
    setChatAgent(null);
    setChatMessage('');
  }, []);

  return (
    <Stack spacing={4}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ xs: 'flex-start', md: 'center' }}>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Agents & Routing
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 720 }}>
            Monitor Neuract and its domain specialists, keep routing hints aligned, and review sandbox health across MES, EMS, BMS, Inventory, and People.
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<AddRoundedIcon />}>
          Register Agent
        </Button>
      </Stack>

      <Grid container spacing={3}>
        {agentRegistry.map(agent => (
          <Grid item xs={12} md={4} key={agent.name}>
            <Paper
              elevation={0}
              sx={{
                p: 3,
                borderRadius: theme => theme.shape.borderRadius,
                border: theme => `1px solid ${theme.palette.divider}`,
                display: 'flex',
                flexDirection: 'column',
                gap: 2
              }}
            >
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Avatar
                  variant="rounded"
                  sx={{
                    bgcolor: 'rgba(102,224,255,0.18)',
                    color: 'primary.main'
                  }}
                >
                  <RouteRoundedIcon />
                </Avatar>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="subtitle1" fontWeight={600}>
                    {agent.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Model: {agent.model}
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Tooltip title={`Chat with ${agent.name}`}>
                    <IconButton color="inherit" size="small" onClick={() => handleOpenChat(agent)}>
                      <ChatBubbleOutlineRoundedIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Chip label={agent.state} size="small" variant="outlined" />
                </Stack>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                {agent.description}
              </Typography>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Health
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={agent.health}
                  sx={{
                    mt: 1,
                    height: 8,
                    borderRadius: theme => theme.shape.borderRadius
                  }}
                />
              </Box>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {agent.routes.map(route => (
                  <Chip key={route} label={route} size="small" icon={<SmartToyRoundedIcon sx={{ fontSize: 14 }} />} />
                ))}
              </Stack>
              <Stack direction="row" spacing={1}>
                <Button variant="outlined" size="small" startIcon={<HealthAndSafetyRoundedIcon />}>
                  View health
                </Button>
                <Button variant="text" size="small" startIcon={<TuneRoundedIcon />}>
                  Adjust routing
                </Button>
              </Stack>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Paper
        elevation={0}
        sx={{
          borderRadius: theme => theme.shape.borderRadius,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Stack spacing={3} sx={{ p: 3 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6" fontWeight={600}>
              Routing Overview
            </Typography>
            <Button size="small" variant="outlined" startIcon={<CachedRoundedIcon />}>
              Sync config
            </Button>
          </Stack>
          <Divider />
          <Stack spacing={2}>
            {routingTable.map(route => (
              <Stack
                key={route.intent}
                direction={{ xs: 'column', md: 'row' }}
                spacing={2}
                alignItems={{ xs: 'flex-start', md: 'center' }}
                sx={{
                  borderRadius: theme => theme.shape.borderRadius,
                  p: 2,
                  border: theme => `1px solid ${theme.palette.divider}`
                }}
              >
                <Chip label={`Intent - ${route.intent}`} color="secondary" variant="outlined" />
                <Stack direction="row" spacing={1} alignItems="center" sx={{ flexGrow: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Route
                  </Typography>
                  <Chip label={route.route} color="primary" variant="outlined" />
                  <Typography variant="body2" color="text.secondary">
                    Fallback
                  </Typography>
                  <Chip label={route.fallback} variant="outlined" />
                </Stack>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Typography variant="caption" color="text.secondary">
                    SLA
                  </Typography>
                  <Chip label={route.sla} variant="outlined" size="small" />
                  <Tooltip title="View trace policy">
                    <IconButton color="inherit" size="small">
                      <InfoRoundedIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>
              </Stack>
            ))}
          </Stack>
        </Stack>
      </Paper>

      <Dialog open={Boolean(chatAgent)} onClose={handleCloseChat} maxWidth="sm" fullWidth>
        {chatAgent && (
          <>
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <RouteRoundedIcon color="secondary" />
              <span>
                Chat with {chatAgent.name}
                <Typography component="span" variant="body2" color="text.secondary" sx={{ display: 'block' }}>
                  Scoped to {chatAgent.routes.join(' | ')}
                </Typography>
              </span>
            </DialogTitle>
            <DialogContent dividers sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Use agent-specific slash commands to stay in context.
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {chatAgent.slashCommands.map(command => (
                  <Chip key={command} label={command} size="small" variant="outlined" />
                ))}
              </Stack>
              <TextField
                multiline
                minRows={4}
                label="Message"
                placeholder="Type your question or action request…"
                value={chatMessage}
                onChange={event => setChatMessage(event.target.value)}
                fullWidth
              />
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseChat}>Cancel</Button>
              <Button onClick={handleSubmitChat} variant="contained" disabled={!chatMessage.trim()}>
                Send to {chatAgent.name}
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Stack>
  );
}

export default function ScenarioComponent({ data }) {
  return <AgentsView />;
}
