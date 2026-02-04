// @ts-nocheck
'use client';

import * as React from 'react';
import {
  Box,
  Button,
  Chip,
  Divider,
  Grid,
  IconButton,
  LinearProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography
} from '@mui/material';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import PendingActionsRoundedIcon from '@mui/icons-material/PendingActionsRounded';
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded';
import VisibilityRoundedIcon from '@mui/icons-material/VisibilityRounded';
import DoneAllRoundedIcon from '@mui/icons-material/DoneAllRounded';

const ingestionQueue = [
  { name: 'Line4 Calibration SOP.pdf', status: 'Ready', progress: 100, visibility: 'Org' },
  { name: 'Peak Saver Playbook.docx', status: 'Needs Review', progress: 86, visibility: 'Team' },
  { name: 'ZoneC HVAC Logs 10-12.csv', status: 'Processing', progress: 42, visibility: 'Private' }
];

const documentRows = [
  { title: 'Edge Gateway Runbook', type: 'PDF', lastIngested: '2h ago', sources: 9, visibility: 'Org', tags: ['Edge', 'Safety'] },
  { title: 'Line 4 OEE Playbook', type: 'DOCX', lastIngested: '25m ago', sources: 5, visibility: 'Team', tags: ['MES', 'Quality'] },
  { title: 'Peak Saver Cost Model', type: 'XLSX', lastIngested: '15m ago', sources: 4, visibility: 'Org', tags: ['EMS', 'Finance'] },
  { title: 'Zone C Airflow Map', type: 'PDF', lastIngested: '1d ago', sources: 3, visibility: 'Team', tags: ['BMS', 'Safety'] }
];

const statusSx = {
  Ready: { color: 'success.main', icon: <CheckCircleRoundedIcon fontSize="small" /> },
  'Needs Review': { color: 'warning.main', icon: <WarningAmberRoundedIcon fontSize="small" /> },
  Processing: { color: 'info.main', icon: <PendingActionsRoundedIcon fontSize="small" /> }
};

function VaultView() {
  return (
    <Stack spacing={4}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ xs: 'flex-start', md: 'center' }}>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Vault & Ingestion
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 720 }}>
            Ingest documents, monitor pipelines, and manage HITL approvals. Visibility badges keep access clear across org, team, and personal scopes.
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<UploadFileRoundedIcon />}>
          Upload Documents
        </Button>
      </Stack>

      <Paper
        elevation={0}
        sx={{
          borderRadius: theme => theme.shape.borderRadius,
          p: 3,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Stack spacing={3}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6" fontWeight={600}>
              Ingestion Queue
            </Typography>
            <Chip label="Neuract guardrails on" color="secondary" variant="outlined" />
          </Stack>
          <Grid container spacing={2}>
            {ingestionQueue.map(item => (
              <Grid item xs={12} md={4} key={item.name}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 2.5,
                    borderRadius: theme => theme.shape.borderRadius,
                    border: theme => `1px solid ${theme.palette.divider}`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2
                  }}
                >
                  <Typography variant="subtitle1" fontWeight={600}>
                    {item.name}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={item.progress}
                    sx={{
                      height: 8,
                      borderRadius: theme => theme.shape.borderRadius,
                      backgroundColor: 'rgba(255,255,255,0.08)'
                    }}
                  />
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                      {statusSx[item.status].icon}
                      <Typography variant="body2" sx={{ color: statusSx[item.status].color }}>
                        {item.status}
                      </Typography>
                    </Stack>
                    <Chip label={item.visibility} size="small" variant="outlined" />
                  </Stack>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Paper>

      <Paper
        elevation={0}
        sx={{
          borderRadius: theme => theme.shape.borderRadius,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Box sx={{ p: 3 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center">
            <Typography variant="h6" fontWeight={600}>
              Document Registry
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button size="small" variant="outlined">
                Filter
              </Button>
              <Button size="small" variant="outlined">
                Export
              </Button>
            </Stack>
          </Stack>
        </Box>
        <Divider />
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Title</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Last Ingested</TableCell>
              <TableCell>Sources</TableCell>
              <TableCell>Visibility</TableCell>
              <TableCell>Tags</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {documentRows.map(row => (
              <TableRow key={row.title} hover>
                <TableCell>{row.title}</TableCell>
                <TableCell>{row.type}</TableCell>
                <TableCell>{row.lastIngested}</TableCell>
                <TableCell>{row.sources}</TableCell>
                <TableCell>
                  <Chip label={row.visibility} size="small" />
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={1}>
                    {row.tags.map(tag => (
                      <Chip key={tag} label={tag} size="small" variant="outlined" />
                    ))}
                  </Stack>
                </TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={1} justifyContent="flex-end">
                    <Tooltip title="Review provenance">
                      <IconButton size="small" color="inherit">
                        <VisibilityRoundedIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Approve ingestion">
                      <IconButton size="small" color="inherit">
                        <DoneAllRoundedIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </Stack>
  );
}

export default function ScenarioComponent({ data }) {
  return <VaultView />;
}
