// @ts-nocheck
'use client';

import * as React from 'react';
import {
  Box,
  Button,
  Chip,
  Divider,
  Grid,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography
} from '@mui/material';
import PersonAddAltRoundedIcon from '@mui/icons-material/PersonAddAltRounded';
import SecurityRoundedIcon from '@mui/icons-material/SecurityRounded';
import ManageAccountsRoundedIcon from '@mui/icons-material/ManageAccountsRounded';
import TaskAltRoundedIcon from '@mui/icons-material/TaskAltRounded';
import Avatar from '@mui/material/Avatar';

const roleMatrix = [
  { role: 'Operator', commandCenter: 'View/Chat', vault: 'Read', agents: 'View', writeActions: 'HITL' },
  { role: 'Domain Lead', commandCenter: 'View/Chat', vault: 'Read/Approve', agents: 'View', writeActions: 'Approve' },
  { role: 'Admin', commandCenter: 'Full', vault: 'Full', agents: 'Manage', writeActions: 'Execute' }
];

const teamMembers = [
  { name: 'Amelia Chen', role: 'Production Supervisor', status: 'Active', visibility: ['MES', 'Inventory & Assets'] },
  { name: 'Luis Ortega', role: 'Energy Analyst', status: 'Pending Invite', visibility: ['EMS', 'BMS'] },
  { name: 'Zara Singh', role: 'People Ops', status: 'Active', visibility: ['People', 'Command Center'] }
];

function PeopleView() {
  return (
    <Stack spacing={4}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} alignItems={{ xs: 'flex-start', md: 'center' }}>
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            People & Permissions
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 720 }}>
            Access drives visibility across chat history, widgets, and write approvals. Keep coverage aligned with Neuract guardrails.
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<PersonAddAltRoundedIcon />}>
          Invite Teammate
        </Button>
      </Stack>

      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: theme => theme.shape.borderRadius,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Stack spacing={3}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <SecurityRoundedIcon color="secondary" />
            <Typography variant="h6" fontWeight={600}>
              Role Matrix
            </Typography>
          </Stack>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Role</TableCell>
                <TableCell>Command Center</TableCell>
                <TableCell>Vault</TableCell>
                <TableCell>Agents</TableCell>
                <TableCell>Write Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {roleMatrix.map(row => (
                <TableRow key={row.role}>
                  <TableCell>{row.role}</TableCell>
                  <TableCell>{row.commandCenter}</TableCell>
                  <TableCell>{row.vault}</TableCell>
                  <TableCell>{row.agents}</TableCell>
                  <TableCell>{row.writeActions}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Stack>
      </Paper>

      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: theme => theme.shape.borderRadius,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Stack spacing={3}>
          <Stack direction="row" spacing={1.5} alignItems="center">
            <ManageAccountsRoundedIcon color="secondary" />
            <Typography variant="h6" fontWeight={600}>
              Team Directory
            </Typography>
          </Stack>
          <Grid container spacing={2}>
            {teamMembers.map(member => (
              <Grid item xs={12} md={4} key={member.name}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 2.5,
                    borderRadius: theme => theme.shape.borderRadius,
                    border: theme => `1px solid ${theme.palette.divider}`,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 1.5
                  }}
                >
                  <Stack direction="row" spacing={1.5} alignItems="center">
                    <Avatar>{member.name.slice(0, 1)}</Avatar>
                    <Box>
                      <Typography variant="subtitle1" fontWeight={600}>
                        {member.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {member.role}
                      </Typography>
                    </Box>
                    <Chip label={member.status} size="small" variant="outlined" sx={{ ml: 'auto' }} />
                  </Stack>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {member.visibility.map(scope => (
                      <Chip key={scope} label={scope} size="small" variant="outlined" />
                    ))}
                  </Stack>
                  <Button size="small" variant="text" startIcon={<TaskAltRoundedIcon />}>
                    Manage access
                  </Button>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Paper>

      <Paper
        elevation={0}
        sx={{
          p: 3,
          borderRadius: theme => theme.shape.borderRadius,
          border: theme => `1px solid ${theme.palette.divider}`
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h6" fontWeight={600}>
              Invite by Email
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Shared chats default to private. Choose project or org scope explicitly.
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
              <TextField placeholder="name@company.com" fullWidth />
              <Button variant="contained">Send Invite</Button>
            </Stack>
          </Grid>
        </Grid>
      </Paper>
    </Stack>
  );
}

export default function ScenarioComponent({ data }) {
  return <PeopleView />;
}
