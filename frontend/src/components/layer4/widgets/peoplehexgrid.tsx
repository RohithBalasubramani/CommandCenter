// @ts-nocheck
'use client';

import * as React from 'react';
import { alpha } from '@mui/material/styles';
import { Box, Chip, Stack, Typography } from '@mui/material';
import { Canvas, useFrame } from '@react-three/fiber';
import { Text, Center, OrbitControls, Float, Environment } from '@react-three/drei';
import * as THREE from 'three';

const ACCENT = '#a855f7'; // Purple for People Ops

const TEAMS = [
  { id: 't1', label: 'Module A', status: 'full', coverage: 98 },
  { id: 't2', label: 'Module B', status: 'gap', coverage: 82 },
  { id: 't3', label: 'Module C', status: 'full', coverage: 95 },
  { id: 't4', label: 'Logistics', status: 'full', coverage: 100 },
  { id: 't5', label: 'QA Shift', status: 'crit', coverage: 64 },
  { id: 't6', label: 'Maintenance', status: 'gap', coverage: 78 },
  { id: 't7', label: 'Safety', status: 'full', coverage: 100 },
  { id: 't8', label: 'Pack', status: 'full', coverage: 92 },
  { id: 't9', label: 'Clean Rm', status: 'gap', coverage: 88 },
  { id: 't10', label: 'Dispatch', status: 'full', coverage: 96 },
  { id: 't11', label: 'Admin', status: 'full', coverage: 100 },
  { id: 't12', label: 'Training', status: 'full', coverage: 100 },
  { id: 't13', label: 'Canteen', status: 'full', coverage: 100 },
  { id: 't14', label: 'Security', status: 'full', coverage: 100 },
];

const STATUS_COLORS = {
  full: '#4ade80', // Green
  gap: '#facc15',  // Yellow
  crit: '#ef4444'  // Red
};

// Hexagon math
const HEX_SIZE = 0.7;
const HEX_WIDTH = Math.sqrt(3) * HEX_SIZE;
const HEX_HEIGHT = 2 * HEX_SIZE;

function HexTile({ position, team, index }) {
  const meshRef = React.useRef(null);
  const [hovered, setHover] = React.useState(false);
  
  // Gentle breathing animation
  useFrame((state) => {
    if (!meshRef.current) return;
    const time = state.clock.getElapsedTime();
    // Randomize phase based on index
    const zOffset = Math.sin(time * 1.5 + index * 0.5) * 0.08; 
    meshRef.current.position.y = position[1] + zOffset + (hovered ? 0.2 : 0);
  });

  const color = STATUS_COLORS[team?.status] || '#ffffff';
  
  return (
    <group position={[position[0], position[1], position[2]]}>
      <mesh
        ref={meshRef}
        rotation={[Math.PI / 2, 0, 0]} // Rotate to lie flat-ish
        onPointerOver={(e) => { e.stopPropagation(); setHover(true); }}
        onPointerOut={() => setHover(false)}
      >
        <cylinderGeometry args={[HEX_SIZE * 0.95, HEX_SIZE * 0.95, 0.2, 6]} />
        <meshPhysicalMaterial
          color={hovered ? color : '#1e1b4b'} // Dark base, colored on hover
          emissive={color}
          emissiveIntensity={hovered ? 0.8 : 0.15}
          transmission={0.6}
          thickness={1.5}
          roughness={0.2}
          metalness={0.6}
          clearcoat={1}
          clearcoatRoughness={0.1}
        />
      </mesh>
      
      {/* Floating Label */}
      {team && (
        <Float speed={2} rotationIntensity={0.1} floatIntensity={0.2}>
          <group position={[0, 0.4, 0]}>
            <Center>
              <Text
                fontSize={0.18}
                color={hovered ? '#ffffff' : 'rgba(255,255,255,0.7)'}
                anchorX="center"
                anchorY="middle"
                font="https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjp-Ek-_EeA.woff"
              >
                {team.label}
              </Text>
            </Center>
            <Center position={[0, -0.22, 0]}>
              <Text
                fontSize={0.14}
                color={color}
                anchorX="center"
                anchorY="middle"
                fontWeight={800}
              >
                {team.coverage}%
              </Text>
            </Center>
          </group>
        </Float>
      )}
    </group>
  );
}

function HexGrid({ teams }) {
  // Spiral layout generator
  const grid = React.useMemo(() => {
    const items = [];
    let teamIndex = 0;
    const rings = 3; // Number of rings
    
    // Center
    if (teams[teamIndex]) items.push({ q: 0, r: 0, team: teams[teamIndex++] });

    // Rings
    for (let radius = 1; radius <= rings; radius++) {
      let q = -radius;
      let r = 0;
      let s = radius;
      
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < radius; j++) {
          // Axial to Cartesian conversion for flat-top hex
          // x = size * (3/2 * q)
          // y = size * (sqrt(3)/2 * q + sqrt(3) * r)
          
          // Using pointy top math from reference earlier but rotated:
          // Let's just use standard axial to pixel
          const x = HEX_SIZE * (3/2 * q);
          const z = HEX_SIZE * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
          
          if (teamIndex < teams.length) {
            items.push({ 
              x, 
              z, 
              team: teams[teamIndex++],
              key: `${q}-${r}`
            });
          }
          
          // Move to next hex in ring
          // Directions: +r, +s, +q, ...
          // Simplified ring logic:
          switch(i) {
            case 0: r++; s--; break;
            case 1: q++; s--; break;
            case 2: q++; r--; break;
            case 3: s++; r--; break;
            case 4: s++; q--; break;
            case 5: r++; q--; break;
          }
        }
      }
    }
    return items;
  }, [teams]);

  return (
    <group>
      {grid.map((item, idx) => (
        <HexTile 
          key={item.team.id} 
          position={[item.x, 0, item.z]} 
          team={item.team} 
          index={idx}
        />
      ))}
    </group>
  );
}

function PeopleHexGrid({ teams, accent = '#a855f7' }) {
  const gridTeams = teams && teams.length > 0 ? teams : TEAMS;
  return (
    <Box
      sx={{
        height: '100%',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
        background: 'radial-gradient(circle at 50% 50%, #1e1b4b 0%, #020617 100%)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: 'inset 0 0 40px rgba(0,0,0,0.5)'
      }}
    >
      <Canvas
        camera={{ position: [0, 5, 6], fov: 45 }}
        gl={{ antialias: true }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#0f0c29']} />
        <fog attach="fog" args={['#0f0c29', 5, 15]} />
        
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 10, 5]} intensity={2} color="#c084fc" />
        <pointLight position={[-5, 2, -5]} intensity={2} color="#4ade80" />
        
        <Float speed={1} rotationIntensity={0.1} floatIntensity={0.1}>
          <group rotation={[0.4, 0, 0]}>
            <HexGrid teams={gridTeams} />
          </group>
        </Float>

        <OrbitControls 
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.8}
          minPolarAngle={Math.PI / 4}
          maxPolarAngle={Math.PI / 2.5}
        />
        
        <Environment preset="city" />
      </Canvas>

      {/* HUD Overlays */}
      <Stack
        direction="row"
        spacing={1}
        sx={{
          position: 'absolute',
          top: 16,
          left: 16,
          zIndex: 10
        }}
      >
        <Chip 
            label="Shift Matrix" 
            size="small" 
            sx={{ 
                bgcolor: 'rgba(255,255,255,0.05)', 
                color: '#fff', 
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255,255,255,0.1)'
            }} 
        />
        <Chip 
            label="Live Pulse" 
            size="small" 
            sx={{ 
                bgcolor: alpha(accent, 0.15), 
                color: accent, 
                backdropFilter: 'blur(8px)',
                border: `1px solid ${alpha(accent, 0.2)}`
            }} 
        />
      </Stack>

      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          right: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          color: 'rgba(255,255,255,0.5)',
          pointerEvents: 'none'
        }}
      >
        <Typography variant="caption">
            Hover tiles for coverage details
        </Typography>
        <Stack direction="row" spacing={1.5}>
            <Stack direction="row" spacing={0.5} alignItems="center">
                <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#4ade80', boxShadow: '0 0 6px #4ade80' }} />
                <Typography variant="caption" sx={{ color: '#fff' }}>Full</Typography>
            </Stack>
            <Stack direction="row" spacing={0.5} alignItems="center">
                <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#facc15', boxShadow: '0 0 6px #facc15' }} />
                <Typography variant="caption" sx={{ color: '#fff' }}>Gap</Typography>
            </Stack>
            <Stack direction="row" spacing={0.5} alignItems="center">
                <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#ef4444', boxShadow: '0 0 6px #ef4444' }} />
                <Typography variant="caption" sx={{ color: '#fff' }}>Critical</Typography>
            </Stack>
        </Stack>
      </Box>
    </Box>
  );
}

export default function ScenarioComponent({ data }) {
  const teams = data?.teams ?? [];
  const accent = data?.accent ?? '#a855f7';
  return <PeopleHexGrid teams={teams} accent={accent} />;
}
