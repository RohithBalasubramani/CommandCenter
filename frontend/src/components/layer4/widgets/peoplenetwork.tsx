// @ts-nocheck
'use client';

import * as React from 'react';
import { alpha } from '@mui/material/styles';
import { Box, Chip, Stack, Typography } from '@mui/material';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Float, Stars } from '@react-three/drei';
import * as THREE from 'three';

const DEFAULT_DEPARTMENTS = [
  { id: 'module', label: 'Module Lines', color: '#f472b6', count: 28, pos: [2, 0, 0] },
  { id: 'cell', label: 'Cell Fab', color: '#38bdf8', count: 22, pos: [-2, 1, 0] },
  { id: 'qa', label: 'QA & Compliance', color: '#a78bfa', count: 14, pos: [0, -2, 1] },
  { id: 'ops', label: 'Facility Ops', color: '#facc15', count: 10, pos: [0, 2, -1] }
];

// Deterministic pseudo-random from seed for stable renders
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

function generateNetwork(departments) {
  const rand = seededRandom(42);
  const nodes = [];
  const links = [];

  departments.forEach((dept) => {
    const [cx, cy, cz] = dept.pos;
    for (let i = 0; i < dept.count; i++) {
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos((rand() * 2) - 1);
      const r = Math.pow(rand(), 0.5) * 1.5;
      const x = cx + r * Math.sin(phi) * Math.cos(theta);
      const y = cy + r * Math.sin(phi) * Math.sin(theta);
      const z = cz + r * Math.cos(phi);
      const position = new THREE.Vector3(x, y, z);

      nodes.push({
        id: `${dept.id}-${i}`,
        position,
        deptId: dept.id,
        color: dept.color,
        size: rand() * 0.06 + 0.03,
        pulseSpeed: rand() * 1.5 + 0.5,
        pulseOffset: rand() * 10
      });

      if (i > 0) {
        const targetIdx = nodes.length - 2;
        if (nodes[targetIdx].deptId === dept.id && rand() > 0.4) {
          links.push({ start: position, end: nodes[targetIdx].position, color: dept.color, opacity: 0.15 });
        }
      }
      if (i > 5 && rand() > 0.7) {
        const targetIdx = nodes.length - Math.floor(rand() * 5) - 2;
        if (targetIdx >= 0 && nodes[targetIdx].deptId === dept.id) {
          links.push({ start: position, end: nodes[targetIdx].position, color: dept.color, opacity: 0.1 });
        }
      }
    }
  });

  for (let i = 0; i < 15; i++) {
    const idxA = Math.floor(rand() * nodes.length);
    const idxB = Math.floor(rand() * nodes.length);
    if (nodes[idxA].deptId !== nodes[idxB].deptId) {
      links.push({ start: nodes[idxA].position, end: nodes[idxB].position, color: '#ffffff', opacity: 0.05 });
    }
  }

  return { nodes, links };
}

function Particle({ node }) {
  const mesh = React.useRef();
  const [hovered, setHovered] = React.useState(false);

  useFrame((state) => {
    if (!mesh.current) return;
    const t = state.clock.elapsedTime;
    // Breathing effect based on time + offset
    const scale = 1 + Math.sin(t * node.pulseSpeed + node.pulseOffset) * 0.3;
    mesh.current.scale.setScalar(hovered ? 2.5 : scale);
  });

  return (
    <mesh
      ref={mesh}
      position={node.position}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[node.size, 16, 16]} />
      <meshStandardMaterial 
        color={node.color} 
        emissive={node.color}
        emissiveIntensity={hovered ? 3 : 1.5}
        toneMapped={false}
      />
      {hovered && (
        <Html distanceFactor={8} zIndexRange={[100, 0]}>
          <div style={{ 
            padding: '6px 10px', 
            background: 'rgba(5, 10, 20, 0.9)', 
            color: 'white', 
            borderRadius: '6px',
            fontSize: '11px',
            fontWeight: 600,
            whiteSpace: 'nowrap',
            border: `1px solid ${node.color}`,
            boxShadow: `0 0 10px ${alpha(node.color, 0.4)}`,
            pointerEvents: 'none',
            transform: 'translate3d(-50%, -140%, 0)'
          }}>
            {node.deptId.toUpperCase()} • Active
          </div>
        </Html>
      )}
    </mesh>
  );
}

function Connections({ links }) {
  const geometry = React.useMemo(() => {
    const points = [];
    const colors = [];

    links.forEach(link => {
      points.push(link.start.x, link.start.y, link.start.z);
      points.push(link.end.x, link.end.y, link.end.z);
      
      const c = new THREE.Color(link.color);
      // Add some alpha to vertex colors if material supports it, 
      // but LineBasicMaterial uses global opacity. We'll stick to color.
      colors.push(c.r, c.g, c.b);
      colors.push(c.r, c.g, c.b);
    });
    
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geo;
  }, [links]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial 
        vertexColors 
        transparent 
        opacity={0.12} 
        depthWrite={false} 
        blending={THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}

function NetworkScene({ nodes, links }) {
  return (
    <group>
      <Float speed={1} rotationIntensity={0.2} floatIntensity={0.2}>
        <group rotation={[0, 0, 0]}>
            {nodes.map(node => (
            <Particle key={node.id} node={node} />
            ))}
            <Connections links={links} />
        </group>
      </Float>
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#ffffff" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4ade80" />
    </group>
  );
}

function PeopleNetwork({ coverage = 86, departments: externalDepts }) {
  const departments = externalDepts || DEFAULT_DEPARTMENTS;
  const { nodes, links } = React.useMemo(() => generateNetwork(departments), [departments]);

  return (
    <Box
      sx={{
        height: '100%',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
        background: 'radial-gradient(circle at 50% 50%, #1e1b4b 0%, #020617 100%)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: 'inset 0 0 60px rgba(0,0,0,0.6)'
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 7], fov: 45 }}
        gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
        dpr={[1, 2]}
      >
        <fog attach="fog" args={['#020617', 6, 20]} />
        <NetworkScene nodes={nodes} links={links} />
        <OrbitControls 
            enableZoom={false} 
            enablePan={false} 
            autoRotate 
            autoRotateSpeed={0.4} 
            minPolarAngle={Math.PI * 0.2}
            maxPolarAngle={Math.PI * 0.8}
        />
      </Canvas>

      {/* Overlay UI - Glassmorphic Cards */}
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
          label="Talent Constellation"
          size="small"
          sx={{
            bgcolor: 'rgba(255,255,255,0.05)',
            color: '#fff',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255,255,255,0.1)',
            fontWeight: 600
          }}
        />
        <Chip
            label="Live"
            size="small"
            sx={{
                bgcolor: alpha('#a855f7', 0.15), 
                color: '#a855f7', 
                backdropFilter: 'blur(8px)',
                border: `1px solid ${alpha('#a855f7', 0.2)}`,
                fontWeight: 600
            }} 
        />
      </Stack>

      {/* Stats Overlay */}
      <Box
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          textAlign: 'right'
        }}
      >
         <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)', display: 'block', mb: 0.5 }}>
            Shift Coverage
         </Typography>
         <Typography variant="h4" sx={{ color: '#fff', fontWeight: 700, lineHeight: 1, textShadow: '0 0 20px rgba(168,85,247,0.5)' }}>
            {coverage}%
         </Typography>
      </Box>

      <Box
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          right: 16,
          pointerEvents: 'none'
        }}
      >
        <Stack direction="row" spacing={3} justifyContent="center" sx={{ flexWrap: 'wrap', gap: 1 }}>
          {departments.map(dept => (
            <Stack key={dept.id} direction="row" alignItems="center" spacing={0.8} sx={{
                bgcolor: 'rgba(0,0,0,0.3)',
                px: 1,
                py: 0.5,
                borderRadius: 99
            }}>
              <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: dept.color, boxShadow: `0 0 8px ${dept.color}` }} />
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.8)', fontWeight: 500 }}>
                {dept.label}
              </Typography>
            </Stack>
          ))}
        </Stack>
      </Box>
    </Box>
  );
}

export default function ScenarioComponent({ data }) {
  const coverage = data?.coverage ?? 86;
  const departments = data?.departments;
  return <PeopleNetwork coverage={coverage} departments={departments} />;
}
