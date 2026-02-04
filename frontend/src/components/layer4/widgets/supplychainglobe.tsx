// @ts-nocheck
'use client';

import * as React from 'react';
import { alpha } from '@mui/material/styles';
import { Box, Chip, Stack, Typography } from '@mui/material';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, OrbitControls, Points, PointMaterial, useTexture, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// Reliable texture URLs from Three.js examples
const TEXTURES = {
  map: 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg',
  bump: 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_normal_2048.jpg',
  specular: 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_specular_2048.jpg',
  clouds: 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_clouds_1024.png'
};

const statusPalette = {
  live: '#00f0ff',
  held: '#ff4d00',
  delivered: '#00ff9d'
};

const toCartesian = (lat, lng, radius = 1) => {
  const phi = THREE.MathUtils.degToRad(90 - lat);
  const theta = THREE.MathUtils.degToRad(lng + 180);
  const r = radius;
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta)
  );
};

function Arc({ lane, radius = 1, thickness = 1.5 }) {
  const lineRef = React.useRef(null);
  const { color, from, to } = lane;

  const points = React.useMemo(() => {
    const start = toCartesian(from.lat, from.lng, radius);
    const end = toCartesian(to.lat, to.lng, radius);
    const mid = start
      .clone()
      .add(end)
      .multiplyScalar(0.5)
      .setLength(radius * (1.25 + Math.random() * 0.1)); 
    return [start, mid, end];
  }, [from.lat, from.lng, radius, to.lat, to.lng]);

  useFrame((state, delta) => {
    if (lineRef.current?.material) {
      lineRef.current.material.dashOffset -= delta * 1.2;
    }
  });

  return (
    <group>
      <Line
        ref={lineRef}
        points={points}
        color={color}
        linewidth={thickness}
        dashed
        dashSize={0.2}
        gapSize={0.1}
        transparent
        opacity={0.9}
        toneMapped={false} 
      />
      <mesh position={points[2]} scale={0.04}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
      <mesh position={points[0]} scale={0.02}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial color={color} opacity={0.6} transparent />
      </mesh>
    </group>
  );
}

function NodeField({ lanes, radius }) {
  const positions = React.useMemo(() => {
    const vectors = [];
    lanes.forEach(lane => {
      vectors.push(toCartesian(lane.from.lat, lane.from.lng, radius * 1.01));
      vectors.push(toCartesian(lane.to.lat, lane.to.lng, radius * 1.01));
    });
    const unique = new Map();
    vectors.forEach(vec => {
      unique.set(vec.toArray().join(','), vec);
    });
    return Float32Array.from(
      Array.from(unique.values()).flatMap(vec => vec.toArray())
    );
  }, [lanes, radius]);

  if (!positions.length) return null;

  return (
    <Points positions={positions} stride={3} key={`${positions.length}-${lanes.length}`}>
      <PointMaterial
        transparent
        color="#ffffff"
        size={0.06}
        sizeAttenuation
        depthWrite={false}
        opacity={0.8}
        toneMapped={false}
      />
    </Points>
  );
}

function GlobeMesh({ radius }) {
  const textures = useTexture(TEXTURES);

  // Ensure color space is correct
  React.useLayoutEffect(() => {
    Object.values(textures).forEach(tex => {
      if (tex) tex.colorSpace = THREE.SRGBColorSpace;
    });
  }, [textures]);

  return (
    <group>
      {/* 1. The Base Earth Sphere - High Visibility */}
      <mesh scale={radius}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          map={textures.map}
          normalMap={textures.bump}
          normalScale={[0.5, 0.5]} // Enhance bumpiness
          roughness={0.6}
          metalness={0.1}
          color="#4466aa" // Tint it blue
          emissive="#112244" // Self-illuminated to prevent "black blob" look
          emissiveIntensity={0.4}
        />
      </mesh>

      {/* 2. Clouds / Atmosphere Layer */}
      <mesh scale={radius * 1.015}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          map={textures.clouds}
          transparent
          opacity={0.3}
          blending={THREE.AdditiveBlending}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      {/* 3. Atmosphere Glow (Fresnel-ish) */}
      <mesh scale={radius * 1.15}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshBasicMaterial
          color="#4080ff"
          transparent
          opacity={0.1}
          side={THREE.BackSide} // Render on the inside of a larger sphere
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function Scene({ lanes, accent }) {
  const radius = 1.3;
  const clampedLanes = lanes.slice(0, 20);

  return (
    <>
      {/* Lighting Setup - Bright and Dramatic */}
      <ambientLight intensity={1.5} color="#ffffff" />
      <directionalLight position={[10, 10, 5]} intensity={2} color="#ffffff" />
      <directionalLight position={[-10, -5, -5]} intensity={3} color={accent} />
      <pointLight position={[0, 5, 0]} intensity={2} color="#4060ff" distance={10} />

      {/* Fog to blend the globe bottom into the void */}
      <fog attach="fog" args={['#020408', 5, 18]} />

      <group position={[0, -0.2, 0]}>
        <GlobeMesh radius={radius} />
        <NodeField lanes={clampedLanes} radius={radius} />
        {clampedLanes.map(lane => (
          <Arc key={lane.id} lane={lane} radius={radius} thickness={2} />
        ))}
      </group>

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.6}
        minPolarAngle={Math.PI * 0.25}
        maxPolarAngle={Math.PI * 0.75}
        dampingFactor={0.05}
        rotateSpeed={0.4}
      />
    </>
  );
}

function SupplyChainGlobe({ lanes = [], accent = '#22d3ee' }) {
  const coloredLanes = React.useMemo(
    () =>
      (lanes.length ? lanes : []).map(lane => ({
        ...lane,
        color: lane.color || statusPalette[lane.status] || accent
      })),
    [accent, lanes]
  );

  return (
    <Box
      sx={{
        height: '100%',
        borderRadius: 3,
        overflow: 'hidden',
        position: 'relative',
        background: 'radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%)',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: 'inset 0 0 40px rgba(0,0,0,0.5)'
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 4.5], fov: 45 }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.3
        }}
        style={{ width: '100%', height: '100%' }}
        dpr={[1, 2]}
      >
        <Scene lanes={coloredLanes} accent={accent} />
      </Canvas>

      {/* UI Overlays */}
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
          label="Globe View"
          size="small"
          sx={{
            bgcolor: 'rgba(255,255,255,0.05)',
            color: '#fff',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255,255,255,0.1)'
          }}
        />
        <Chip
          label="Live"
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
          pointerEvents: 'none'
        }}
      >
        <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
          Interactive 3D View
        </Typography>
        <Stack direction="row" spacing={0.5}>
          {Object.entries(statusPalette).map(([key, value]) => (
            <Box
              key={key}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                bgcolor: 'rgba(0,0,0,0.4)',
                px: 1,
                py: 0.5,
                borderRadius: 1,
                border: '1px solid rgba(255,255,255,0.05)'
              }}
            >
              <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: value, boxShadow: `0 0 6px ${value}` }} />
              <Typography variant="caption" sx={{ color: '#eee', fontWeight: 500, lineHeight: 1 }}>
                {key}
              </Typography>
            </Box>
          ))}
        </Stack>
      </Box>
    </Box>
  );
}

export default function ScenarioComponent({ data }) {
  const lanes = data?.lanes ?? [];
  const accent = data?.accent ?? '#22d3ee';
  return <SupplyChainGlobe lanes={lanes} accent={accent} />;
}
