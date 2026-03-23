import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  color: string;
  pulsePhase: number;
}

interface Packet {
  fromIdx: number;
  toIdx: number;
  progress: number;   // 0 → 1
  speed: number;
  color: string;
  alive: boolean;
}

interface PulseRing {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  color: string;
  alive: boolean;
}

const NODE_COLORS  = ['#0ea5e9', '#6366f1', '#8b5cf6', '#06b6d4'];
const ALERT_COLORS = ['#f97316', '#ef4444'];

export function CryptoBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Particles ----------------------------------------------------------------
    const mkParticle = (): Particle => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      size: Math.random() * 1.8 + 0.8,
      opacity: Math.random() * 0.45 + 0.25,
      color: NODE_COLORS[Math.floor(Math.random() * NODE_COLORS.length)],
      pulsePhase: Math.random() * Math.PI * 2,
    });

    const count = Math.min(80, Math.floor((canvas.width * canvas.height) / 14000));
    const particles: Particle[] = Array.from({ length: count }, mkParticle);

    // Packets ------------------------------------------------------------------
    const packets: Packet[] = [];
    const spawnPacket = () => {
      const from = Math.floor(Math.random() * particles.length);
      let to = Math.floor(Math.random() * particles.length);
      while (to === from) to = Math.floor(Math.random() * particles.length);
      const isAlert = Math.random() < 0.18;
      packets.push({
        fromIdx: from,
        toIdx: to,
        progress: 0,
        speed: Math.random() * 0.008 + 0.004,
        color: isAlert
          ? ALERT_COLORS[Math.floor(Math.random() * ALERT_COLORS.length)]
          : NODE_COLORS[Math.floor(Math.random() * NODE_COLORS.length)],
        alive: true,
      });
    };
    // Seed initial packets
    for (let i = 0; i < 6; i++) spawnPacket();

    // Pulse rings --------------------------------------------------------------
    const pulses: PulseRing[] = [];
    const spawnPulse = () => {
      const p = particles[Math.floor(Math.random() * particles.length)];
      const isAlert = Math.random() < 0.2;
      pulses.push({
        x: p.x,
        y: p.y,
        radius: 4,
        maxRadius: Math.random() * 40 + 30,
        color: isAlert ? '#ef4444' : '#0ea5e9',
        alive: true,
      });
    };

    const MAX_DIST = 180;
    let frame = 0;

    // Pre-build hex grid path (static, drawn once per frame as bg)
    const HEX_R  = 32;
    const HEX_W  = HEX_R * 2;
    const HEX_H  = Math.sqrt(3) * HEX_R;
    const drawHexGrid = () => {
      ctx.save();
      ctx.strokeStyle = 'rgba(14,165,233,0.045)';
      ctx.lineWidth = 0.5;
      const cols = Math.ceil(canvas.width  / (HEX_W * 0.75)) + 2;
      const rows = Math.ceil(canvas.height / HEX_H) + 2;
      for (let col = -1; col < cols; col++) {
        for (let row = -1; row < rows; row++) {
          const cx = col * HEX_W * 0.75;
          const cy = row * HEX_H + (col % 2 === 0 ? 0 : HEX_H / 2);
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 180) * (60 * i - 30);
            const px = cx + HEX_R * Math.cos(angle);
            const py = cy + HEX_R * Math.sin(angle);
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.stroke();
        }
      }
      ctx.restore();
    };

    const draw = () => {
      frame++;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Hex grid ---------------------------------------------------------------
      drawHexGrid();

      // Update + wrap particles ------------------------------------------------
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        p.pulsePhase += 0.018;
        if (p.x < -10)              p.x = canvas.width  + 10;
        if (p.x > canvas.width + 10) p.x = -10;
        if (p.y < -10)              p.y = canvas.height + 10;
        if (p.y > canvas.height + 10) p.y = -10;
      }

      // Draw connections -------------------------------------------------------
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const d  = Math.sqrt(dx * dx + dy * dy);
          if (d < MAX_DIST) {
            const alpha = (1 - d / MAX_DIST) * 0.14;
            ctx.beginPath();
            ctx.strokeStyle = `rgba(14,165,233,${alpha})`;
            ctx.lineWidth = 0.6;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes -------------------------------------------------------------
      for (const p of particles) {
        const pulse = Math.sin(p.pulsePhase) * 0.25 + 0.75;
        const s = p.size * pulse;

        // Outer glow
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, s * 5);
        g.addColorStop(0, p.color + '55');
        g.addColorStop(1, p.color + '00');
        ctx.beginPath();
        ctx.fillStyle = g;
        ctx.arc(p.x, p.y, s * 5, 0, Math.PI * 2);
        ctx.fill();

        // Core dot
        ctx.beginPath();
        ctx.globalAlpha = p.opacity * pulse;
        ctx.fillStyle = p.color;
        ctx.arc(p.x, p.y, s, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Draw data packets ------------------------------------------------------
      for (const pkt of packets) {
        if (!pkt.alive) continue;
        pkt.progress += pkt.speed;
        if (pkt.progress >= 1) { pkt.alive = false; continue; }

        const a = particles[pkt.fromIdx];
        const b = particles[pkt.toIdx];
        const px = a.x + (b.x - a.x) * pkt.progress;
        const py = a.y + (b.y - a.y) * pkt.progress;

        // Trail
        const trailLen = 0.06;
        const t0 = Math.max(0, pkt.progress - trailLen);
        const tx = a.x + (b.x - a.x) * t0;
        const ty = a.y + (b.y - a.y) * t0;
        const grad = ctx.createLinearGradient(tx, ty, px, py);
        grad.addColorStop(0, pkt.color + '00');
        grad.addColorStop(1, pkt.color + 'cc');
        ctx.beginPath();
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.moveTo(tx, ty);
        ctx.lineTo(px, py);
        ctx.stroke();

        // Head dot
        ctx.beginPath();
        ctx.fillStyle = pkt.color;
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Remove dead packets, spawn new ones ------------------------------------
      const alive = packets.filter(p => p.alive);
      packets.length = 0;
      alive.forEach(p => packets.push(p));
      if (Math.random() < 0.025) spawnPacket();

      // Draw pulse rings -------------------------------------------------------
      if (frame % 120 === 0) spawnPulse();
      for (const ring of pulses) {
        if (!ring.alive) continue;
        ring.radius += 0.8;
        if (ring.radius > ring.maxRadius) { ring.alive = false; continue; }
        const progress = ring.radius / ring.maxRadius;
        ctx.beginPath();
        ctx.strokeStyle = ring.color + Math.round((1 - progress) * 80).toString(16).padStart(2, '0');
        ctx.lineWidth = 1;
        ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
        ctx.stroke();
      }
      const aliveRings = pulses.filter(r => r.alive);
      pulses.length = 0;
      aliveRings.forEach(r => pulses.push(r));

      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0, opacity: 0.75 }}
    />
  );
}
