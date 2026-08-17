import { useMemo, useState } from 'react';
import { Network, Grid, Sparkles } from 'lucide-react';
import { getSpatialBilinearScores, MazeGrid } from '../model/labyrinth_transformer';

interface SpatialBiasMapProps {
  maze: MazeGrid;
}

export default function SpatialBiasMap({ maze }: SpatialBiasMapProps) {
  const [hoveredCells, setHoveredCells] = useState<{ u: number; v: number } | null>(null);

  const { scores, min, max } = useMemo(() => {
    return getSpatialBilinearScores();
  }, []);

  const getCellColor = (val: number) => {
    const norm = (val - min) / (max - min || 1);
    const r = Math.round(16 + norm * 30);
    const g = Math.round(24 + norm * 180);
    const b = Math.round(32 + norm * 200);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 shadow-xl flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Network className="w-5 h-5 text-cyan-400" /> Learned 2D Spatial &amp; Ordering Bias Map
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Bilinear Query-Key interaction matrix: Bias(u, v) = PE_u W_q W_k^T PE_v^T across 36 spatial grid cells ({maze.name})
          </p>
        </div>

        <div className="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-850 text-xs">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-zinc-400">Score Range:</span>
          <span className="font-mono text-zinc-200 font-bold">[{min.toFixed(2)}, {max.toFixed(2)}]</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* 36x36 Heatmap Matrix */}
        <div className="lg:col-span-8 bg-zinc-950 p-4 rounded-xl border border-zinc-850 flex flex-col items-center">
          <div className="flex w-full items-center justify-between text-[10px] text-zinc-500 font-mono mb-2">
            <span>Key Position Cell Index v (0 to 35) &rarr;</span>
            {hoveredCells && (
              <span className="text-indigo-400 font-bold bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                Cell u({hoveredCells.u}) &rarr; Cell v({hoveredCells.v}): {scores[hoveredCells.u][hoveredCells.v].toFixed(3)}
              </span>
            )}
          </div>

          <div className="w-full aspect-square max-w-[480px] grid grid-cols-36 gap-[1px] p-1 bg-zinc-900 border border-zinc-800 rounded">
            {scores.map((row, u) =>
              row.map((val, v) => (
                <div
                  key={`${u}-${v}`}
                  onMouseEnter={() => setHoveredCells({ u, v })}
                  onMouseLeave={() => setHoveredCells(null)}
                  style={{ backgroundColor: getCellColor(val) }}
                  className="w-full aspect-square rounded-[0.5px] hover:ring-1 hover:ring-white transition-all cursor-crosshair"
                  title={`Query Cell u(${u}) -> Key Cell v(${v}) | Score: ${val.toFixed(2)}`}
                />
              ))
            )}
          </div>

          <div className="flex w-full justify-between items-center text-[10px] font-mono text-zinc-500 mt-2">
            <span>&uarr; Query Position Cell Index u (0 to 35)</span>
            <div className="flex items-center gap-2">
              <span>Min ({min.toFixed(1)})</span>
              <div className="w-20 h-2 bg-gradient-to-r from-[rgb(16,24,32)] to-[rgb(46,204,232)] rounded" />
              <span>Max ({max.toFixed(1)})</span>
            </div>
          </div>
        </div>

        {/* Informational Side Panel */}
        <div className="lg:col-span-4 bg-zinc-950 p-5 rounded-xl border border-zinc-850 flex flex-col justify-between h-full space-y-4">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-300 mb-2 flex items-center gap-1.5">
              <Grid className="w-4 h-4 text-indigo-400" /> Spatial Connectivity &amp; Distance Decay
            </h3>
            <div className="text-xs text-zinc-400 leading-relaxed space-y-2">
              <p>
                The block-diagonal band structure in the 36x36 matrix demonstrates how the Transformer encodes 2D grid topology:
              </p>
            </div>
          </div>

          <div className="space-y-2.5 text-[11px] text-zinc-400">
            <div className="p-2.5 bg-zinc-900/60 rounded-lg border border-zinc-800">
              <strong className="text-cyan-400 block mb-1">1. Main Diagonal Band</strong>
              Represents self-attention bias for nearby cells in the same row/column. High bilinear scores align adjacent spatial steps.
            </div>

            <div className="p-2.5 bg-zinc-900/60 rounded-lg border border-zinc-800">
              <strong className="text-indigo-400 block mb-1">2. Periodic Row Offsets</strong>
              Secondary parallel diagonals correspond to vertical step jumps (moving &plusmn;6 steps up or down in the 6x6 grid).
            </div>

            <div className="p-2.5 bg-zinc-900/60 rounded-lg border border-zinc-800">
              <strong className="text-emerald-400 block mb-1">3. Topological Alignment</strong>
              Interactions between choice cells are biased to maintain long-range topological reachability over walls.
            </div>
          </div>

          <div className="text-[10px] text-zinc-500 italic border-t border-zinc-900 pt-3">
            *Computed from Layer 1 attention weight projections (W_q, W_k) and 2D sinusoidal spatial embeddings.
          </div>
        </div>
      </div>
    </div>
  );
}
