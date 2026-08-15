import { useState, useMemo } from 'react';
import { Network, Layers, Info } from 'lucide-react';
import { getRawWeights, computePCA } from '../model/transformer';

type Mode = 'tokens_only' | 'token_plus_pos' | 'all_combinations';

interface PointData {
  id: string;
  digit: number;
  posIdx?: number;
  label: string;
  rawVec: number[];
  norm: number;
  x: number;
  y: number;
}

export default function EmbeddingScatterPlot() {
  const [mode, setMode] = useState<Mode>('token_plus_pos');
  const [selectedPos, setSelectedPos] = useState<number>(0);
  const [hoveredPoint, setHoveredPoint] = useState<PointData | null>(null);

  const weights = getRawWeights();

  const digitColors = [
    '#ef4444', // 0 red
    '#f97316', // 1 orange
    '#f59e0b', // 2 amber
    '#eab308', // 3 yellow
    '#84cc16', // 4 lime
    '#10b981', // 5 emerald
    '#06b6d4', // 6 cyan
    '#3b82f6', // 7 blue
    '#6366f1', // 8 indigo
    '#a855f7', // 9 purple
  ];

  const pcaResult = useMemo(() => {
    const rawPoints: { id: string; digit: number; posIdx?: number; label: string; vec: number[] }[] = [];

    if (mode === 'tokens_only') {
      for (let d = 0; d < 10; d++) {
        rawPoints.push({
          id: `t_${d}`,
          digit: d,
          label: `Digit ${d}`,
          vec: weights.embedding[d],
        });
      }
    } else if (mode === 'token_plus_pos') {
      const pe = weights.pe[selectedPos];
      for (let d = 0; d < 10; d++) {
        const sumVec = weights.embedding[d].map((v, idx) => v + pe[idx]);
        rawPoints.push({
          id: `t_${d}_p_${selectedPos}`,
          digit: d,
          posIdx: selectedPos,
          label: `Digit ${d} @ P${selectedPos + 1}`,
          vec: sumVec,
        });
      }
    } else {
      // all_combinations: 10 digits x 5 positions
      for (let d = 0; d < 10; d++) {
        for (let p = 0; p < 5; p++) {
          const sumVec = weights.embedding[d].map((v, idx) => v + weights.pe[p][idx]);
          rawPoints.push({
            id: `t_${d}_p_${p}`,
            digit: d,
            posIdx: p,
            label: `${d} (P${p + 1})`,
            vec: sumVec,
          });
        }
      }
    }

    const pca = computePCA(rawPoints.map(p => p.vec));

    const pointDataList: PointData[] = rawPoints.map((p, idx) => {
      const vec = p.vec;
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      return {
        id: p.id,
        digit: p.digit,
        posIdx: p.posIdx,
        label: p.label,
        rawVec: vec,
        norm,
        x: pca.coords[idx]?.x || 0,
        y: pca.coords[idx]?.y || 0,
      };
    });

    return {
      points: pointDataList,
      varExplained: pca.varianceExplained,
    };
  }, [mode, selectedPos, weights]);

  // SVG bounds and scales
  const svgWidth = 600;
  const svgHeight = 400;
  const padding = 50;

  const { minX, maxX, minY, maxY } = useMemo(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    pcaResult.points.forEach(p => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });

    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    return {
      minX: minX - dx * 0.15,
      maxX: maxX + dx * 0.15,
      minY: minY - dy * 0.15,
      maxY: maxY + dy * 0.15,
    };
  }, [pcaResult]);

  const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX)) * (svgWidth - 2 * padding);
  const scaleY = (y: number) => svgHeight - padding - ((y - minY) / (maxY - minY)) * (svgHeight - 2 * padding);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Network className="w-5 h-5 text-cyan-400" /> Vocabulary & Positional Embedding Space
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            2D PCA projection of the 32-dimensional embedding representations ($d_{'{model}'}=32$).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
            <button
              onClick={() => setMode('tokens_only')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                mode === 'tokens_only' ? 'bg-cyan-500 text-zinc-950 font-bold' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Token Embeddings Only
            </button>
            <button
              onClick={() => setMode('token_plus_pos')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                mode === 'token_plus_pos' ? 'bg-cyan-500 text-zinc-950 font-bold' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Token + Position
            </button>
            <button
              onClick={() => setMode('all_combinations')}
              className={`px-3 py-1 rounded font-medium transition-all ${
                mode === 'all_combinations' ? 'bg-cyan-500 text-zinc-950 font-bold' : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              All 50 Combinations
            </button>
          </div>

          {mode === 'token_plus_pos' && (
            <div className="flex items-center gap-1.5 bg-zinc-950 px-2.5 py-1 rounded-lg border border-zinc-800 text-xs">
              <span className="text-zinc-500 font-semibold">Pos:</span>
              {[0, 1, 2, 3, 4].map(p => (
                <button
                  key={p}
                  onClick={() => setSelectedPos(p)}
                  className={`w-6 h-6 rounded text-xs font-bold transition-all ${
                    selectedPos === p
                      ? 'bg-cyan-500 text-zinc-950 shadow-[0_0_8px_rgba(6,182,212,0.4)]'
                      : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  P{p + 1}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-zinc-950 rounded-xl border border-zinc-800 p-4 relative flex flex-col items-center">
          <div className="w-full flex justify-between text-[11px] font-mono text-zinc-400 mb-2 px-2">
            <span>PC1 (Variance: {pcaResult.varExplained[0].toFixed(1)}%)</span>
            <span>PC2 (Variance: {pcaResult.varExplained[1].toFixed(1)}%)</span>
          </div>

          <svg width={svgWidth} height={svgHeight} className="w-full h-auto overflow-visible select-none">
            {/* Grid lines */}
            <line
              x1={scaleX(0)}
              y1={padding}
              x2={scaleX(0)}
              y2={svgHeight - padding}
              stroke="#27272a"
              strokeDasharray="4,4"
              strokeWidth="1.5"
            />
            <line
              x1={padding}
              y1={scaleY(0)}
              x2={svgWidth - padding}
              y2={scaleY(0)}
              stroke="#27272a"
              strokeDasharray="4,4"
              strokeWidth="1.5"
            />

            {/* Path connecting digits 0 -> 9 in sequence for tokens_only or token_plus_pos */}
            {mode !== 'all_combinations' && (
              <path
                d={pcaResult.points
                  .slice()
                  .sort((a, b) => a.digit - b.digit)
                  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.x)} ${scaleY(p.y)}`)
                  .join(' ')}
                fill="none"
                stroke="rgba(6, 182, 212, 0.25)"
                strokeWidth="2"
                strokeDasharray="3,3"
              />
            )}

            {/* Scatter points */}
            {pcaResult.points.map(p => {
              const cx = scaleX(p.x);
              const cy = scaleY(p.y);
              const isHovered = hoveredPoint?.id === p.id;
              const color = digitColors[p.digit];

              return (
                <g
                  key={p.id}
                  className="cursor-pointer transition-transform duration-200"
                  onMouseEnter={() => setHoveredPoint(p)}
                  onMouseLeave={() => setHoveredPoint(null)}
                >
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 12 : mode === 'all_combinations' ? 6 : 9}
                    fill={color}
                    fillOpacity={isHovered ? 0.9 : 0.75}
                    stroke={isHovered ? '#ffffff' : color}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    className="transition-all"
                  />
                  <text
                    x={cx}
                    y={cy + (mode === 'all_combinations' ? 3 : 4)}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize={mode === 'all_combinations' ? '8px' : '10px'}
                    fontWeight="bold"
                    pointerEvents="none"
                  >
                    {mode === 'all_combinations' ? p.digit : p.digit}
                  </text>
                </g>
              );
            })}
          </svg>

          <div className="mt-3 flex flex-wrap justify-center gap-3 text-[10px] font-mono text-zinc-400">
            {digitColors.map((col, d) => (
              <div key={d} className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: col }} />
                <span>Digit {d}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-zinc-950 p-5 rounded-xl border border-zinc-800 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-2 flex items-center gap-1.5">
              <Info className="w-4 h-4" /> Point Inspector
            </h3>

            {hoveredPoint ? (
              <div className="space-y-3 bg-zinc-900/60 p-4 rounded-xl border border-zinc-800">
                <div>
                  <span className="text-[10px] text-zinc-500 font-semibold uppercase">Token Label</span>
                  <div className="text-base font-extrabold text-zinc-100 flex items-center gap-2">
                    <span
                      className="w-3.5 h-3.5 rounded-full"
                      style={{ backgroundColor: digitColors[hoveredPoint.digit] }}
                    />
                    {hoveredPoint.label}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                  <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                    <span className="text-[9px] text-zinc-500 block">PC1 Coord</span>
                    <span className="text-cyan-400 font-bold">{hoveredPoint.x.toFixed(3)}</span>
                  </div>
                  <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                    <span className="text-[9px] text-zinc-500 block">PC2 Coord</span>
                    <span className="text-cyan-400 font-bold">{hoveredPoint.y.toFixed(3)}</span>
                  </div>
                  <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                    <span className="text-[9px] text-zinc-500 block">Vector L2 Norm</span>
                    <span className="text-violet-400 font-bold">{hoveredPoint.norm.toFixed(3)}</span>
                  </div>
                  <div className="bg-zinc-950 p-2 rounded border border-zinc-850">
                    <span className="text-[9px] text-zinc-500 block">Dimensions</span>
                    <span className="text-emerald-400 font-bold">32D</span>
                  </div>
                </div>

                <div>
                  <span className="text-[10px] text-zinc-500 font-semibold uppercase block mb-1">
                    Sample Vector Slices (First 6 dims)
                  </span>
                  <p className="text-[10px] font-mono text-zinc-400 bg-zinc-950 p-2 rounded border border-zinc-850 break-all">
                    [{hoveredPoint.rawVec.slice(0, 6).map(v => v.toFixed(3)).join(', ')}, ...]
                  </p>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-zinc-900/40 rounded-xl border border-zinc-850 text-xs text-zinc-400 leading-relaxed">
                Hover over any point in the 2D scatter plot to inspect its exact low-dimensional principal component coordinates, vector norms, and embedding dimensions.
              </div>
            )}
          </div>

          <div className="p-3 bg-zinc-900/40 border border-zinc-850 rounded-lg text-[11px] text-zinc-400 leading-relaxed space-y-1.5">
            <div className="font-semibold text-zinc-200 flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-cyan-400" /> Positional Encoding Effect
            </div>
            <p>
              When sinusoidal positional encodings are added ($E(token) + PE(pos)$), the token embeddings shift linearly in vector space. This allows self-attention heads to differentiate between identical digits at different sequence positions while maintaining numerical ordering geometries.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
