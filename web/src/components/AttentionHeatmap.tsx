import { useState } from 'react';
import { TransformerActivationTrace } from '../model/labyrinth_transformer';

interface AttentionHeatmapProps {
  trace: TransformerActivationTrace;
  selectedLayer: number;
  selectedHead: number;
  onSelectHead: (layer: number, head: number) => void;
  hoveredTokenIdx: number | null;
  setHoveredTokenIdx: (idx: number | null) => void;
}

export default function AttentionHeatmap({
  trace,
  selectedLayer,
  selectedHead,
  onSelectHead,
  hoveredTokenIdx,
  setHoveredTokenIdx,
}: AttentionHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    layer: number;
    head: number;
    q: number;
    k: number;
    val: number;
  } | null>(null);

  const layers = [0, 1];
  const heads = [0, 1];
  const { nodesInfo } = trace;

  const getCellColor = (val: number, isSelected: boolean) => {
    const opacity = Math.min(Math.max(val, 0), 1);
    if (isSelected) {
      if (opacity < 0.05) return 'rgba(139, 92, 246, 0.05)';
      if (opacity < 0.15) return 'rgba(139, 92, 246, 0.25)';
      if (opacity < 0.35) return 'rgba(139, 92, 246, 0.55)';
      if (opacity < 0.65) return 'rgba(139, 92, 246, 0.8)';
      return 'rgba(139, 92, 246, 1.0)';
    } else {
      if (opacity < 0.05) return 'rgba(99, 102, 241, 0.05)';
      if (opacity < 0.15) return 'rgba(99, 102, 241, 0.2)';
      if (opacity < 0.35) return 'rgba(99, 102, 241, 0.4)';
      if (opacity < 0.65) return 'rgba(99, 102, 241, 0.7)';
      return 'rgba(99, 102, 241, 0.9)';
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <span className="text-violet-400">🔥</span> Multi-Head Spatial Attention Maps (36x36 Grid Matrix)
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Softmax attention matrix A = softmax(Q K^T / sqrt(d_k)) from 36 spatial grid cells to all 36 key locations.
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs text-zinc-400">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-zinc-900 border border-zinc-800 inline-block"></span>
            <span>0.0</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded inline-block bg-indigo-500/50"></span>
            <span>0.5</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded inline-block bg-violet-500 shadow-[0_0_8px_rgba(139,92,246,0.6)]"></span>
            <span>1.0</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {layers.map(l => (
          <div key={l} className="space-y-4">
            <h3 className="text-sm font-semibold text-zinc-300 border-b border-zinc-800 pb-2 flex items-center justify-between">
              <span>Layer {l + 1}</span>
              <span className="text-xs font-normal text-zinc-500">2 Attention Heads</span>
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {heads.map(h => {
                const isActive = selectedLayer === l && selectedHead === h;
                const attnWeights = trace.layers[l].heads[h].attnWeights;

                return (
                  <div
                    key={h}
                    onClick={() => onSelectHead(l, h)}
                    className={`cursor-pointer rounded-lg p-3 transition-all ${
                      isActive
                        ? 'bg-zinc-950 border border-violet-500/50 shadow-[0_0_15px_rgba(139,92,246,0.15)]'
                        : 'bg-zinc-950/40 border border-zinc-800/80 hover:border-zinc-700/80'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs font-medium text-zinc-300">
                        Head {h === 0 ? '0 (Topological)' : '1 (Spatial)'}
                      </span>
                      {isActive && <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse"></span>}
                    </div>

                    <div className="relative">
                      {/* 36x36 Attention Heatmap Grid */}
                      <div className="grid grid-cols-36 gap-[0.5px] aspect-square bg-zinc-900 p-1 rounded border border-zinc-800">
                        {attnWeights.map((row, qIdx) =>
                          row.map((val, kIdx) => {
                            const cellHovered =
                              hoveredCell?.layer === l &&
                              hoveredCell?.head === h &&
                              hoveredCell?.q === qIdx &&
                              hoveredCell?.k === kIdx;

                            const isTokenHighlighted = hoveredTokenIdx === qIdx || hoveredTokenIdx === kIdx;

                            return (
                              <div
                                key={`${qIdx}-${kIdx}`}
                                onMouseEnter={() => {
                                  setHoveredCell({ layer: l, head: h, q: qIdx, k: kIdx, val });
                                  setHoveredTokenIdx(qIdx);
                                }}
                                onMouseLeave={() => {
                                  setHoveredCell(null);
                                  setHoveredTokenIdx(null);
                                }}
                                style={{
                                  backgroundColor: getCellColor(val, isActive),
                                }}
                                className={`w-full aspect-square rounded-[0.2px] transition-all duration-150 ${
                                  cellHovered
                                    ? 'ring-1 ring-white scale-125 z-10 shadow-[0_0_8px_rgba(255,255,255,0.4)]'
                                    : isTokenHighlighted && !hoveredCell
                                    ? 'ring-[0.5px] ring-zinc-500'
                                    : ''
                                }`}
                              />
                            );
                          })
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-zinc-950/80 border border-zinc-800 rounded-lg min-h-[64px] flex items-center justify-between transition-all">
        {hoveredCell ? (
          <div className="flex items-center justify-between w-full">
            <div>
              <p className="text-sm font-medium text-zinc-200">
                Layer {hoveredCell.layer + 1}, Head {hoveredCell.head + 1} Attention Activation:
              </p>
              <p className="text-xs text-zinc-400 mt-0.5">
                Query Cell <span className="text-violet-400 font-bold">#{hoveredCell.q}</span> ({nodesInfo[hoveredCell.q]?.type}) paying attention to Key Cell <span className="text-violet-400 font-bold">#{hoveredCell.k}</span> ({nodesInfo[hoveredCell.k]?.type})
              </p>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-violet-400">{(hoveredCell.val * 100).toFixed(1)}%</div>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Softmax Score</p>
            </div>
          </div>
        ) : (
          <div className="text-zinc-500 text-xs italic flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-zinc-700"></span>
            Hover over any cell in the 36x36 matrix to inspect query-key attention scores.
          </div>
        )}
      </div>
    </div>
  );
}
