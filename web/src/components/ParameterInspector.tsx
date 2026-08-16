import { useState, useMemo } from 'react';
import { Cpu, Search, Database, Layers } from 'lucide-react';
import { getNetworkParameterDetails } from '../model/transformer';

export default function ParameterInspector() {
  const { tensors, totalParams, groupBreakdown } = useMemo(() => getNetworkParameterDetails(), []);

  const [selectedTensorId, setSelectedTensorId] = useState<string>('embedding');
  const [selectedGroup, setSelectedGroup] = useState<string>('All');
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; val: number } | null>(null);

  const selectedTensor = useMemo(
    () => tensors.find(t => t.id === selectedTensorId) || tensors[0],
    [tensors, selectedTensorId]
  );

  const groups = useMemo(() => ['All', ...new Set(tensors.map(t => t.group))], [tensors]);

  const filteredTensors = useMemo(() => {
    if (selectedGroup === 'All') return tensors;
    return tensors.filter(t => t.group === selectedGroup);
  }, [tensors, selectedGroup]);

  const getHeatmapColor = (val: number, min: number, max: number) => {
    const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const norm = Math.min(1, Math.max(-1, val / absMax));

    if (norm >= 0) {
      // Positive: Purple / Indigo
      const r = Math.round(24 + norm * 100);
      const g = Math.round(24 + norm * 80);
      const b = Math.round(36 + norm * 200);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Negative: Rose / Orange
      const absNorm = Math.abs(norm);
      const r = Math.round(36 + absNorm * 200);
      const g = Math.round(24 + absNorm * 60);
      const b = Math.round(24 + absNorm * 60);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" /> Full Network Parameter Inspector
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Deep-dive into every parameter tensor, layer shape, and weight matrix across the model.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800 flex items-center gap-2">
            <Database className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="text-[9px] text-zinc-500 uppercase block font-semibold">Total Parameters</span>
              <span className="text-sm font-extrabold text-emerald-400 font-mono">
                {totalParams.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Layer Groups Overview Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {groupBreakdown.map(gb => {
          const isSelected = selectedGroup === gb.group;
          return (
            <button
              key={gb.group}
              onClick={() => setSelectedGroup(gb.group)}
              className={`p-3 rounded-xl border transition-all text-left flex flex-col justify-between ${
                isSelected
                  ? 'bg-indigo-950/30 border-indigo-500 ring-1 ring-indigo-500/30 text-indigo-300'
                  : 'bg-zinc-950 border-zinc-850 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">{gb.group}</span>
                <Layers className="w-3.5 h-3.5 text-zinc-600" />
              </div>
              <div className="mt-2 text-sm font-extrabold font-mono text-zinc-100">
                {gb.count.toLocaleString()} <span className="text-[10px] text-zinc-500 font-normal">params</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Tensor Selector and Deep Dive Split View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left List of Tensors */}
        <div className="bg-zinc-950 rounded-xl border border-zinc-800 p-4 space-y-3 flex flex-col h-[480px]">
          <div className="flex items-center justify-between text-xs text-zinc-400 pb-2 border-b border-zinc-850">
            <span className="font-bold flex items-center gap-1">
              <Search className="w-3.5 h-3.5 text-indigo-400" /> Tensors ({filteredTensors.length})
            </span>
            <div className="flex gap-1">
              {groups.map(g => (
                <button
                  key={g}
                  onClick={() => setSelectedGroup(g)}
                  className={`text-[9px] px-2 py-0.5 rounded font-mono ${
                    selectedGroup === g ? 'bg-indigo-500 text-white font-bold' : 'bg-zinc-900 text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
            {filteredTensors.map(t => {
              const isSelected = selectedTensorId === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => {
                    setSelectedTensorId(t.id);
                    setHoveredCell(null);
                  }}
                  className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex items-center justify-between ${
                    isSelected
                      ? 'bg-indigo-950/40 border-indigo-500 text-indigo-200 shadow-[0_0_10px_rgba(99,102,241,0.2)]'
                      : 'bg-zinc-900/50 border-zinc-850 hover:border-zinc-700 text-zinc-300'
                  }`}
                >
                  <div className="flex flex-col truncate pr-2">
                    <span className="font-semibold truncate">{t.name}</span>
                    <span className="text-[9px] text-zinc-500 font-mono">{t.group}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-950 border border-zinc-800 text-cyan-400">
                      [{t.shape.join('×')}]
                    </span>
                    <span className="text-[10px] font-mono font-bold text-emerald-400">
                      {t.paramCount}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Deep Dive Inspector */}
        <div className="lg:col-span-2 bg-zinc-950 rounded-xl border border-zinc-800 p-5 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-2">
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 block">
                  Active Tensor Deep-Dive
                </span>
                <h3 className="text-base font-extrabold text-zinc-100 mt-0.5">
                  {selectedTensor.name}
                </h3>
              </div>

              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 text-indigo-300">
                  Shape: [{selectedTensor.shape.join(' × ')}]
                </span>
                <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-bold">
                  {selectedTensor.paramCount} parameters
                </span>
              </div>
            </div>

            {/* Summary Statistics */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-4 text-xs font-mono">
              <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                <span className="text-[9px] text-zinc-500 uppercase block font-semibold">Min Value</span>
                <span className="text-rose-400 font-bold">{selectedTensor.min.toFixed(4)}</span>
              </div>
              <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                <span className="text-[9px] text-zinc-500 uppercase block font-semibold">Max Value</span>
                <span className="text-indigo-400 font-bold">{selectedTensor.max.toFixed(4)}</span>
              </div>
              <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                <span className="text-[9px] text-zinc-500 uppercase block font-semibold">Mean</span>
                <span className="text-cyan-400 font-bold">{selectedTensor.mean.toFixed(4)}</span>
              </div>
              <div className="bg-zinc-900/60 p-2.5 rounded-lg border border-zinc-800">
                <span className="text-[9px] text-zinc-500 uppercase block font-semibold">Std Dev</span>
                <span className="text-emerald-400 font-bold">{selectedTensor.std.toFixed(4)}</span>
              </div>
            </div>

            {/* Matrix / Vector Visualization */}
            <div className="mt-4 bg-zinc-900/80 p-4 rounded-xl border border-zinc-850 flex flex-col items-center min-h-[220px] justify-center">
              {Array.isArray(selectedTensor.data[0]) ? (
                // 2D Matrix Heatmap
                <div className="w-full space-y-2">
                  <div className="flex justify-between items-center text-[10px] font-mono text-zinc-400">
                    <span>
                      Matrix View ({selectedTensor.shape[0]} rows × {selectedTensor.shape[1]} cols)
                    </span>
                    {hoveredCell && (
                      <span className="text-cyan-400 font-bold bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                        Row {hoveredCell.r}, Col {hoveredCell.c}: {hoveredCell.val.toFixed(4)}
                      </span>
                    )}
                  </div>

                  <div className="max-h-[240px] overflow-auto border border-zinc-800 rounded p-1 bg-zinc-950 custom-scrollbar">
                    <div
                      className="grid gap-[1px]"
                      style={{
                        gridTemplateColumns: `repeat(${selectedTensor.shape[1]}, minmax(8px, 1fr))`,
                      }}
                    >
                      {(selectedTensor.data as number[][]).map((row, rIdx) =>
                        row.map((val, cIdx) => (
                          <div
                            key={`${rIdx}-${cIdx}`}
                            onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, val })}
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{
                              backgroundColor: getHeatmapColor(val, selectedTensor.min, selectedTensor.max),
                            }}
                            className="aspect-square rounded-[1px] hover:ring-1 hover:ring-white transition-all cursor-crosshair min-w-[6px]"
                            title={`Row ${rIdx}, Col ${cIdx}: ${val.toFixed(4)}`}
                          />
                        ))
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                // 1D Vector Bar Plot
                <div className="w-full space-y-3">
                  <div className="flex justify-between items-center text-[10px] font-mono text-zinc-400">
                    <span>Vector View ({selectedTensor.shape[0]} elements)</span>
                    {hoveredCell && (
                      <span className="text-cyan-400 font-bold bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                        Index {hoveredCell.c}: {hoveredCell.val.toFixed(4)}
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-8 sm:grid-cols-16 gap-1 bg-zinc-950 p-2 rounded border border-zinc-800">
                    {(selectedTensor.data as number[]).map((val, idx) => (
                      <div
                        key={idx}
                        onMouseEnter={() => setHoveredCell({ r: 0, c: idx, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className="flex flex-col items-center gap-1 cursor-pointer group"
                      >
                        <div className="w-full h-16 bg-zinc-900 rounded border border-zinc-850 flex items-end justify-center p-0.5 relative overflow-hidden">
                          <div
                            className="w-full rounded-sm transition-all"
                            style={{
                              height: `${Math.min(100, (Math.abs(val) / (Math.abs(selectedTensor.max) || 1)) * 100)}%`,
                              backgroundColor: getHeatmapColor(val, selectedTensor.min, selectedTensor.max),
                            }}
                          />
                        </div>
                        <span className="text-[8px] font-mono text-zinc-500 group-hover:text-zinc-200">
                          {idx}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 pt-2 border-t border-zinc-900">
            <span>Color Scale: Rose (- negative) &bull; Dark (0) &bull; Indigo/Purple (+ positive)</span>
            <span>Weights loaded from model_weights.json</span>
          </div>
        </div>
      </div>
    </div>
  );
}
