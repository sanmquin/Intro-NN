import { useState, useMemo } from 'react';
import { BarChart2 } from 'lucide-react';
import { ModelKey, getMagnitudeBilinearScores } from '../model/transformer';
import MatrixCellDerivationModal from './MatrixCellDerivationModal';

interface LearnedOrderingBiasCardProps {
  modelKey: ModelKey;
}

export default function LearnedOrderingBiasCard({ modelKey }: LearnedOrderingBiasCardProps) {
  const [inspectCell, setInspectCell] = useState<{ u: number; v: number } | null>(null);

  const magnitudeMatrix = useMemo(() => {
    return getMagnitudeBilinearScores(modelKey);
  }, [modelKey]);

  const magnitudeMinMax = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;
    magnitudeMatrix.forEach(row => {
      row.forEach(v => {
        if (v < min) min = v;
        if (v > max) max = v;
      });
    });
    return { min, max };
  }, [magnitudeMatrix]);

  const getBilinearCellColor = (val: number, min: number, max: number) => {
    const norm = (val - min) / (max - min || 1);
    const r = Math.round(16 + norm * 20);
    const g = Math.round(24 + norm * 160);
    const b = Math.round(32 + norm * 180);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm flex flex-col justify-between space-y-4">
      <div>
        <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2 mb-1.5">
          <BarChart2 className="w-4 h-4 text-cyan-400" /> Learned Ordering Bias Map
        </h3>
        <p className="text-xs text-zinc-400 leading-relaxed">
          Bilinear Query-Key magnitude weights matrix:
          <span className="font-mono text-cyan-300 block my-1">
            Bias(u, v) = e_u · W_q · W_k^T · e_v^T
          </span>
          Click on any cell (u, v) to inspect the ground-up dot product derivation between digit embeddings.
        </p>
      </div>

      <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-900 flex flex-col items-center">
        <div className="flex pl-6 mb-1 w-full justify-between text-[9px] font-mono text-zinc-500">
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
            <span key={i} className="w-4 text-center">{i}</span>
          ))}
        </div>

        <div className="flex w-full">
          <div className="flex flex-col pr-1.5 justify-between text-[9px] font-mono text-zinc-500 w-6 text-right">
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => (
              <span key={i} className="h-4 flex items-center justify-end">{i}</span>
            ))}
          </div>

          <div className="grid grid-cols-10 gap-0.5 flex-1 p-1 bg-zinc-900 border border-zinc-800 rounded">
            {magnitudeMatrix.map((row, uIdx) =>
              row.map((val, vIdx) => {
                const isInspected = inspectCell?.u === uIdx && inspectCell?.v === vIdx;
                return (
                  <div
                    key={`${uIdx}-${vIdx}`}
                    onClick={() => setInspectCell({ u: uIdx, v: vIdx })}
                    title={`Bilinear Score (${uIdx}, ${vIdx}): ${val.toFixed(2)} (Click to inspect)`}
                    style={{
                      backgroundColor: getBilinearCellColor(val, magnitudeMinMax.min, magnitudeMinMax.max),
                    }}
                    className={`w-full aspect-square rounded-[1px] hover:ring-1 hover:ring-white transition-all cursor-crosshair ${
                      isInspected ? 'ring-2 ring-emerald-400 scale-110 z-10' : ''
                    }`}
                  />
                );
              })
            )}
          </div>
        </div>
      </div>

      <div className="p-3 bg-zinc-950/60 border border-zinc-850/80 rounded-lg text-[10px] text-zinc-500 leading-relaxed">
        *Computed directly from Layer 1 weight parameters. Demonstrates how integers align along a continuous magnitude scale.
      </div>

      {inspectCell && (
        <MatrixCellDerivationModal
          type="learned_bias"
          modelKey={modelKey}
          cell={{ row: inspectCell.u, col: inspectCell.v }}
          onClose={() => setInspectCell(null)}
        />
      )}
    </div>
  );
}
