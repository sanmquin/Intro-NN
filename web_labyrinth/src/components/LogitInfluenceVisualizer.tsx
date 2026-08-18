import { useState } from 'react';
import { ArrowDownRight, Layers, Sparkles } from 'lucide-react';
import { TransformerActivationTrace, getRawWeights } from '../model/labyrinth_transformer';

interface LogitInfluenceVisualizerProps {
  trace: TransformerActivationTrace;
}

export default function LogitInfluenceVisualizer({ trace }: LogitInfluenceVisualizerProps) {
  const [selectedDirIdx, setSelectedDirIdx] = useState<number>(0);

  const weights = getRawWeights();
  const dirNames: ('Up' | 'Down' | 'Left' | 'Right')[] = ['Up', 'Down', 'Left', 'Right'];
  const selectedDirName = dirNames[selectedDirIdx];

  const { stepTrace, finalNorm } = trace;
  const { agentIndex, directions } = stepTrace;

  const finalVector = finalNorm[agentIndex]; // [32]
  const dirWeights = weights.fc_dir.weight[selectedDirIdx]; // [32]
  const dirBias = weights.fc_dir.bias[selectedDirIdx];

  const contributions = finalVector.map((act, i) => act * dirWeights[i]);
  const rawLogit = contributions.reduce((s, v) => s + v, 0) + dirBias;

  const f = (n: number) => n.toFixed(3);

  // Top positive and top negative feature channels
  const channels = contributions.map((val, dimIndex) => ({
    dimIndex,
    activation: finalVector[dimIndex],
    weight: dirWeights[dimIndex],
    contribution: val,
  }));

  const topPositive = [...channels].sort((a, b) => b.contribution - a.contribution).slice(0, 5);
  const topNegative = [...channels].sort((a, b) => a.contribution - b.contribution).slice(0, 5);

  const selectedDirOpt = directions.find(d => d.direction === selectedDirName);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-400" /> Directional Classifier Logit Influence
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Deconstruct how agent final hidden activations (h) project through FC weights (W_dir) to compute candidate move logits.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-xs text-zinc-500 font-semibold uppercase">Target Direction:</span>
            <div className="flex gap-1">
              {dirNames.map((dir, idx) => (
                <button
                  key={dir}
                  onClick={() => setSelectedDirIdx(idx)}
                  className={`px-2.5 py-1 rounded text-xs font-bold transition-all ${
                    selectedDirIdx === idx
                      ? 'bg-emerald-500 text-zinc-950 shadow-[0_0_10px_rgba(16,185,129,0.4)]'
                      : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {dir}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Target Direction Status Bar */}
      <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Direction Choice:
          </span>
          <span className="text-sm font-bold text-emerald-400 font-mono">
            Move {selectedDirName} {selectedDirOpt?.isOptimal ? '(Optimal Shortest Path)' : '(Suboptimal / Wall)'}
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-zinc-500 mr-2">Raw FC Logit:</span>
            <span className="text-emerald-400 font-bold">{f(rawLogit)}</span>
          </div>
          <div className="bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-zinc-500 mr-2">Softmax Prob:</span>
            <span className="text-cyan-400 font-bold">
              {((selectedDirOpt?.probability || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Driver Breakdown Split View */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Drivers */}
        <div className="bg-zinc-950 p-5 rounded-xl border border-zinc-800 space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-emerald-400" /> Top Positive Driving Feature Channels
          </h3>
          <p className="text-[11px] text-zinc-400 leading-relaxed">
            These 32D channels in final norm output (h) strongly align with move {selectedDirName}&apos;s classifier weights, boosting the logit.
          </p>

          <div className="space-y-2.5">
            {topPositive.map((item, idx) => (
              <div
                key={idx}
                className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-850 flex items-center justify-between text-xs font-mono"
              >
                <div className="flex items-center gap-3">
                  <span className="w-6 h-6 rounded bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 flex items-center justify-center font-bold text-[10px]">
                    D{item.dimIndex}
                  </span>
                  <div>
                    <div className="text-zinc-300">
                      Act: <span className="text-violet-300">{f(item.activation)}</span> &bull; Weight:{' '}
                      <span className="text-cyan-300">{f(item.weight)}</span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-emerald-400 font-extrabold text-sm">
                    +{f(item.contribution)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Suppressors */}
        <div className="bg-zinc-950 p-5 rounded-xl border border-zinc-800 space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
            <ArrowDownRight className="w-4 h-4 text-rose-400" /> Top Negative Suppressing Feature Channels
          </h3>
          <p className="text-[11px] text-zinc-400 leading-relaxed">
            These feature channels misalign with move {selectedDirName}&apos;s output weights, suppressing the logit score.
          </p>

          <div className="space-y-2.5">
            {topNegative.map((item, idx) => (
              <div
                key={idx}
                className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-850 flex items-center justify-between text-xs font-mono"
              >
                <div className="flex items-center gap-3">
                  <span className="w-6 h-6 rounded bg-rose-500/20 border border-rose-500/40 text-rose-400 flex items-center justify-center font-bold text-[10px]">
                    D{item.dimIndex}
                  </span>
                  <div>
                    <div className="text-zinc-300">
                      Act: <span className="text-violet-300">{f(item.activation)}</span> &bull; Weight:{' '}
                      <span className="text-cyan-300">{f(item.weight)}</span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-rose-400 font-extrabold text-sm">
                    {f(item.contribution)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Full 32 Dimension Contribution Spectrum */}
      <div className="bg-zinc-950 p-5 rounded-xl border border-zinc-800 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Full 32-Dimension Contribution Spectrum (h_i · W_dir[m, k])
          </h3>
          <span className="text-[10px] font-mono text-zinc-500">FC Bias b_dir: {f(dirBias)}</span>
        </div>

        <div className="grid grid-cols-4 sm:grid-cols-8 lg:grid-cols-16 gap-1.5 pt-2">
          {contributions.map((val, dimIdx) => {
            const isPos = val >= 0;
            const maxVal = Math.max(...contributions.map(Math.abs)) || 1;
            const heightPct = Math.min(100, (Math.abs(val) / maxVal) * 100);

            return (
              <div key={dimIdx} className="flex flex-col items-center gap-1 group cursor-crosshair">
                <div className="w-full h-20 bg-zinc-900 rounded border border-zinc-850 flex flex-col justify-end p-0.5 relative overflow-hidden">
                  <div
                    className={`w-full rounded-sm transition-all ${
                      isPos ? 'bg-emerald-500' : 'bg-rose-500'
                    }`}
                    style={{ height: `${heightPct}%` }}
                  />
                </div>
                <span className="text-[8px] font-mono text-zinc-500 group-hover:text-zinc-200">
                  d{dimIdx}
                </span>
                <span className="text-[8px] font-mono font-bold text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  {f(val)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
