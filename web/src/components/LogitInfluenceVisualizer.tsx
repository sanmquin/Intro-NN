import { useState, useMemo } from 'react';
import { ArrowDownRight, Layers, Sparkles } from 'lucide-react';
import { TransformerActivationTrace, getLogitAttribution } from '../model/transformer';

interface LogitInfluenceVisualizerProps {
  trace: TransformerActivationTrace;
}

export default function LogitInfluenceVisualizer({ trace }: LogitInfluenceVisualizerProps) {
  const [selectedPosIdx, setSelectedPosIdx] = useState<number>(0);
  const [selectedDigit, setSelectedDigit] = useState<number>(trace.predictions[0]);

  const tokens = trace.inputTokens;

  const logitAttribution = useMemo(() => {
    return getLogitAttribution(trace, selectedPosIdx, selectedDigit);
  }, [trace, selectedPosIdx, selectedDigit]);

  const winnerDigit = trace.predictions[selectedPosIdx];

  const f = (num: number) => num.toFixed(3);

  // Softmax calculation across all 10 logits for selected position
  const softmaxProbs = useMemo(() => {
    const rawLogits = trace.logits[selectedPosIdx];
    const maxL = Math.max(...rawLogits);
    const exps = rawLogits.map(l => Math.exp(l - maxL));
    const sumE = exps.reduce((s, e) => s + e, 0);
    return exps.map(e => e / sumE);
  }, [trace, selectedPosIdx]);

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-400" /> Last Layer to Output FC Influence
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Analyze how final 32D Transformer representations ($h_i$) project through classifier weights ($W_{'{out}'}$) to compute logits.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-xs text-zinc-500 font-semibold uppercase">Position:</span>
            <div className="flex gap-1">
              {tokens.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setSelectedPosIdx(idx);
                    setSelectedDigit(trace.predictions[idx]);
                  }}
                  className={`w-7 h-7 rounded text-xs font-bold transition-all flex flex-col items-center justify-center ${
                    selectedPosIdx === idx
                      ? 'bg-emerald-500 text-zinc-950 shadow-[0_0_10px_rgba(16,185,129,0.4)]'
                      : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <span>P{idx + 1}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Target Digit Selector Bar */}
      <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Target Output Digit:
          </span>
          <div className="flex gap-1">
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(digit => {
              const isSelected = selectedDigit === digit;
              const isWinner = winnerDigit === digit;
              return (
                <button
                  key={digit}
                  onClick={() => setSelectedDigit(digit)}
                  className={`w-8 h-8 rounded-lg font-mono font-extrabold text-xs transition-all relative ${
                    isSelected
                      ? 'bg-emerald-500 text-zinc-950 shadow-[0_0_10px_rgba(16,185,129,0.4)] scale-105'
                      : isWinner
                      ? 'bg-emerald-950/60 border border-emerald-500/50 text-emerald-300'
                      : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {digit}
                  {isWinner && !isSelected && (
                    <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-emerald-400" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-zinc-500 mr-2">Raw Logit:</span>
            <span className="text-emerald-400 font-bold">{f(logitAttribution.totalLogit)}</span>
          </div>
          <div className="bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800">
            <span className="text-zinc-500 mr-2">Softmax Prob:</span>
            <span className="text-cyan-400 font-bold">
              {(softmaxProbs[selectedDigit] * 100).toFixed(1)}%
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
            These feature dimensions in Layer 2 output ($h$) strongly align with digit {selectedDigit}&apos;s output weights ($W_{'{out}'}$), driving up the logit prediction.
          </p>

          <div className="space-y-2.5">
            {logitAttribution.topPositive.map((item, idx) => (
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
            These feature dimensions misalign with digit {selectedDigit}&apos;s classifier weights, suppressing the logit score.
          </p>

          <div className="space-y-2.5">
            {logitAttribution.topNegative.map((item, idx) => (
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

      {/* Full 32 Dimension Contribution Bar Grid */}
      <div className="bg-zinc-950 p-5 rounded-xl border border-zinc-800 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Full 32-Dimension Contribution Spectrum (h_i · W_out[v, k])
          </h3>
          <span className="text-[10px] font-mono text-zinc-500">Bias b_out: {f(logitAttribution.fcBias)}</span>
        </div>

        <div className="grid grid-cols-4 sm:grid-cols-8 lg:grid-cols-16 gap-1.5 pt-2">
          {logitAttribution.contributions.map((val, dimIdx) => {
            const isPos = val >= 0;
            const maxVal = Math.max(...logitAttribution.contributions.map(Math.abs)) || 1;
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
