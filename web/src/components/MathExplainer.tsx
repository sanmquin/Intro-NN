import { useState } from 'react';
import { TransformerActivationTrace } from '../model/transformer';

interface MathExplainerProps {
  trace: TransformerActivationTrace;
  selectedLayer: number;
  selectedHead: number;
  activeMathBlock: string;
}

export default function MathExplainer({
  trace,
  selectedLayer,
  selectedHead,
  activeMathBlock,
}: MathExplainerProps) {
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(0);

  const tokens = trace.inputTokens;
  const activeToken = tokens[selectedTokenIdx];

  const f = (num: number) => num.toFixed(3);

  const formatVector = (vec: number[], limit: number = 8) => {
    const sliced = vec.slice(0, limit).map(v => f(v));
    const trailing = vec.length > limit ? ', ...' : '';
    return `[${sliced.join(', ')}${trailing}] (size ${vec.length})`;
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-zinc-800 pb-4 mb-6 gap-4">
        <div>
          <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2">
            <span className="text-indigo-400">📝</span> Mathematical Execution Trace
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Real-time numerical outputs of active activations from the forward pass.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-800">
          <span className="text-xs text-zinc-500 font-semibold uppercase">Token Position:</span>
          <div className="flex gap-1">
            {tokens.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedTokenIdx(idx)}
                className={`w-6 h-6 rounded text-xs font-bold transition-all ${
                  selectedTokenIdx === idx
                    ? 'bg-indigo-500 text-white shadow-[0_0_8px_rgba(99,102,241,0.5)]'
                    : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>

      {activeMathBlock === 'embed' && (
        <div className="space-y-6">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800">
            <h3 className="text-sm font-bold text-cyan-400 mb-2">1. Embedding & Positional Encoding</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Before processing, each input integer token $t$ is projected to a high-dimensional vector. Since transformers have no recurrence, sinusoidal positional encodings are added to inform the model of sequence coordinates.
            </p>
            <div className="mt-4 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800/80 font-mono text-[11px] text-zinc-300 space-y-1.5">
              <div>Formula: <span className="text-violet-300 font-bold">X_sum[i] = Embedding(t) + PositionalEncoding[i]</span></div>
              <div>Dimension: <span className="text-cyan-400 font-semibold">d_model = 32</span></div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Token Embedding</span>
              <div className="text-lg font-bold text-zinc-100 mt-1 mb-2">Token {activeToken}</div>
              <p className="text-[11px] font-mono text-cyan-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.embeddings[selectedTokenIdx])}
              </p>
            </div>

            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Pos Encoding (Pos #{selectedTokenIdx + 1})</span>
              <div className="text-lg font-bold text-zinc-100 mt-1 mb-2">Sine-Cosine</div>
              <p className="text-[11px] font-mono text-cyan-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.posEncodings[selectedTokenIdx])}
              </p>
            </div>

            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Summed Input Vector</span>
              <div className="text-lg font-bold text-zinc-100 mt-1 mb-2">Initial Sum</div>
              <p className="text-[11px] font-mono text-violet-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.initialSum[selectedTokenIdx])}
              </p>
            </div>
          </div>
        </div>
      )}

      {(activeMathBlock === 'layer1' || activeMathBlock === 'layer2') && (
        <div className="space-y-6">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800">
            <h3 className="text-sm font-bold text-violet-400 mb-2">
              Transformer Layer {selectedLayer + 1} - Multi-Head Attention (Head {selectedHead + 1})
            </h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              The normalized input vector is projected to Query ($Q$), Key ($K$), and Value ($V$) subspaces using learned projection matrices:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 mt-4 gap-3 font-mono text-[11px] text-zinc-300">
              <div className="p-3 bg-zinc-900/50 rounded-lg border border-zinc-800/80 space-y-1">
                <div className="font-bold text-violet-300">Dot-Product Scores Formula:</div>
                <div>AttentionScores[i, j] = (Q[i] · K[j]) / sqrt(d_k)</div>
                <div>AttentionWeights = Softmax(AttentionScores, dim=-1)</div>
              </div>
              <div className="p-3 bg-zinc-900/50 rounded-lg border border-zinc-800/80 space-y-1">
                <div className="font-bold text-violet-300">Feed-Forward (FFN) Formula:</div>
                <div>FFN_mid = GELU(LayerNorm(x) · W1 + b1)</div>
                <div>FFN_out = FFN_mid · W2 + b2</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Query Vector (q)</span>
              <div className="text-xs font-semibold text-zinc-300 mt-1 mb-2">Active Token: {activeToken}</div>
              <p className="text-[11px] font-mono text-zinc-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.layers[selectedLayer].heads[selectedHead].queries[selectedTokenIdx])}
              </p>
            </div>

            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Key Vector (k)</span>
              <div className="text-xs font-semibold text-zinc-300 mt-1 mb-2">Active Token: {activeToken}</div>
              <p className="text-[11px] font-mono text-zinc-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.layers[selectedLayer].heads[selectedHead].keys[selectedTokenIdx])}
              </p>
            </div>

            <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
              <span className="text-[10px] uppercase font-bold tracking-wider text-zinc-500">Value Vector (v)</span>
              <div className="text-xs font-semibold text-zinc-300 mt-1 mb-2">Active Token: {activeToken}</div>
              <p className="text-[11px] font-mono text-zinc-400 break-all leading-relaxed bg-zinc-950 p-2.5 rounded border border-zinc-800/60">
                {formatVector(trace.layers[selectedLayer].heads[selectedHead].values[selectedTokenIdx])}
              </p>
            </div>
          </div>

          <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
            <h4 className="text-xs font-bold text-zinc-300 mb-3 uppercase tracking-wider">
              Attention Coefficient vector for Position #{selectedTokenIdx + 1} (Token {activeToken})
            </h4>
            <div className="space-y-2">
              {trace.layers[selectedLayer].heads[selectedHead].attnWeights[selectedTokenIdx].map((weight, targetIdx) => (
                <div key={targetIdx} className="flex items-center justify-between text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                    <span className="text-zinc-400">Attends to position #{targetIdx + 1} (Token {tokens[targetIdx]}):</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-zinc-950 rounded-full h-1.5 border border-zinc-850 overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full rounded-full"
                        style={{ width: `${weight * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-indigo-400 font-bold w-12 text-right">
                      {(weight * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeMathBlock === 'output' && (
        <div className="space-y-6">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800">
            <h3 className="text-sm font-bold text-emerald-400 mb-2">3. Classifier Head & Final Output Projection</h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              The final normalized representations are mapped back to log-probabilities (logits) over the vocabulary space (digits 0 to 9) using a dense output projection layer.
            </p>
            <div className="mt-4 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800/80 font-mono text-[11px] text-zinc-300 space-y-1.5">
              <div>Formula: <span className="text-emerald-300 font-bold">Logits = LayerNorm(X_final) · W_out + b_out</span></div>
              <div>Output Prediction: <span className="text-emerald-300 font-bold">Token = argmax(Logits)</span></div>
            </div>
          </div>

          <div className="bg-zinc-950/40 p-4 rounded-xl border border-zinc-800/80">
            <h4 className="text-xs font-bold text-zinc-300 mb-4 uppercase tracking-wider">
              Prediction Logits for Position #{selectedTokenIdx + 1}
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {trace.logits[selectedTokenIdx].map((logit, val) => {
                const isWinner = trace.predictions[selectedTokenIdx] === val;
                return (
                  <div
                    key={val}
                    className={`p-3 rounded-lg border font-mono flex flex-col justify-between items-center transition-all ${
                      isWinner
                        ? 'bg-emerald-950/30 border-emerald-500 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.15)] scale-105'
                        : 'bg-zinc-950 border-zinc-850 text-zinc-400'
                    }`}
                  >
                    <span className="text-[10px] text-zinc-500 uppercase font-semibold">Digit</span>
                    <span className="text-lg font-bold mt-1 mb-1">{val}</span>
                    <span className={`text-xs ${isWinner ? 'font-bold text-emerald-400' : 'text-zinc-500'}`}>
                      {f(logit)}
                    </span>
                    {isWinner && (
                      <span className="text-[8px] uppercase font-bold tracking-wider text-emerald-500 mt-1">Winner</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
