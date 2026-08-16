import { useState } from 'react';
import { ArrowRight, Calculator, CheckCircle2, Hash, Layers, Sparkles } from 'lucide-react';
import {
  TransformerActivationTrace,
  ModelKey,
  getAttentionCellDerivation,
  getLearnedBiasCellDerivation,
  AttentionCellDerivation,
  LearnedBiasCellDerivation,
} from '../model/transformer';

interface MatrixCellDerivationModalProps {
  type: 'attention' | 'learned_bias';
  trace?: TransformerActivationTrace;
  modelKey: ModelKey;
  layerIdx?: number;
  headIdx?: number;
  // Cell selection coordinates
  cell: {
    row: number; // qPos or uDigit
    col: number; // kPos or vDigit
  } | null;
  onClose: () => void;
}

export default function MatrixCellDerivationModal({
  type,
  trace,
  modelKey,
  layerIdx = 0,
  headIdx = 0,
  cell,
  onClose,
}: MatrixCellDerivationModalProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'dot_product' | 'softmax' | 'value_context'>('overview');

  if (!cell) return null;

  const f = (n: number) => n.toFixed(3);

  const isAttention = type === 'attention' && trace;
  const attnDerivation: AttentionCellDerivation | null = isAttention
    ? getAttentionCellDerivation(trace, layerIdx, headIdx, cell.row, cell.col)
    : null;

  const biasDerivation: LearnedBiasCellDerivation | null = !isAttention
    ? getLearnedBiasCellDerivation(modelKey, cell.row, cell.col)
    : null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 shadow-2xl space-y-5 backdrop-blur-md animate-in fade-in zoom-in-95 duration-150">
      <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
            <Calculator className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2">
              Ground-Up Cell Derivation Inspector
              <span className="px-2 py-0.5 rounded-full text-[10px] bg-violet-500/20 text-violet-300 font-mono border border-violet-500/30">
                Cell [{cell.row}, {cell.col}]
              </span>
            </h3>
            <p className="text-[11px] text-zinc-400">
              {isAttention
                ? `Dynamic Forward Pass: Pos #${cell.row + 1} (Token ${trace?.inputTokens[cell.row]}) → Pos #${cell.col + 1} (Token ${trace?.inputTokens[cell.col]})`
                : `Static Learned Bias: Query Digit ${cell.row} → Key Digit ${cell.col}`}
            </p>
          </div>
        </div>

        <button
          onClick={onClose}
          className="text-xs text-zinc-500 hover:text-zinc-200 px-2.5 py-1 rounded bg-zinc-950 border border-zinc-800 hover:border-zinc-700 transition-all"
        >
          Close
        </button>
      </div>

      {/* Pipeline Navigation Tabs */}
      <div className="flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs gap-1 overflow-x-auto">
        <button
          onClick={() => setActiveTab('overview')}
          className={`flex-1 min-w-[110px] py-1.5 px-3 rounded font-medium transition-all flex items-center justify-center gap-1.5 ${
            activeTab === 'overview'
              ? 'bg-indigo-500 text-white font-bold shadow-[0_0_10px_rgba(99,102,241,0.3)]'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" /> 1. Overview
        </button>
        <button
          onClick={() => setActiveTab('dot_product')}
          className={`flex-1 min-w-[110px] py-1.5 px-3 rounded font-medium transition-all flex items-center justify-center gap-1.5 ${
            activeTab === 'dot_product'
              ? 'bg-indigo-500 text-white font-bold shadow-[0_0_10px_rgba(99,102,241,0.3)]'
              : 'text-zinc-400 hover:text-zinc-200'
          }`}
        >
          <Hash className="w-3.5 h-3.5" /> 2. Q·K Dot Product
        </button>
        {isAttention && (
          <>
            <button
              onClick={() => setActiveTab('softmax')}
              className={`flex-1 min-w-[110px] py-1.5 px-3 rounded font-medium transition-all flex items-center justify-center gap-1.5 ${
                activeTab === 'softmax'
                  ? 'bg-indigo-500 text-white font-bold shadow-[0_0_10px_rgba(99,102,241,0.3)]'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" /> 3. Softmax Weight
            </button>
            <button
              onClick={() => setActiveTab('value_context')}
              className={`flex-1 min-w-[110px] py-1.5 px-3 rounded font-medium transition-all flex items-center justify-center gap-1.5 ${
                activeTab === 'value_context'
                  ? 'bg-indigo-500 text-white font-bold shadow-[0_0_10px_rgba(99,102,241,0.3)]'
                  : 'text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <CheckCircle2 className="w-3.5 h-3.5" /> 4. Value Context
            </button>
          </>
        )}
      </div>

      {/* Tab Content 1: Overview */}
      {activeTab === 'overview' && (
        <div className="space-y-4">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 space-y-2">
            <h4 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
              Mathematical Equation Flow
            </h4>
            <p className="text-xs text-zinc-300 leading-relaxed font-mono bg-zinc-900/80 p-3 rounded border border-zinc-800">
              {isAttention
                ? `Score(i, j) = (q_i · k_j) / √d_k  →  Attn(i, j) = Softmax_j(Score(i, j))`
                : `Bias(u, v) = e_u · W_q · W_k^T · e_v^T = q(u) · k(v)`}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-850 space-y-1">
              <span className="text-[10px] text-zinc-500 uppercase font-semibold">Step A: Dot Product Sum</span>
              <div className="text-base font-extrabold text-zinc-100 font-mono">
                {attnDerivation ? f(attnDerivation.dotProductSum) : biasDerivation ? f(biasDerivation.bilinearScore) : 0}
              </div>
              <p className="text-[10px] text-zinc-400">Sum of elementwise q_c * k_c over vector dimension.</p>
            </div>

            {isAttention && attnDerivation && (
              <>
                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-850 space-y-1">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Step B: Scaled Score S_ij</span>
                  <div className="text-base font-extrabold text-cyan-400 font-mono">
                    {f(attnDerivation.scaledScore)}
                  </div>
                  <p className="text-[10px] text-zinc-400">Scaled by 1 / √{attnDerivation.d_k} = 1 / {f(attnDerivation.sqrt_d_k)}</p>
                </div>

                <div className="bg-zinc-950 p-3.5 rounded-xl border border-zinc-850 space-y-1">
                  <span className="text-[10px] text-zinc-500 uppercase font-semibold">Step C: Final Softmax Weight</span>
                  <div className="text-base font-extrabold text-violet-400 font-mono">
                    {(attnDerivation.softmaxWeight * 100).toFixed(1)}%
                  </div>
                  <p className="text-[10px] text-zinc-400">Fractional routing weight assigned to this key.</p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Tab Content 2: Elementwise Dot Product */}
      {activeTab === 'dot_product' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-zinc-300">
              Elementwise Vector Multiplication (q_c × k_c)
            </span>
            <span className="text-indigo-400 font-mono font-bold">
              Total Sum = {attnDerivation ? f(attnDerivation.dotProductSum) : biasDerivation ? f(biasDerivation.bilinearScore) : 0}
            </span>
          </div>

          <div className="max-h-[220px] overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
            {(attnDerivation?.elementwiseProds || biasDerivation?.elementwiseProds || []).map((item) => (
              <div
                key={item.dimIndex}
                className="bg-zinc-950 p-2 rounded-lg border border-zinc-850 flex items-center justify-between text-xs font-mono"
              >
                <div className="flex items-center gap-2">
                  <span className="w-6 h-5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400 text-[10px] flex items-center justify-center font-bold">
                    c{item.dimIndex}
                  </span>
                  <span className="text-cyan-400">{f(item.qVal)}</span>
                  <span className="text-zinc-600">×</span>
                  <span className="text-violet-400">{f(item.kVal)}</span>
                </div>
                <div className="flex items-center gap-2">
                  <ArrowRight className="w-3 h-3 text-zinc-600" />
                  <span className={`font-bold ${item.prod >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {item.prod >= 0 ? '+' : ''}{f(item.prod)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab Content 3: Softmax Derivation */}
      {activeTab === 'softmax' && attnDerivation && (
        <div className="space-y-4 text-xs font-mono">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 space-y-3">
            <h4 className="font-bold text-cyan-400 uppercase tracking-wider text-[10px]">
              Softmax Row Exponent Normalization
            </h4>

            <div className="space-y-2 text-zinc-300">
              <div className="flex justify-between p-2 bg-zinc-900 rounded border border-zinc-850">
                <span className="text-zinc-400">Raw Score S_ij:</span>
                <span className="text-cyan-400 font-bold">{f(attnDerivation.scaledScore)}</span>
              </div>
              <div className="flex justify-between p-2 bg-zinc-900 rounded border border-zinc-850">
                <span className="text-zinc-400">Exp Score e^(S_ij - S_max):</span>
                <span className="text-violet-400 font-bold">{f(attnDerivation.expValue)}</span>
              </div>
              <div className="flex justify-between p-2 bg-zinc-900 rounded border border-zinc-850">
                <span className="text-zinc-400">Sum of Exponents across Row:</span>
                <span className="text-amber-400 font-bold">{f(attnDerivation.rowExpSum)}</span>
              </div>
              <div className="flex justify-between p-2.5 bg-violet-950/30 rounded border border-violet-500/40 text-violet-200">
                <span className="font-bold">Softmax Weight A_ij:</span>
                <span className="font-extrabold text-violet-300 text-sm">
                  {(attnDerivation.softmaxWeight * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Content 4: Value Context Aggregation */}
      {activeTab === 'value_context' && attnDerivation && (
        <div className="space-y-4 text-xs font-mono">
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 space-y-2">
            <h4 className="font-bold text-emerald-400 uppercase tracking-wider text-[10px]">
              Context Vector Contribution (A_ij × v_j)
            </h4>
            <p className="text-[11px] text-zinc-400 font-sans leading-relaxed">
              The softmax weight <span className="text-violet-400 font-bold">{(attnDerivation.softmaxWeight * 100).toFixed(1)}%</span> scales Key position #{attnDerivation.keyPos + 1}&apos;s Value vector before summing into position #{attnDerivation.queryPos + 1}&apos;s context representation.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 max-h-[180px] overflow-y-auto custom-scrollbar">
            {attnDerivation.weightedValueVector.map((val, idx) => (
              <div key={idx} className="bg-zinc-950 p-2 rounded border border-zinc-850 flex justify-between">
                <span className="text-zinc-500">v[{idx}]:</span>
                <span className="text-emerald-400 font-bold">{f(val)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
