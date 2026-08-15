import { ArrowRight, Layers, Shuffle, Sparkles, Binary } from 'lucide-react';
import { TransformerActivationTrace } from '../model/transformer';

interface ArchitectureDiagramProps {
  trace: TransformerActivationTrace;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  selectedHead: number;
  activeMathBlock: string;
  setActiveMathBlock: (block: string) => void;
}

export default function ArchitectureDiagram({
  trace,
  selectedLayer,
  onSelectLayer,
  selectedHead,
  activeMathBlock,
  setActiveMathBlock,
}: ArchitectureDiagramProps) {
  const blocks = [
    {
      id: 'embed',
      name: 'Embedding + PE',
      desc: 'Token embedding layer projects integers 0-9 into 32D dense vectors, adding sinusoidal positional encodings.',
      icon: <Shuffle className="w-4 h-4 text-cyan-400" />,
      color: 'border-cyan-500/30 hover:border-cyan-400 bg-cyan-950/10 text-cyan-400',
      activeColor: 'border-cyan-400 ring-2 ring-cyan-400/20 bg-cyan-950/20 text-cyan-300',
    },
    {
      id: 'layer1',
      name: 'Transformer Layer 1',
      desc: 'Pre-LN Multi-Head Attention (2 Heads, 16D each) & Position-wise Feed-Forward Network with GELU.',
      icon: <Layers className="w-4 h-4 text-violet-400" />,
      color: 'border-violet-500/30 hover:border-violet-400 bg-violet-950/10 text-violet-400',
      activeColor: 'border-violet-400 ring-2 ring-violet-400/20 bg-violet-950/20 text-violet-300',
    },
    {
      id: 'layer2',
      name: 'Transformer Layer 2',
      desc: 'Processes intermediate context vectors via Layer 2 Pre-LN MHA & FFN, mapping hidden routing patterns.',
      icon: <Layers className="w-4 h-4 text-indigo-400" />,
      color: 'border-indigo-500/30 hover:border-indigo-400 bg-indigo-950/10 text-indigo-400',
      activeColor: 'border-indigo-400 ring-2 ring-indigo-400/20 bg-indigo-950/20 text-indigo-300',
    },
    {
      id: 'output',
      name: 'Classifier Head',
      desc: 'Projects 32D Context through Final LayerNorm & Linear projection to output 10D vocab logits.',
      icon: <Binary className="w-4 h-4 text-emerald-400" />,
      color: 'border-emerald-500/30 hover:border-emerald-400 bg-emerald-950/10 text-emerald-400',
      activeColor: 'border-emerald-400 ring-2 ring-emerald-400/20 bg-emerald-950/20 text-emerald-300',
    },
  ];

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm">
      <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2 mb-2">
        <span className="text-indigo-400">⚡</span> Transformer Architecture
      </h2>
      <p className="text-xs text-zinc-400 mb-6">
        Click on any block below to inspect its mathematical formulas and active numerical traces.
      </p>

      <div className="flex flex-col lg:flex-row items-center gap-4 justify-between bg-zinc-950 p-6 rounded-xl border border-zinc-800">

        <div className="flex flex-col items-center gap-1 min-w-[100px] text-center">
          <span className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wider">Input</span>
          <div className="flex gap-1.5 mt-1 bg-zinc-900 p-2 rounded-lg border border-zinc-800">
            {trace.inputTokens.map((tok, idx) => (
              <span key={idx} className="w-7 h-7 rounded flex items-center justify-center bg-zinc-800 border border-zinc-700 text-xs font-bold text-zinc-300">
                {tok}
              </span>
            ))}
          </div>
        </div>

        <ArrowRight className="w-5 h-5 text-zinc-600 rotate-90 lg:rotate-0" />

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:flex items-center gap-4 flex-1 w-full lg:w-auto">
          {blocks.map((block) => {
            const isLayer1 = block.id === 'layer1';
            const isLayer2 = block.id === 'layer2';
            const isActive = activeMathBlock === block.id;

            return (
              <div
                key={block.id}
                onClick={() => {
                  setActiveMathBlock(block.id);
                  if (isLayer1) onSelectLayer(0);
                  if (isLayer2) onSelectLayer(1);
                }}
                className={`flex-1 min-h-[96px] cursor-pointer p-3.5 rounded-xl border transition-all duration-200 text-left select-none ${
                  isActive ? block.activeColor : block.color
                }`}
              >
                <div className="flex items-center gap-2">
                  <div className="p-1 rounded bg-zinc-900/80 border border-zinc-800">
                    {block.icon}
                  </div>
                  <span className="text-xs font-semibold tracking-wide">
                    {block.name}
                  </span>
                </div>
                <p className="text-[10px] text-zinc-400 mt-2 leading-relaxed line-clamp-2">
                  {block.desc}
                </p>

                {(isLayer1 || isLayer2) && (
                  <div className="flex gap-1.5 mt-2.5">
                    <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
                      selectedLayer === (isLayer1 ? 0 : 1) && selectedHead === 0
                        ? 'bg-violet-500 text-white font-bold'
                        : 'bg-zinc-900/60 text-zinc-500'
                    }`}>
                      Head 1
                    </span>
                    <span className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
                      selectedLayer === (isLayer1 ? 0 : 1) && selectedHead === 1
                        ? 'bg-violet-500 text-white font-bold'
                        : 'bg-zinc-900/60 text-zinc-500'
                    }`}>
                      Head 2
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <ArrowRight className="w-5 h-5 text-zinc-600 rotate-90 lg:rotate-0" />

        <div className="flex flex-col items-center gap-1 min-w-[100px] text-center">
          <span className="text-[10px] text-emerald-500 uppercase font-semibold tracking-wider flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-emerald-400" /> Predictions
          </span>
          <div className="flex gap-1.5 mt-1 bg-emerald-950/10 p-2 rounded-lg border border-emerald-900/30">
            {trace.predictions.map((tok, idx) => (
              <span key={idx} className="w-7 h-7 rounded flex items-center justify-center bg-emerald-900/20 border border-emerald-500/30 text-xs font-bold text-emerald-400">
                {tok}
              </span>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
