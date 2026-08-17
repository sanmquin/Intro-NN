import { ArrowRight, Layers, Shuffle, Sparkles, Compass } from 'lucide-react';
import { TransformerActivationTrace } from '../model/labyrinth_transformer';

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
      name: 'Grid + 2D Spatial PE',
      desc: '36 Grid Cell tokens (Path, Wall, Start, Goal, Agent, Visited) + 2D Sinusoidal Spatial Positional Encodings into 32D.',
      icon: <Shuffle className="w-4 h-4 text-cyan-400" />,
      color: 'border-cyan-500/30 hover:border-cyan-400 bg-cyan-950/10 text-cyan-400',
      activeColor: 'border-cyan-400 ring-2 ring-cyan-400/20 bg-cyan-950/20 text-cyan-300',
    },
    {
      id: 'layer1',
      name: 'Labyrinth Layer 1',
      desc: 'Topological Self-Attention (2 Heads, 16D each) & Feed-Forward Network processing local spatial cell adjacency.',
      icon: <Layers className="w-4 h-4 text-violet-400" />,
      color: 'border-violet-500/30 hover:border-violet-400 bg-violet-950/10 text-violet-400',
      activeColor: 'border-violet-400 ring-2 ring-violet-400/20 bg-violet-950/20 text-violet-300',
    },
    {
      id: 'layer2',
      name: 'Labyrinth Layer 2',
      desc: 'Global Bifurcation & Goal Reachability Attention, routing long-range decision nodes across the maze topology.',
      icon: <Layers className="w-4 h-4 text-indigo-400" />,
      color: 'border-indigo-500/30 hover:border-indigo-400 bg-indigo-950/10 text-indigo-400',
      activeColor: 'border-indigo-400 ring-2 ring-indigo-400/20 bg-indigo-950/20 text-indigo-300',
    },
    {
      id: 'output',
      name: 'Classifier Move FC',
      desc: 'Projects agent position representation through Final LayerNorm to 4 directional logits (Up, Down, Left, Right).',
      icon: <Compass className="w-4 h-4 text-emerald-400" />,
      color: 'border-emerald-500/30 hover:border-emerald-400 bg-emerald-950/10 text-emerald-400',
      activeColor: 'border-emerald-400 ring-2 ring-emerald-400/20 bg-emerald-950/20 text-emerald-300',
    },
  ];

  const { stepTrace, directionalProbs } = trace;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm">
      <h2 className="text-xl font-semibold text-zinc-100 flex items-center gap-2 mb-2">
        <span className="text-indigo-400">⚡</span> Labyrinth Transformer Architecture Pipeline
      </h2>
      <p className="text-xs text-zinc-400 mb-6">
        Step-by-step forward pass processing 36 spatial grid cells through 2 Multi-Head Attention Encoder layers.
      </p>

      <div className="flex flex-col lg:flex-row items-center gap-4 justify-between bg-zinc-950 p-6 rounded-xl border border-zinc-800">
        {/* Input State */}
        <div className="flex flex-col items-center gap-1 min-w-[110px] text-center">
          <span className="text-[10px] text-zinc-500 uppercase font-semibold tracking-wider">Agent Location</span>
          <div className="mt-1 bg-zinc-900 px-3 py-2 rounded-lg border border-zinc-800 font-mono text-xs font-bold text-indigo-400">
            ({stepTrace.agentPos[0]}, {stepTrace.agentPos[1]})
          </div>
          <span className="text-[9px] text-zinc-500 mt-0.5">{stepTrace.nodeType.toUpperCase()}</span>
        </div>

        <ArrowRight className="w-5 h-5 text-zinc-600 rotate-90 lg:rotate-0" />

        {/* Architecture Blocks */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:flex items-center gap-4 flex-1 w-full lg:w-auto">
          {blocks.map(block => {
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
                  <div className="p-1 rounded bg-zinc-900/80 border border-zinc-800">{block.icon}</div>
                  <span className="text-xs font-semibold tracking-wide">{block.name}</span>
                </div>
                <p className="text-[10px] text-zinc-400 mt-2 leading-relaxed line-clamp-2">{block.desc}</p>

                {(isLayer1 || isLayer2) && (
                  <div className="flex gap-1.5 mt-2.5">
                    <span
                      className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
                        selectedLayer === (isLayer1 ? 0 : 1) && selectedHead === 0
                          ? 'bg-violet-500 text-white font-bold'
                          : 'bg-zinc-900/60 text-zinc-500'
                      }`}
                    >
                      Head 0 (Topological)
                    </span>
                    <span
                      className={`text-[8px] px-1.5 py-0.5 rounded font-mono ${
                        selectedLayer === (isLayer1 ? 0 : 1) && selectedHead === 1
                          ? 'bg-violet-500 text-white font-bold'
                          : 'bg-zinc-900/60 text-zinc-500'
                      }`}
                    >
                      Head 1 (Spatial)
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <ArrowRight className="w-5 h-5 text-zinc-600 rotate-90 lg:rotate-0" />

        {/* Output Action */}
        <div className="flex flex-col items-center gap-1 min-w-[120px] text-center">
          <span className="text-[10px] text-emerald-500 uppercase font-semibold tracking-wider flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-emerald-400" /> Move Softmax
          </span>
          <div className="mt-1 bg-emerald-950/20 px-3 py-1.5 rounded-lg border border-emerald-500/30 font-mono text-[11px] font-bold text-emerald-400">
            Top: {(Math.max(...Object.values(directionalProbs)) * 100).toFixed(0)}%
          </div>
        </div>
      </div>
    </div>
  );
}
