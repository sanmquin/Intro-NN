import { useState, useMemo } from 'react';
import { Sparkles, Brain, Cpu, RefreshCw, BarChart2 } from 'lucide-react';
import { runInference, getMagnitudeBilinearScores } from './model/transformer';
import AttentionHeatmap from './components/AttentionHeatmap';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import MathExplainer from './components/MathExplainer';

export default function App() {
  const [inputTokens, setInputTokens] = useState<number[]>([4, 2, 8, 1, 3]);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [activeMathBlock, setActiveMathBlock] = useState<string>('layer1');
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);

  const trace = useMemo(() => {
    return runInference(inputTokens);
  }, [inputTokens]);

  const magnitudeMatrix = useMemo(() => {
    return getMagnitudeBilinearScores();
  }, []);

  const presets = [
    { name: 'Mixed Random', seq: [4, 2, 8, 1, 3] },
    { name: 'Reverse Sorted', seq: [9, 7, 5, 3, 1] },
    { name: 'Fully Sorted', seq: [0, 2, 4, 6, 8] },
    { name: 'Duplicate Values', seq: [5, 2, 5, 1, 2] },
  ];

  const handlePresetClick = (seq: number[]) => {
    setInputTokens([...seq]);
  };

  const handleDigitChange = (idx: number, newVal: number) => {
    const updated = [...inputTokens];
    updated[idx] = Math.min(Math.max(newVal, 0), 9);
    setInputTokens(updated);
  };

  const onSelectHead = (layer: number, head: number) => {
    setSelectedLayer(layer);
    setSelectedHead(head);
    setActiveMathBlock(layer === 0 ? 'layer1' : 'layer2');
  };

  const getBilinearCellColor = (val: number, min: number, max: number) => {
    const norm = (val - min) / (max - min || 1);
    const r = Math.round(16 + norm * 20);
    const g = Math.round(24 + norm * 160);
    const b = Math.round(32 + norm * 180);
    return `rgb(${r}, ${g}, ${b})`;
  };

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

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">

      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500 to-violet-500 rounded-xl shadow-[0_0_15px_rgba(99,102,241,0.25)]">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-md font-bold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Transformer Attention Visualizer
            </h1>
            <p className="text-[10px] text-zinc-500 font-medium">
              Interactive Interpretability & Numerical Forward Trace
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-indigo-500/10 border border-indigo-500/25 text-indigo-400 flex items-center gap-1">
            <Cpu className="w-2.5 h-2.5" /> Trained Sorter Model
          </span>
          <span className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-emerald-500/10 border border-emerald-500/25 text-emerald-400">
            Accuracy: 99.9%
          </span>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-zinc-900/40 border border-zinc-900 rounded-xl p-5 flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-1.5 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Presets
              </h3>
              <p className="text-[11px] text-zinc-400 leading-relaxed mb-4">
                Choose any pre-configured sequence to test edge cases or reversed arrays.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => handlePresetClick(preset.seq)}
                  className="px-3 py-2 rounded-lg bg-zinc-950 border border-zinc-850 hover:border-zinc-700 text-[10px] font-medium transition-all text-zinc-300 hover:text-zinc-100 flex items-center justify-between"
                >
                  <span>{preset.name}</span>
                  <span className="text-[9px] font-mono text-zinc-500">[{preset.seq.join(',')}]</span>
                </button>
              ))}
            </div>
          </div>

          <div className="lg:col-span-2 bg-zinc-900/60 border border-zinc-850/80 rounded-xl p-5 flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-1 flex items-center gap-1.5">
                <RefreshCw className="w-3.5 h-3.5 text-violet-400" /> Interactive Input Sequencer
              </h3>
              <p className="text-[11px] text-zinc-400 leading-relaxed mb-4">
                Click on the arrows below to increment or decrement the tokens of the 5-length sequence.
              </p>
            </div>

            <div className="flex justify-between items-center gap-4 bg-zinc-950/80 p-4 rounded-xl border border-zinc-900">
              {inputTokens.map((tok, idx) => (
                <div key={idx} className="flex flex-col items-center gap-1">
                  <button
                    onClick={() => handleDigitChange(idx, tok + 1)}
                    disabled={tok >= 9}
                    className="w-8 h-5 rounded flex items-center justify-center bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs text-zinc-400 hover:text-zinc-200 disabled:opacity-30 disabled:pointer-events-none"
                  >
                    ▲
                  </button>

                  <div className="w-12 h-12 rounded-xl bg-gradient-to-b from-zinc-850 to-zinc-900 border border-zinc-750 flex items-center justify-center shadow-[0_4px_12px_rgba(0,0,0,0.25)] relative overflow-hidden">
                    <span className="text-lg font-extrabold text-zinc-100 font-mono">
                      {tok}
                    </span>
                    <span className="absolute bottom-1 text-[8px] text-zinc-500 font-semibold uppercase tracking-wider">
                      P{idx + 1}
                    </span>
                  </div>

                  <button
                    onClick={() => handleDigitChange(idx, tok - 1)}
                    disabled={tok <= 0}
                    className="w-8 h-5 rounded flex items-center justify-center bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-xs text-zinc-400 hover:text-zinc-200 disabled:opacity-30 disabled:pointer-events-none"
                  >
                    ▼
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section>
          <ArchitectureDiagram
            trace={trace}
            selectedLayer={selectedLayer}
            onSelectLayer={setSelectedLayer}
            selectedHead={selectedHead}
            activeMathBlock={activeMathBlock}
            setActiveMathBlock={setActiveMathBlock}
          />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <AttentionHeatmap
              trace={trace}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
              onSelectHead={onSelectHead}
              hoveredTokenIdx={hoveredTokenIdx}
              setHoveredTokenIdx={setHoveredTokenIdx}
            />
          </div>

          <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-6 backdrop-blur-sm flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-2 mb-1.5">
                <BarChart2 className="w-4 h-4 text-cyan-400" /> learned Ordering Bias
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed mb-4">
                Bilinear Query-Key magnitude weights matrix:
                {" $$\\mathbf{Bias}(u, v) = \\mathbf{e}_u \\mathbf{W}_q \\mathbf{W}_k^T \\mathbf{e}_v^T$$"}
                The diagonal pattern demonstrates that the model maps integers on a continuous numerical scale, aligning matching magnitudes.
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
                      return (
                        <div
                          key={`${uIdx}-${vIdx}`}
                          title={`Bilinear Score (${uIdx}, ${vIdx}): ${val.toFixed(2)}`}
                          style={{
                            backgroundColor: getBilinearCellColor(val, magnitudeMinMax.min, magnitudeMinMax.max),
                          }}
                          className="w-full aspect-square rounded-[1px] hover:ring-1 hover:ring-white transition-all cursor-crosshair"
                        />
                      );
                    })
                  )}
                </div>
              </div>
            </div>

            <div className="mt-4 p-3 bg-zinc-950/60 border border-zinc-850/80 rounded-lg text-[10px] text-zinc-500 leading-relaxed">
              *Computed directly from Layer 1 weight parameters. The self-attention matrix converges to this continuous, diagonally-aligned magnitude bias.
            </div>
          </div>
        </section>

        <section>
          <MathExplainer
            trace={trace}
            selectedLayer={selectedLayer}
            selectedHead={selectedHead}
            activeMathBlock={activeMathBlock}
          />
        </section>

      </main>

      <footer className="border-t border-zinc-900 bg-zinc-950/60 py-6 px-6 text-center text-xs text-zinc-500">
        <p className="max-w-2xl mx-auto leading-relaxed">
          This visualizer illustrates how a Transformer encoder learns to route elements transitively based on contextual inputs. By dissecting Q, K, and V spaces, students and engineers can trace the exact mechanism of self-attention.
        </p>
        <p className="mt-2 text-[10px] text-zinc-600 font-mono">
          Made with ✨ for educational research. Deployed to Netlify.
        </p>
      </footer>
    </div>
  );
}
