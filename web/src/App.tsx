import { useState, useMemo } from 'react';
import { Sparkles, Brain, Cpu, RefreshCw, Network, Database, Layers, CheckCircle2 } from 'lucide-react';
import {
  ModelKey,
  MODEL_CONFIGS,
  runInference,
  getNetworkParameterDetails,
} from './model/transformer';
import AttentionHeatmap from './components/AttentionHeatmap';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import MathExplainer from './components/MathExplainer';
import EmbeddingScatterPlot from './components/EmbeddingScatterPlot';
import ParameterInspector from './components/ParameterInspector';
import LogitInfluenceVisualizer from './components/LogitInfluenceVisualizer';
import LearnedOrderingBiasCard from './components/LearnedOrderingBiasCard';
import EducationalStepper from './components/EducationalStepper';

type MainTab = 'attention' | 'embeddings' | 'parameters' | 'influence';

export default function App() {
  const [modelKey, setModelKey] = useState<ModelKey>('1l_2h');
  const [inputTokens, setInputTokens] = useState<number[]>([4, 2, 8, 1, 3]);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [activeMathBlock, setActiveMathBlock] = useState<string>('layer1');
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<MainTab>('attention');
  const [currentStepId, setCurrentStepId] = useState<string>('step_3_attention');

  const trace = useMemo(() => {
    return runInference(inputTokens, modelKey);
  }, [inputTokens, modelKey]);

  const networkDetails = useMemo(() => {
    return getNetworkParameterDetails(modelKey);
  }, [modelKey]);

  const activeModelConfig = MODEL_CONFIGS[modelKey];

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

  const handleModelChange = (key: ModelKey) => {
    setModelKey(key);
    setSelectedLayer(0);
    setSelectedHead(0);
    setActiveMathBlock('layer1');
  };

  const handleStepSelect = (stepId: string) => {
    setCurrentStepId(stepId);
    if (stepId === 'step_1_embed') setActiveMathBlock('embed');
    else if (stepId === 'step_2_projections' || stepId === 'step_3_attention' || stepId === 'step_4_ffn') setActiveMathBlock('layer1');
    else if (stepId === 'step_5_logits') setActiveMathBlock('output');
    else if (stepId === 'step_6_weights') setActiveTab('embeddings');
  };

  const onSelectHead = (layer: number, head: number) => {
    setSelectedLayer(layer);
    setSelectedHead(head);
    setActiveMathBlock(layer === 0 ? 'layer1' : 'layer2');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">

      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500 to-violet-500 rounded-xl shadow-[0_0_15px_rgba(99,102,241,0.25)]">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-md font-bold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Transformer Attention Visualizer & Educational Guide
            </h1>
            <p className="text-[10px] text-zinc-500 font-medium">
              Interactive Interpretability, Ground-Up Matrix Derivations & Model Architecture Explorer
            </p>
          </div>
        </div>

        {/* Model Selector Bar */}
        <div className="flex flex-wrap items-center gap-2 bg-zinc-900/90 p-1.5 rounded-xl border border-zinc-800">
          <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500 px-2 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-indigo-400" /> Architecture:
          </span>

          {(Object.keys(MODEL_CONFIGS) as ModelKey[]).map((key) => {
            const cfg = MODEL_CONFIGS[key];
            const isSelected = modelKey === key;
            return (
              <button
                key={key}
                onClick={() => handleModelChange(key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.35)]'
                    : 'bg-zinc-950 text-zinc-400 hover:text-zinc-200 border border-zinc-800/80'
                }`}
              >
                {isSelected && <CheckCircle2 className="w-3 h-3 text-emerald-300" />}
                <span>{cfg.layers}L-{cfg.heads}H</span>
                <span className="text-[9px] font-mono text-zinc-400 opacity-80">
                  ({cfg.key === '1l_1h' ? '1 Head' : cfg.key === '1l_2h' ? '2 Heads' : '2 Layers'})
                </span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Model Active Description Info Banner */}
      <div className="bg-zinc-900/40 border-b border-zinc-900 px-6 py-2.5 backdrop-blur-md flex flex-col sm:flex-row sm:items-center justify-between text-xs text-zinc-400 gap-2">
        <div className="flex items-center gap-2">
          <span className="font-bold text-zinc-200">{activeModelConfig.name}:</span>
          <span>{activeModelConfig.description}</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px] shrink-0">
          <span className="text-emerald-400 font-bold">Accuracy: 99.9%</span>
          <span className="text-zinc-500">&bull;</span>
          <span className="text-indigo-400">{networkDetails.totalParams.toLocaleString()} parameters</span>
        </div>
      </div>

      {/* Main Navigation Tabs */}
      <div className="bg-zinc-950/60 border-b border-zinc-900 sticky top-[65px] z-20 px-6 py-2.5 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center gap-2">
          <button
            onClick={() => setActiveTab('attention')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'attention'
                ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Brain className="w-3.5 h-3.5" /> Attention & Execution Trace
          </button>

          <button
            onClick={() => setActiveTab('embeddings')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'embeddings'
                ? 'bg-cyan-500 text-zinc-950 shadow-[0_0_12px_rgba(6,182,212,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Network className="w-3.5 h-3.5" /> Vocabulary Embeddings (PCA)
          </button>

          <button
            onClick={() => setActiveTab('parameters')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'parameters'
                ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Database className="w-3.5 h-3.5" /> Network Parameters Inspector
          </button>

          <button
            onClick={() => setActiveTab('influence')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'influence'
                ? 'bg-emerald-500 text-zinc-950 shadow-[0_0_12px_rgba(16,185,129,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Classifier Logit Influence
          </button>
        </div>
      </div>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

        {activeTab === 'attention' && (
          <section>
            <EducationalStepper
              currentStepId={currentStepId}
              onSelectStep={handleStepSelect}
            />
          </section>
        )}

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-zinc-900/40 border border-zinc-900 rounded-xl p-5 flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 mb-1.5 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Sequence Presets
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

          <div>
            <LearnedOrderingBiasCard modelKey={modelKey} />
          </div>
        </section>

        {activeTab === 'attention' && (
          <section>
            <MathExplainer
              trace={trace}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
              activeMathBlock={activeMathBlock}
            />
          </section>
        )}

        {activeTab === 'embeddings' && (
          <section>
            <EmbeddingScatterPlot />
          </section>
        )}

        {activeTab === 'parameters' && (
          <section>
            <ParameterInspector />
          </section>
        )}

        {activeTab === 'influence' && (
          <section>
            <LogitInfluenceVisualizer trace={trace} />
          </section>
        )}

      </main>

      <footer className="border-t border-zinc-900 bg-zinc-950/60 py-6 px-6 text-center text-xs text-zinc-500">
        <p className="max-w-2xl mx-auto leading-relaxed">
          This visualizer illustrates how Transformer encoders learn to route elements transitively based on contextual inputs. Dissect Q, K, and V spaces, inspect ground-up cell matrix derivations, and compare 1-layer vs 2-layer architectures.
        </p>
        <p className="mt-2 text-[10px] text-zinc-600 font-mono">
          Made with ✨ for educational research. Deployed to Netlify.
        </p>
      </footer>
    </div>
  );
}
