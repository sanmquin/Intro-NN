import { useState, useMemo } from 'react';
import { Sparkles, Brain, Cpu, Network, Database, Layers, GitFork, Compass } from 'lucide-react';
import {
  runLabyrinthInference,
  PRESET_MAZES,
  MazeGrid,
} from './model/labyrinth_transformer';
import MazeGridVisualizer from './components/MazeGridVisualizer';
import ResearchQuestionsPanel from './components/ResearchQuestionsPanel';
import AttentionHeatmap from './components/AttentionHeatmap';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import SpatialBiasMap from './components/SpatialBiasMap';
import ParameterInspector from './components/ParameterInspector';
import LogitInfluenceVisualizer from './components/LogitInfluenceVisualizer';

type MainTab = 'research' | 'attention' | 'spatial_bias' | 'parameters' | 'influence';

export default function App() {
  const [selectedMaze, setSelectedMaze] = useState<MazeGrid>(PRESET_MAZES[0]);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [selectedLayer, setSelectedLayer] = useState<number>(1); // Layer 2 default for topological attention
  const [selectedHead, setSelectedHead] = useState<number>(0);   // Head 0 default for topological routing
  const [activeMathBlock, setActiveMathBlock] = useState<string>('layer2');
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<MainTab>('research');

  // Run real-time Labyrinth Transformer inference
  const trace = useMemo(() => {
    return runLabyrinthInference(selectedMaze, currentStep);
  }, [selectedMaze, currentStep]);

  const handleMazeSelect = (maze: MazeGrid) => {
    setSelectedMaze(maze);
    setCurrentStep(0);
  };

  const onSelectHead = (layer: number, head: number) => {
    setSelectedLayer(layer);
    setSelectedHead(head);
    setActiveMathBlock(layer === 0 ? 'layer1' : 'layer2');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* App Header */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500 via-purple-500 to-violet-600 rounded-xl shadow-[0_0_15px_rgba(99,102,241,0.3)]">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-md font-bold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Labyrinth Transformer Visualizer
            </h1>
            <p className="text-[10px] text-zinc-500 font-medium">
              Step-by-Step Maze Solver &amp; Attention Interpretability Engine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/10 border border-indigo-500/25 text-indigo-400 flex items-center gap-1">
            <Cpu className="w-3 h-3" /> Step-by-Step Solver
          </span>
          <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 border border-emerald-500/25 text-emerald-400">
            Topological Routing: 100%
          </span>
        </div>
      </header>

      {/* Main Navigation Tabs */}
      <div className="bg-zinc-950/60 border-b border-zinc-900 sticky top-[65px] z-20 px-6 py-2.5 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center gap-2">
          <button
            onClick={() => setActiveTab('research')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'research'
                ? 'bg-amber-500 text-zinc-950 shadow-[0_0_12px_rgba(245,158,11,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <GitFork className="w-3.5 h-3.5" /> Research Questions &amp; Analytics
          </button>

          <button
            onClick={() => setActiveTab('attention')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'attention'
                ? 'bg-indigo-500 text-white shadow-[0_0_12px_rgba(99,102,241,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Compass className="w-3.5 h-3.5" /> Step Solver &amp; Attention Maps
          </button>

          <button
            onClick={() => setActiveTab('spatial_bias')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'spatial_bias'
                ? 'bg-cyan-500 text-zinc-950 shadow-[0_0_12px_rgba(6,182,212,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Network className="w-3.5 h-3.5" /> Learned Spatial Bias Map
          </button>

          <button
            onClick={() => setActiveTab('parameters')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2 ${
              activeTab === 'parameters'
                ? 'bg-violet-500 text-white shadow-[0_0_12px_rgba(139,92,246,0.3)] font-bold'
                : 'bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800'
            }`}
          >
            <Database className="w-3.5 h-3.5" /> Network Parameter Inspector
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

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Preset Maze Selector Bar */}
        <section className="bg-zinc-900/40 border border-zinc-900 rounded-xl p-5">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-3">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Select Maze Preset
              </h3>
              <p className="text-[11px] text-zinc-400 mt-0.5">
                Choose a maze configuration to analyze attention routing across different topological structures.
              </p>
            </div>
            <div className="text-xs font-mono text-zinc-400 bg-zinc-950 px-3 py-1.5 rounded-lg border border-zinc-850">
              Active: <strong className="text-indigo-400">{selectedMaze.name}</strong>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {PRESET_MAZES.map(maze => {
              const isSelected = selectedMaze.id === maze.id;
              return (
                <button
                  key={maze.id}
                  onClick={() => handleMazeSelect(maze)}
                  className={`p-3.5 rounded-xl border text-left transition-all flex flex-col justify-between ${
                    isSelected
                      ? 'bg-indigo-950/40 border-indigo-500 shadow-[0_0_12px_rgba(99,102,241,0.25)] text-indigo-200'
                      : 'bg-zinc-950 border-zinc-850 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold truncate">{maze.name}</span>
                      {isSelected && <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />}
                    </div>
                    <p className="text-[10px] text-zinc-400 leading-relaxed line-clamp-2">
                      {maze.description}
                    </p>
                  </div>
                  <div className="mt-2 text-[9px] font-mono text-zinc-500 flex justify-between border-t border-zinc-900 pt-2">
                    <span>6x6 Grid</span>
                    <span>Start (0,0) &rarr; Goal (5,5)</span>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* Step-by-Step Maze Visualizer (Always visible at top for active step context) */}
        <section>
          <MazeGridVisualizer
            trace={trace}
            currentStep={currentStep}
            onStepChange={setCurrentStep}
            selectedLayer={selectedLayer}
            selectedHead={selectedHead}
          />
        </section>

        {/* Tab 1: Research Questions Panel */}
        {activeTab === 'research' && (
          <section>
            <ResearchQuestionsPanel trace={trace} />
          </section>
        )}

        {/* Tab 2: Attention & Step Trace */}
        {activeTab === 'attention' && (
          <div className="space-y-6">
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

            <section>
              <AttentionHeatmap
                trace={trace}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                onSelectHead={onSelectHead}
                hoveredTokenIdx={hoveredTokenIdx}
                setHoveredTokenIdx={setHoveredTokenIdx}
              />
            </section>
          </div>
        )}

        {/* Tab 3: Learned Spatial Bias Map */}
        {activeTab === 'spatial_bias' && (
          <section>
            <SpatialBiasMap maze={selectedMaze} />
          </section>
        )}

        {/* Tab 4: Network Parameter Inspector */}
        {activeTab === 'parameters' && (
          <section>
            <ParameterInspector />
          </section>
        )}

        {/* Tab 5: Classifier Logit Influence */}
        {activeTab === 'influence' && (
          <section>
            <LogitInfluenceVisualizer trace={trace} />
          </section>
        )}
      </main>

      <footer className="border-t border-zinc-900 bg-zinc-950/60 py-6 px-6 text-center text-xs text-zinc-500">
        <p className="max-w-2xl mx-auto leading-relaxed">
          This Labyrinth Transformer Visualizer illustrates how self-attention mechanisms perform step-by-step spatial navigation and topological choice-node routing. By analyzing Q-K projections across bifurcations, researchers and students can trace exact reachability representations.
        </p>
        <p className="mt-2 text-[10px] text-zinc-600 font-mono">
          Made with ✨ for educational research. Deployed to Netlify.
        </p>
      </footer>
    </div>
  );
}
