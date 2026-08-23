import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight, Eye, Compass, Flag, MapPin, ShieldAlert } from 'lucide-react';
import { TransformerActivationTrace } from '../model/labyrinth_transformer';

interface MazeGridVisualizerProps {
  trace: TransformerActivationTrace;
  currentStep: number;
  onStepChange: (step: number) => void;
  selectedLayer: number;
  selectedHead: number;
}

export default function MazeGridVisualizer({
  trace,
  currentStep,
  onStepChange,
  selectedLayer,
  selectedHead,
}: MazeGridVisualizerProps) {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1000); // ms per step
  const [activeOverlay, setActiveOverlay] = useState<'none' | 'attention' | 'logits'>('attention');

  const { maze, totalSteps, stepTrace, layers, nodesInfo } = trace;
  const { agentPos, nodeType, visitedPath, directions } = stepTrace;

  // Auto playback effect
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isPlaying) {
      timer = setInterval(() => {
        if (currentStep < totalSteps) {
          onStepChange(currentStep + 1);
        } else {
          setIsPlaying(false);
        }
      }, playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isPlaying, currentStep, totalSteps, playbackSpeed, onStepChange]);

  const handlePrev = () => {
    if (currentStep > 0) onStepChange(currentStep - 1);
  };

  const handleNext = () => {
    if (currentStep < totalSteps) onStepChange(currentStep + 1);
  };

  // Get attention weights from current agent cell query
  const agentIndex = agentPos[0] * maze.cols + agentPos[1];
  const headAttn = layers[selectedLayer]?.heads[selectedHead]?.attnWeights[agentIndex] || [];

  const getNodeBadgeColor = (type: string) => {
    switch (type) {
      case 'bifurcation':
        return 'bg-amber-500/20 text-amber-400 border-amber-500/40';
      case 'linear':
        return 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40';
      case 'dead_end':
        return 'bg-rose-500/20 text-rose-400 border-rose-500/40';
      case 'goal':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
      case 'start':
        return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40';
      default:
        return 'bg-zinc-800 text-zinc-400 border-zinc-700';
    }
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-5 shadow-xl flex flex-col gap-5">
      {/* Top Header & Node Status */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-zinc-100 flex items-center gap-2">
              <Compass className="w-5 h-5 text-indigo-400" /> Interactive Maze Step Solver
            </h2>
            <span
              className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border uppercase tracking-wider ${getNodeBadgeColor(
                nodeType
              )}`}
            >
              {nodeType === 'bifurcation' ? '⚡ Decision Bifurcation' : nodeType} Node
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Agent at <span className="font-mono text-zinc-200">({agentPos[0]}, {agentPos[1]})</span> | Step{' '}
            <span className="font-mono font-bold text-indigo-400">{currentStep}</span> of {totalSteps}
          </p>
        </div>

        {/* Overlay Selector Toggles */}
        <div className="flex items-center gap-1.5 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
          <span className="text-[10px] uppercase font-bold text-zinc-500 px-2 flex items-center gap-1">
            <Eye className="w-3 h-3" /> Grid View:
          </span>
          <button
            onClick={() => setActiveOverlay('none')}
            className={`px-2.5 py-1 rounded-md transition-all text-xs font-medium ${
              activeOverlay === 'none'
                ? 'bg-zinc-800 text-zinc-100 font-bold'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Clean Grid
          </button>
          <button
            onClick={() => setActiveOverlay('attention')}
            className={`px-2.5 py-1 rounded-md transition-all text-xs font-medium ${
              activeOverlay === 'attention'
                ? 'bg-indigo-500 text-white font-bold shadow-[0_0_10px_rgba(99,102,241,0.4)]'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Attention Heatmap
          </button>
          <button
            onClick={() => setActiveOverlay('logits')}
            className={`px-2.5 py-1 rounded-md transition-all text-xs font-medium ${
              activeOverlay === 'logits'
                ? 'bg-emerald-500 text-zinc-950 font-bold shadow-[0_0_10px_rgba(16,185,129,0.4)]'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Goal Likelihood
          </button>
        </div>
      </div>

      {/* Main Content Area: Grid & Stepping Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* 6x6 Maze Grid */}
        <div className="lg:col-span-7 flex flex-col items-center justify-center p-4 bg-zinc-950 rounded-xl border border-zinc-850 relative">
          <div className="grid grid-cols-6 gap-1.5 w-full max-w-[380px] aspect-square p-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-2xl relative">
            {Array.from({ length: 36 }).map((_, idx) => {
              const r = Math.floor(idx / 6);
              const c = idx % 6;
              const val = maze.grid[r][c];

              const isStart = r === maze.start[0] && c === maze.start[1];
              const isGoal = r === maze.goal[0] && c === maze.goal[1];
              const isAgent = r === agentPos[0] && c === agentPos[1];
              const isVisited =
                !isAgent && visitedPath.some(([vr, vc]) => vr === r && vc === c);
              const isWall = val === 3;
              const nodeInfo = nodesInfo[idx];

              const attnWeight = headAttn[idx] || 0;

              // Color determination
              let bgClass = 'bg-zinc-850 hover:bg-zinc-800 border-zinc-800';
              if (isWall) {
                bgClass = 'bg-zinc-950 border-zinc-900 opacity-90';
              } else if (isAgent) {
                bgClass =
                  'bg-gradient-to-br from-indigo-500 to-violet-600 border-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.6)] z-20 scale-105';
              } else if (isGoal) {
                bgClass = 'bg-amber-500/20 border-amber-500/60 text-amber-400';
              } else if (isStart) {
                bgClass = 'bg-emerald-500/20 border-emerald-500/60 text-emerald-400';
              } else if (isVisited) {
                bgClass = 'bg-indigo-950/50 border-indigo-900/60 text-indigo-300';
              }

              // Attention heat overlay style
              let overlayStyle: React.CSSProperties = {};
              if (activeOverlay === 'attention' && !isWall && !isAgent) {
                const alpha = Math.min(1.0, attnWeight * 2.5);
                overlayStyle = {
                  backgroundColor: `rgba(99, 102, 241, ${alpha})`,
                };
              }

              return (
                <div
                  key={idx}
                  style={overlayStyle}
                  className={`relative aspect-square rounded-md border flex flex-col items-center justify-center p-1 transition-all duration-200 select-none ${bgClass}`}
                  title={`Cell (${r},${c}) | Type: ${nodeInfo.type} | Attn: ${(
                    attnWeight * 100
                  ).toFixed(1)}%`}
                >
                  {/* Grid cell labels */}
                  {isAgent ? (
                    <div className="flex flex-col items-center">
                      <MapPin className="w-5 h-5 text-white animate-bounce" />
                      <span className="text-[8px] font-extrabold text-white uppercase tracking-tighter">
                        AGENT
                      </span>
                    </div>
                  ) : isGoal ? (
                    <div className="flex flex-col items-center">
                      <Flag className="w-4 h-4 text-amber-400" />
                      <span className="text-[8px] font-bold text-amber-400">GOAL</span>
                    </div>
                  ) : isStart ? (
                    <div className="flex flex-col items-center">
                      <span className="text-xs font-black text-emerald-400">S</span>
                      <span className="text-[8px] text-emerald-400 font-bold">START</span>
                    </div>
                  ) : isWall ? (
                    <span className="text-[9px] font-mono text-zinc-700">#</span>
                  ) : isVisited ? (
                    <div className="w-2 h-2 rounded-full bg-indigo-400/60" />
                  ) : (
                    nodeInfo.type === 'bifurcation' && (
                      <span className="text-[8px] font-bold text-amber-400/80 bg-amber-500/10 px-1 rounded">
                        ⚡
                      </span>
                    )
                  )}

                  {/* Attention Percentage Overlay Text */}
                  {activeOverlay === 'attention' && !isWall && !isAgent && (
                    <span className="absolute bottom-0.5 right-0.5 text-[8px] font-mono font-extrabold text-white/90 bg-black/40 px-0.5 rounded">
                      {(attnWeight * 100).toFixed(0)}%
                    </span>
                  )}

                  {/* Coordinate Label */}
                  <span className="absolute top-0.5 left-0.5 text-[7px] font-mono text-zinc-600">
                    {r},{c}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Grid Legend */}
          <div className="flex flex-wrap items-center justify-center gap-3 mt-3 text-[10px] text-zinc-400">
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-indigo-500" /> Current Agent
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-emerald-500/40 border border-emerald-500" /> Start
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-amber-500/40 border border-amber-500" /> Goal
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-indigo-900/60" /> Visited Path
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-amber-500/20 text-amber-400">⚡</span> Bifurcation
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded bg-zinc-950 border border-zinc-900" /> Wall
            </span>
          </div>
        </div>

        {/* Stepping Controls & Candidate Direction Probabilities */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Stepping Controller Box */}
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 flex flex-col gap-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center justify-between">
              <span>Advance Solver Inference</span>
              <span className="text-[10px] font-mono text-indigo-400">
                Step {currentStep} / {totalSteps}
              </span>
            </h3>

            {/* Step Progress Bar */}
            <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden border border-zinc-800">
              <div
                style={{ width: `${(currentStep / (totalSteps || 1)) * 100}%` }}
                className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full transition-all duration-300"
              />
            </div>

            {/* Main Stepper Slider Controls */}
            <input
              type="range"
              min={0}
              max={totalSteps}
              value={currentStep}
              onChange={e => onStepChange(parseInt(e.target.value, 10))}
              className="w-full accent-indigo-500 bg-zinc-900 rounded cursor-pointer"
            />

            {/* Control Buttons */}
            <div className="flex items-center justify-between gap-2 pt-1">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => onStepChange(0)}
                  disabled={currentStep === 0}
                  className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 disabled:opacity-30 disabled:pointer-events-none transition-all"
                  title="Jump to Start"
                >
                  <SkipBack className="w-4 h-4" />
                </button>
                <button
                  onClick={handlePrev}
                  disabled={currentStep === 0}
                  className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 disabled:opacity-30 disabled:pointer-events-none transition-all"
                  title="Previous Step"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
              </div>

              {/* Play / Pause Toggle Button */}
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
                  isPlaying
                    ? 'bg-amber-500 text-zinc-950 shadow-[0_0_15px_rgba(245,158,11,0.4)]'
                    : 'bg-indigo-500 text-white shadow-[0_0_15px_rgba(99,102,241,0.4)]'
                }`}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" /> Pause Auto
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> Auto Play
                  </>
                )}
              </button>

              <div className="flex items-center gap-1">
                <button
                  onClick={handleNext}
                  disabled={currentStep === totalSteps}
                  className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 disabled:opacity-30 disabled:pointer-events-none transition-all"
                  title="Next Step"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => onStepChange(totalSteps)}
                  disabled={currentStep === totalSteps}
                  className="p-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-300 disabled:opacity-30 disabled:pointer-events-none transition-all"
                  title="Jump to Goal"
                >
                  <SkipForward className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Playback speed selector */}
            <div className="flex items-center justify-between text-[10px] text-zinc-500 pt-1 border-t border-zinc-900">
              <span>Auto-stepping Speed:</span>
              <div className="flex gap-1">
                {[
                  { label: '0.5s', val: 500 },
                  { label: '1.0s', val: 1000 },
                  { label: '2.0s', val: 2000 },
                ].map(s => (
                  <button
                    key={s.val}
                    onClick={() => setPlaybackSpeed(s.val)}
                    className={`px-2 py-0.5 rounded ${
                      playbackSpeed === s.val
                        ? 'bg-zinc-800 text-zinc-200 font-bold'
                        : 'text-zinc-500 hover:text-zinc-300'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Directional Move Candidate Probabilities & Logits */}
          <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 flex flex-col gap-2.5">
            <h4 className="text-xs font-bold text-zinc-300 flex items-center justify-between">
              <span>Next-Step Move Probabilities</span>
              <span className="text-[10px] text-zinc-500 font-normal">
                Softmax over FC Logits
              </span>
            </h4>

            <div className="space-y-2">
              {directions.map(dir => (
                <div key={dir.direction} className="space-y-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
                      {dir.direction === 'Up' && '▲ Up'}
                      {dir.direction === 'Down' && '▼ Down'}
                      {dir.direction === 'Left' && '◀ Left'}
                      {dir.direction === 'Right' && '▶ Right'}
                      {dir.isOptimal && (
                        <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.2 rounded border border-emerald-500/30">
                          Optimal Path
                        </span>
                      )}
                      {dir.distToGoal === -1 && (
                        <span className="text-[9px] bg-rose-500/20 text-rose-400 px-1.5 py-0.2 rounded border border-rose-500/30 flex items-center gap-0.5">
                          <ShieldAlert className="w-2.5 h-2.5" /> Wall
                        </span>
                      )}
                    </span>
                    <span className="font-mono text-zinc-400 text-[10px]">
                      {(dir.probability * 100).toFixed(1)}% ({dir.logit.toFixed(2)})
                    </span>
                  </div>

                  <div className="w-full bg-zinc-900 h-1.5 rounded-full overflow-hidden">
                    <div
                      style={{ width: `${dir.probability * 100}%` }}
                      className={`h-full transition-all duration-300 ${
                        dir.isOptimal
                          ? 'bg-emerald-500'
                          : dir.distToGoal === -1
                          ? 'bg-rose-500/40'
                          : 'bg-indigo-500'
                      }`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
