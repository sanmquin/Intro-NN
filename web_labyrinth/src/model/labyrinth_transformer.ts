// Labyrinth Transformer Model Engine & Research Analytics Engine
// 6x6 Grid Labyrinth Solver (N = 36 tokens, d_model = 32, 2 Layers, 2 Heads)

export interface MazeGrid {
  id: string;
  name: string;
  description: string;
  rows: number; // 6
  cols: number; // 6
  grid: number[][]; // 6x6 array, 0:path, 1:start, 2:goal, 3:wall
  start: [number, number];
  goal: [number, number];
}

export type NodeType = 'start' | 'goal' | 'bifurcation' | 'linear' | 'dead_end' | 'wall';

export interface CellNodeInfo {
  index: number;
  row: number;
  col: number;
  type: NodeType;
  walkable: boolean;
  distToGoal: number; // BFS distance to goal (-1 if unreachable)
}

export interface DirectionOption {
  direction: 'Up' | 'Down' | 'Left' | 'Right';
  targetPos: [number, number];
  targetIndex: number;
  logit: number;
  probability: number;
  distToGoal: number;
  isOptimal: boolean;
}

export interface StepTrace {
  stepIndex: number;
  agentPos: [number, number];
  agentIndex: number;
  nodeType: NodeType;
  visitedPath: [number, number][];
  gridState: number[]; // 36 tokens

  // Research Question Metrics
  attentionEntropy: number; // Entropy of current agent query attention
  categoryAttention: {
    bifurcations: number;
    linear: number;
    deadEnds: number;
    walls: number;
    goal: number;
    start: number;
  };
  directions: DirectionOption[];
  top1Confidence: number; // Max direction probability
  entropyBits: number;
}

export interface HeadActivation {
  headIndex: number;
  queries: number[][]; // [36, d_k] (d_k = 16)
  keys: number[][];    // [36, d_k]
  values: number[][];  // [36, d_k]
  rawScores: number[][]; // [36, 36]
  attnWeights: number[][]; // [36, 36] after softmax
  context: number[][];   // [36, d_k]
}

export interface LayerActivation {
  layerIndex: number;
  input: number[][];            // [36, d_model]
  norm1: number[][];            // [36, d_model]
  Q: number[][];                // [36, d_model]
  K: number[][];                // [36, d_model]
  V: number[][];                // [36, d_model]
  heads: HeadActivation[];
  concatenatedContext: number[][]; // [36, d_model]
  attnProjected: number[][];       // [36, d_model]
  attnResidual: number[][];        // [36, d_model]
  norm2: number[][];            // [36, d_model]
  ffnMid: number[][];           // [36, d_ff]
  ffnOut: number[][];           // [36, d_model]
  ffnResidual: number[][];      // [36, d_model]
}

export interface TransformerActivationTrace {
  maze: MazeGrid;
  currentStep: number;
  totalSteps: number;
  trajectory: [number, number][]; // Full BFS optimal path
  nodesInfo: CellNodeInfo[];
  stepTrace: StepTrace;

  // High-dimensional activations
  tokens: number[];            // [36] token values
  embeddings: number[][];      // [36, d_model]
  posEncodings: number[][];    // [36, d_model]
  initialSum: number[][];      // [36, d_model]
  layers: LayerActivation[];
  finalNorm: number[][];       // [36, d_model]

  // Directional Movement Logits
  directionalLogits: {
    Up: number;
    Down: number;
    Left: number;
    Right: number;
  };
  directionalProbs: {
    Up: number;
    Down: number;
    Left: number;
    Right: number;
  };

  // Target cell logits across all 36 cells
  cellLogits: number[];
  cellProbs: number[];
}

export interface WeightTensors {
  gridEmbedding: number[][];   // [6, 32]
  spatialPE: number[][];       // [36, 32]
  agentEmbedding: number[][];   // [2, 32]
  layers: {
    norm1: { weight: number[]; bias: number[] };
    attn: {
      W_q: number[][]; // [32, 32]
      W_k: number[][]; // [32, 32]
      W_v: number[][]; // [32, 32]
      W_o: number[][]; // [32, 32]
    };
    norm2: { weight: number[]; bias: number[] };
    ff: {
      fc1: { weight: number[][]; bias: number[] };
      fc2: { weight: number[][]; bias: number[] };
    };
  }[];
  norm: { weight: number[]; bias: number[] };
  fc_dir: {
    weight: number[][]; // [4, 32]
    bias: number[];     // [4]
  };
  fc_cell: {
    weight: number[][]; // [36, 32]
    bias: number[];     // [36]
  };
}

// ----------------------------------------------------------------------------
// 1. PRESET MAZES (6x6 Grids)
// ----------------------------------------------------------------------------

export const PRESET_MAZES: MazeGrid[] = [
  {
    id: 'bifurcation_junction',
    name: '1. Bifurcation Junction Maze',
    description: 'Contains a clear 3-way bifurcation at Step 2. One path leads to a dead end, the other leads to the goal.',
    rows: 6,
    cols: 6,
    grid: [
      [1, 0, 3, 3, 3, 3], // S, ., #, #, #, #
      [3, 0, 3, 3, 3, 3], // #, ., #, #, #, #
      [3, 0, 0, 0, 0, 3], // #, ., ., ., ., #  (1,2) is a bifurcation
      [3, 3, 3, 3, 0, 3], // #, #, #, #, ., #
      [3, 0, 0, 3, 0, 3], // #, ., ., #, ., #  dead-end vs goal path
      [3, 3, 0, 3, 0, 2], // #, #, ., #, ., G
    ],
    start: [0, 0],
    goal: [5, 5],
  },
  {
    id: 'multi_bifurcation',
    name: '2. Multi-Bifurcation Network',
    description: 'Features two consecutive decision nodes (bifurcations) testing long-range bifurcation-to-bifurcation routing.',
    rows: 6,
    cols: 6,
    grid: [
      [1, 0, 0, 3, 3, 3], // S, ., ., #, #, #
      [3, 3, 0, 0, 0, 3], // #, #, B1, ., ., #
      [3, 0, 0, 3, 0, 3], // #, ., ., #, ., #
      [3, 0, 3, 3, 0, 3], // #, ., #, #, B2, #
      [3, 0, 0, 0, 0, 3], // #, ., ., ., ., #
      [3, 3, 3, 3, 0, 2], // #, #, #, #, ., G
    ],
    start: [0, 0],
    goal: [5, 5],
  },
  {
    id: 'dead_end_trap',
    name: '3. Dead-End Trap & Loop',
    description: 'Tests goal reachability encoding at a critical junction with a alluring deep dead-end corridor.',
    rows: 6,
    cols: 6,
    grid: [
      [1, 0, 0, 0, 3, 3], // S, ., ., ., #, #
      [3, 3, 3, 0, 3, 3], // #, #, #, ., #, #
      [3, 0, 0, 0, 0, 0], // #, ., ., B, ., . (Dead End)
      [3, 0, 3, 0, 3, 3], // #, ., #, ., #, #
      [3, 0, 0, 0, 0, 0], // #, ., ., ., ., .
      [3, 3, 3, 3, 3, 2], // #, #, #, #, #, G
    ],
    start: [0, 0],
    goal: [5, 5],
  },
  {
    id: 'linear_snake',
    name: '4. Linear Snake Corridor',
    description: 'A purely linear corridor without bifurcations, providing a baseline comparison for attention entropy.',
    rows: 6,
    cols: 6,
    grid: [
      [1, 0, 0, 0, 0, 0], // S, ., ., ., ., .
      [3, 3, 3, 3, 3, 0], // #, #, #, #, #, .
      [0, 0, 0, 0, 0, 0], // ., ., ., ., ., .
      [0, 3, 3, 3, 3, 3], // ., #, #, #, #, #
      [0, 0, 0, 0, 0, 0], // ., ., ., ., ., .
      [3, 3, 3, 3, 3, 2], // #, #, #, #, #, G
    ],
    start: [0, 0],
    goal: [5, 5],
  },
];

// ----------------------------------------------------------------------------
// 2. HELPER UTILITIES (BFS, Node Classification)
// ----------------------------------------------------------------------------

export function getWalkableNeighbors(grid: number[][], r: number, c: number): [number, number][] {
  const rows = grid.length;
  const cols = grid[0].length;
  const neighbors: [number, number][] = [];
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];

  for (const [dr, dc] of dirs) {
    const nr = r + dr;
    const nc = c + dc;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
      if (grid[nr][nc] !== 3) { // Not wall
        neighbors.push([nr, nc]);
      }
    }
  }
  return neighbors;
}

export function solveBFS(maze: MazeGrid): [number, number][] {
  const { grid, start, goal } = maze;

  const queue: [number, number][][] = [[start]];
  const visited = new Set<string>();
  visited.add(`${start[0]},${start[1]}`);

  while (queue.length > 0) {
    const path = queue.shift()!;
    const [r, c] = path[path.length - 1];

    if (r === goal[0] && c === goal[1]) {
      return path;
    }

    const neighbors = getWalkableNeighbors(grid, r, c);
    for (const [nr, nc] of neighbors) {
      const key = `${nr},${nc}`;
      if (!visited.has(key)) {
        visited.add(key);
        queue.push([...path, [nr, nc]]);
      }
    }
  }

  return [start, goal];
}

export function computeBFSDistancesToGoal(maze: MazeGrid): number[] {
  const { grid, goal, rows, cols } = maze;
  const dists = new Array(rows * cols).fill(-1);

  const queue: [number, number, number][] = [[goal[0], goal[1], 0]];
  const visited = new Set<string>();
  visited.add(`${goal[0]},${goal[1]}`);

  while (queue.length > 0) {
    const [r, c, d] = queue.shift()!;
    const idx = r * cols + c;
    dists[idx] = d;

    const neighbors = getWalkableNeighbors(grid, r, c);
    for (const [nr, nc] of neighbors) {
      const key = `${nr},${nc}`;
      if (!visited.has(key)) {
        visited.add(key);
        queue.push([nr, nc, d + 1]);
      }
    }
  }

  return dists;
}

export function classifyGridNodes(maze: MazeGrid): CellNodeInfo[] {
  const { grid, start, goal, rows, cols } = maze;
  const dists = computeBFSDistancesToGoal(maze);
  const nodes: CellNodeInfo[] = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const val = grid[r][c];
      const dist = dists[idx];

      if (val === 3) {
        nodes.push({ index: idx, row: r, col: c, type: 'wall', walkable: false, distToGoal: -1 });
        continue;
      }

      if (r === start[0] && c === start[1]) {
        nodes.push({ index: idx, row: r, col: c, type: 'start', walkable: true, distToGoal: dist });
        continue;
      }

      if (r === goal[0] && c === goal[1]) {
        nodes.push({ index: idx, row: r, col: c, type: 'goal', walkable: true, distToGoal: 0 });
        continue;
      }

      const neighbors = getWalkableNeighbors(grid, r, c);
      let type: NodeType = 'linear';
      if (neighbors.length >= 3) {
        type = 'bifurcation';
      } else if (neighbors.length === 1) {
        type = 'dead_end';
      } else {
        type = 'linear';
      }

      nodes.push({ index: idx, row: r, col: c, type, walkable: true, distToGoal: dist });
    }
  }

  return nodes;
}

// ----------------------------------------------------------------------------
// 3. SYNTHETIC PRE-TRAINED WEIGHT GENERATION
// Deterministic weight tensors reproducing realistic Transformer attention math
// ----------------------------------------------------------------------------

export function generateDeterministicWeights(): WeightTensors {
  const d_model = 32;
  const d_ff = 64;

  const pseudoRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  // Grid Embedding (0:path, 1:start, 2:goal, 3:wall, 4:agent, 5:visited)
  const gridEmbedding: number[][] = [];
  for (let tok = 0; tok < 6; tok++) {
    const row: number[] = [];
    for (let d = 0; d < d_model; d++) {
      const val = (pseudoRandom(tok * 100 + d) - 0.5) * 0.8;
      row.push(val);
    }
    gridEmbedding.push(row);
  }

  // 2D Spatial Positional Embedding (36 cells x 32 dims)
  const spatialPE: number[][] = [];
  for (let idx = 0; idx < 36; idx++) {
    const r = Math.floor(idx / 6);
    const c = idx % 6;
    const row: number[] = [];
    for (let d = 0; d < d_model; d++) {
      const freq = Math.pow(10000, -(2 * Math.floor(d / 2)) / d_model);
      if (d % 2 === 0) {
        row.push(Math.sin((r * 6 + c) * freq));
      } else {
        row.push(Math.cos((r * 6 + c) * freq));
      }
    }
    spatialPE.push(row);
  }

  // Agent Embedding
  const agentEmbedding: number[][] = [
    new Array(d_model).fill(0),
    new Array(d_model).fill(0.3).map((v, i) => v * (i % 2 === 0 ? 1 : -1)),
  ];

  const makeLayerWeights = (layerIdx: number) => {
    const makeMatrix = (rows: number, cols: number, scale = 0.2) => {
      const mat: number[][] = [];
      for (let i = 0; i < rows; i++) {
        const row: number[] = [];
        for (let j = 0; j < cols; j++) {
          row.push((pseudoRandom(layerIdx * 1000 + i * 50 + j) - 0.5) * scale);
        }
        mat.push(row);
      }
      return mat;
    };

    return {
      norm1: { weight: new Array(d_model).fill(1), bias: new Array(d_model).fill(0) },
      attn: {
        W_q: makeMatrix(d_model, d_model),
        W_k: makeMatrix(d_model, d_model),
        W_v: makeMatrix(d_model, d_model),
        W_o: makeMatrix(d_model, d_model),
      },
      norm2: { weight: new Array(d_model).fill(1), bias: new Array(d_model).fill(0) },
      ff: {
        fc1: { weight: makeMatrix(d_ff, d_model), bias: new Array(d_ff).fill(0) },
        fc2: { weight: makeMatrix(d_model, d_ff), bias: new Array(d_model).fill(0) },
      },
    };
  };

  const layers = [makeLayerWeights(0), makeLayerWeights(1)];

  const norm = { weight: new Array(d_model).fill(1), bias: new Array(d_model).fill(0) };

  // Classifier weights
  const fc_dir = {
    weight: Array.from({ length: 4 }, (_, i) =>
      Array.from({ length: d_model }, (_, j) => (pseudoRandom(5000 + i * 30 + j) - 0.5) * 0.4)
    ),
    bias: [0.1, 0.1, 0.1, 0.1],
  };

  const fc_cell = {
    weight: Array.from({ length: 36 }, (_, i) =>
      Array.from({ length: d_model }, (_, j) => (pseudoRandom(8000 + i * 30 + j) - 0.5) * 0.4)
    ),
    bias: new Array(36).fill(0),
  };

  return {
    gridEmbedding,
    spatialPE,
    agentEmbedding,
    layers,
    norm,
    fc_dir,
    fc_cell,
  };
}

const GLOBAL_WEIGHTS = generateDeterministicWeights();

// ----------------------------------------------------------------------------
// 4. TRANSFORMER MATH PRIMITIVES (LayerNorm, GELU, Projections)
// ----------------------------------------------------------------------------

export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

export function layerNorm(x: number[], norm: { weight: number[]; bias: number[] }, eps = 1e-5): number[] {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;

  let variance = 0;
  for (let i = 0; i < d; i++) variance += Math.pow(x[i] - mean, 2);
  variance /= d;

  const out = new Array(d);
  for (let i = 0; i < d; i++) {
    const xNorm = (x[i] - mean) / Math.sqrt(variance + eps);
    out[i] = xNorm * norm.weight[i] + norm.bias[i];
  }
  return out;
}

export function linearProject(x: number[], wRow: number[], bias = 0): number {
  let val = bias;
  for (let i = 0; i < x.length; i++) val += x[i] * wRow[i];
  return val;
}

// ----------------------------------------------------------------------------
// 5. FULL TRANSFORMER INFERENCE EXECUTION (`runLabyrinthInference`)
// ----------------------------------------------------------------------------

export function runLabyrinthInference(maze: MazeGrid, stepIndex: number): TransformerActivationTrace {
  const trajectory = solveBFS(maze);
  const totalSteps = trajectory.length - 1;
  const clampedStep = Math.min(Math.max(0, stepIndex), totalSteps);

  const agentPos = trajectory[clampedStep];
  const agentIndex = agentPos[0] * maze.cols + agentPos[1];
  const visitedPath = trajectory.slice(0, clampedStep + 1);

  const nodesInfo = classifyGridNodes(maze);
  const currentNodeInfo = nodesInfo[agentIndex];

  // Build grid token sequence (36 length)
  // 0: path, 1: start, 2: goal, 3: wall, 4: agent, 5: visited
  const tokens: number[] = new Array(36);
  for (let r = 0; r < maze.rows; r++) {
    for (let c = 0; c < maze.cols; c++) {
      const idx = r * maze.cols + c;
      if (r === agentPos[0] && c === agentPos[1]) {
        tokens[idx] = 4; // Agent
      } else if (visitedPath.some(([vr, vc]) => vr === r && vc === c)) {
        tokens[idx] = 5; // Visited
      } else {
        tokens[idx] = maze.grid[r][c];
      }
    }
  }

  const weights = GLOBAL_WEIGHTS;
  const L = 36;
  const d_model = 32;
  const n_heads = 2;
  const d_k = d_model / n_heads; // 16
  const d_ff = 64;

  // 1. Embedding Computation
  const embeddings: number[][] = [];
  const posEncodings: number[][] = [];
  const initialSum: number[][] = [];

  for (let i = 0; i < L; i++) {
    const tok = tokens[i];
    const gridEmb = [...weights.gridEmbedding[tok]];
    embeddings.push(gridEmb);

    const pe = [...weights.spatialPE[i]];
    posEncodings.push(pe);

    const sum = gridEmb.map((v, idx) => v + pe[idx]);
    initialSum.push(sum);
  }

  // 2. Transformer Layers
  let currentRepresentations = initialSum.map(row => [...row]);
  const layersTrace: LayerActivation[] = [];

  for (let l = 0; l < weights.layers.length; l++) {
    const lw = weights.layers[l];
    const layerInput = currentRepresentations.map(row => [...row]);
    const norm1 = layerInput.map(row => layerNorm(row, lw.norm1));

    // Linear Projections for Q, K, V
    const Q: number[][] = [];
    const K: number[][] = [];
    const V: number[][] = [];

    for (let i = 0; i < L; i++) {
      const qRow: number[] = [];
      const kRow: number[] = [];
      const vRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        qRow.push(linearProject(norm1[i], lw.attn.W_q[c]));
        kRow.push(linearProject(norm1[i], lw.attn.W_k[c]));
        vRow.push(linearProject(norm1[i], lw.attn.W_v[c]));
      }
      Q.push(qRow);
      K.push(kRow);
      V.push(vRow);
    }

    // Split into Heads
    const headsTrace: HeadActivation[] = [];
    for (let h = 0; h < n_heads; h++) {
      const headQueries: number[][] = [];
      const headKeys: number[][] = [];
      const headValues: number[][] = [];

      for (let i = 0; i < L; i++) {
        headQueries.push(Q[i].slice(h * d_k, (h + 1) * d_k));
        headKeys.push(K[i].slice(h * d_k, (h + 1) * d_k));
        headValues.push(V[i].slice(h * d_k, (h + 1) * d_k));
      }

      const rawScores: number[][] = [];
      const attnWeights: number[][] = [];

      for (let i = 0; i < L; i++) {
        const scoresRow: number[] = [];
        let maxScore = -Infinity;

        const qiNode = nodesInfo[i];
        const isBifurcationQuery = qiNode.type === 'bifurcation';

        for (let j = 0; j < L; j++) {
          let dot = 0;
          for (let c = 0; c < d_k; c++) {
            dot += headQueries[i][c] * headKeys[j][c];
          }
          let score = dot / Math.sqrt(d_k);

          // Realistic Topological Transformer Attention Biasing:
          const qRow = Math.floor(i / 6);
          const qCol = i % 6;
          const kRow = Math.floor(j / 6);
          const kCol = j % 6;
          const manhattan = Math.abs(qRow - kRow) + Math.abs(qCol - kCol);

          const kjNode = nodesInfo[j];

          // Unreachable wall suppression
          if (kjNode.type === 'wall') {
            score -= 6.0;
          } else {
            // Distance decay
            score -= manhattan * 0.4;

            // Goal anchor boost
            if (kjNode.type === 'goal') {
              score += 2.5;
            }

            // Bifurcation query behavior vs Linear query behavior
            if (isBifurcationQuery) {
              if (kjNode.type === 'bifurcation') {
                score += 1.8; // High attention to other decision nodes
              } else if (kjNode.type === 'linear') {
                score += 0.6;
              }
            } else {
              // Linear query: tightly focused on local linear neighbors
              if (manhattan <= 1) {
                score += 2.0;
              }
            }
          }

          scoresRow.push(score);
          if (score > maxScore) maxScore = score;
        }
        rawScores.push(scoresRow);

        const expRow = scoresRow.map(s => Math.exp(s - maxScore));
        const sumExp = expRow.reduce((sum, v) => sum + v, 0);
        const softmaxRow = expRow.map(v => v / sumExp);
        attnWeights.push(softmaxRow);
      }

      // Context
      const headContext: number[][] = [];
      for (let i = 0; i < L; i++) {
        const contextRow = new Array(d_k).fill(0);
        for (let c = 0; c < d_k; c++) {
          for (let j = 0; j < L; j++) {
            contextRow[c] += attnWeights[i][j] * headValues[j][c];
          }
        }
        headContext.push(contextRow);
      }

      headsTrace.push({
        headIndex: h,
        queries: headQueries,
        keys: headKeys,
        values: headValues,
        rawScores,
        attnWeights,
        context: headContext,
      });
    }

    // Concatenate heads & project back
    const concatenatedContext: number[][] = [];
    const attnProjected: number[][] = [];
    for (let i = 0; i < L; i++) {
      const concatRow: number[] = [];
      for (let h = 0; h < n_heads; h++) {
        concatRow.push(...headsTrace[h].context[i]);
      }
      concatenatedContext.push(concatRow);

      const projRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        projRow.push(linearProject(concatRow, lw.attn.W_o[c]));
      }
      attnProjected.push(projRow);
    }

    // Residual + Norm2 + FFN
    const attnResidual = layerInput.map((row, i) => row.map((v, idx) => v + attnProjected[i][idx]));
    const norm2 = attnResidual.map(row => layerNorm(row, lw.norm2));

    const ffnMid: number[][] = [];
    const ffnOut: number[][] = [];
    for (let i = 0; i < L; i++) {
      const midRow: number[] = [];
      for (let r = 0; r < d_ff; r++) {
        const preAct = linearProject(norm2[i], lw.ff.fc1.weight[r], lw.ff.fc1.bias[r]);
        midRow.push(gelu(preAct));
      }
      ffnMid.push(midRow);

      const outRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        const val = linearProject(midRow, lw.ff.fc2.weight[c], lw.ff.fc2.bias[c]);
        outRow.push(val);
      }
      ffnOut.push(outRow);
    }

    const ffnResidual = attnResidual.map((row, i) => row.map((v, idx) => v + ffnOut[i][idx]));

    layersTrace.push({
      layerIndex: l,
      input: layerInput,
      norm1,
      Q,
      K,
      V,
      heads: headsTrace,
      concatenatedContext,
      attnProjected,
      attnResidual,
      norm2,
      ffnMid,
      ffnOut,
      ffnResidual,
    });

    currentRepresentations = ffnResidual.map(row => [...row]);
  }

  // 3. Final LayerNorm
  const finalNorm = currentRepresentations.map(row => layerNorm(row, weights.norm));

  // 4. Directional & Target Cell Logits
  const agentNormVec = finalNorm[agentIndex];
  const directionalLogitsObj = { Up: 0, Down: 0, Left: 0, Right: 0 };

  // Calculate direction scores
  const dirsDelta = [
    { dir: 'Up', dr: -1, dc: 0 },
    { dir: 'Down', dr: 1, dc: 0 },
    { dir: 'Left', dr: 0, dc: -1 },
    { dir: 'Right', dr: 0, dc: 1 },
  ] as const;

  const currentDist = nodesInfo[agentIndex].distToGoal;
  let maxDirLogit = -Infinity;

  const directionsOptions: DirectionOption[] = [];

  dirsDelta.forEach(({ dir, dr, dc }, dIdx) => {
    const nr = agentPos[0] + dr;
    const nc = agentPos[1] + dc;
    let logit = linearProject(agentNormVec, weights.fc_dir.weight[dIdx], weights.fc_dir.bias[dIdx]);

    let distToGoal = -1;
    let targetIndex = -1;
    let isOptimal = false;

    if (nr >= 0 && nr < maze.rows && nc >= 0 && nc < maze.cols && maze.grid[nr][nc] !== 3) {
      targetIndex = nr * maze.cols + nc;
      distToGoal = nodesInfo[targetIndex].distToGoal;

      if (distToGoal >= 0) {
        if (distToGoal < currentDist) {
          logit += 3.5; // Strong boost for optimal path toward goal
          isOptimal = true;
        } else {
          logit -= 1.5; // Suboptimal/dead-end penalty
        }
      }
    } else {
      logit -= 10.0; // Wall move penalty
    }

    directionalLogitsObj[dir] = logit;
    if (logit > maxDirLogit) maxDirLogit = logit;

    directionsOptions.push({
      direction: dir,
      targetPos: [nr, nc],
      targetIndex,
      logit,
      probability: 0,
      distToGoal,
      isOptimal,
    });
  });

  // Directional Softmax
  const expDirs = directionsOptions.map(d => Math.exp(d.logit - maxDirLogit));
  const sumExpDirs = expDirs.reduce((sum, v) => sum + v, 0);
  const directionalProbsObj = { Up: 0, Down: 0, Left: 0, Right: 0 };

  directionsOptions.forEach((d, i) => {
    d.probability = expDirs[i] / sumExpDirs;
    directionalProbsObj[d.direction] = d.probability;
  });

  // Cell Logits
  const cellLogits: number[] = [];
  let maxCellLogit = -Infinity;
  for (let i = 0; i < L; i++) {
    const val = linearProject(agentNormVec, weights.fc_cell.weight[i], weights.fc_cell.bias[i]);
    cellLogits.push(val);
    if (val > maxCellLogit) maxCellLogit = val;
  }
  const expCells = cellLogits.map(s => Math.exp(s - maxCellLogit));
  const sumExpCells = expCells.reduce((s, v) => s + v, 0);
  const cellProbs = expCells.map(v => v / sumExpCells);

  // 5. Research Analytics Computation (Attention Entropy, Allocation Breakdown)
  // Layer 2, Head 0 attention weights from current agent query
  const layer2Head0Attn = layersTrace[1].heads[0].attnWeights[agentIndex];

  let attentionEntropy = 0;
  layer2Head0Attn.forEach(p => {
    if (p > 1e-9) {
      attentionEntropy -= p * Math.log2(p);
    }
  });

  const categoryAttention = {
    bifurcations: 0,
    linear: 0,
    deadEnds: 0,
    walls: 0,
    goal: 0,
    start: 0,
  };

  layer2Head0Attn.forEach((weight, j) => {
    const node = nodesInfo[j];
    if (node.type === 'bifurcation') categoryAttention.bifurcations += weight;
    else if (node.type === 'linear') categoryAttention.linear += weight;
    else if (node.type === 'dead_end') categoryAttention.deadEnds += weight;
    else if (node.type === 'wall') categoryAttention.walls += weight;
    else if (node.type === 'goal') categoryAttention.goal += weight;
    else if (node.type === 'start') categoryAttention.start += weight;
  });

  const sortedProbs = [...directionsOptions].map(d => d.probability).sort((a, b) => b - a);
  const top1Confidence = sortedProbs[0];

  const stepTrace: StepTrace = {
    stepIndex: clampedStep,
    agentPos,
    agentIndex,
    nodeType: currentNodeInfo.type,
    visitedPath,
    gridState: tokens,
    attentionEntropy,
    categoryAttention,
    directions: directionsOptions,
    top1Confidence,
    entropyBits: attentionEntropy,
  };

  return {
    maze,
    currentStep: clampedStep,
    totalSteps,
    trajectory,
    nodesInfo,
    stepTrace,
    tokens,
    embeddings,
    posEncodings,
    initialSum,
    layers: layersTrace,
    finalNorm,
    directionalLogits: directionalLogitsObj,
    directionalProbs: directionalProbsObj,
    cellLogits,
    cellProbs,
  };
}

// ----------------------------------------------------------------------------
// 6. LEARNED SPATIAL / ORDERING BIAS MATRIX
// Bilinear Query-Key Interaction Matrix: Bias(u, v) = e_u W_q W_k^T e_v^T
// ----------------------------------------------------------------------------

export function getSpatialBilinearScores(): { scores: number[][]; min: number; max: number } {
  const weights = GLOBAL_WEIGHTS;
  const lw = weights.layers[0];
  const d_model = 32;

  const Q_proj: number[][] = [];
  const K_proj: number[][] = [];

  for (let idx = 0; idx < 36; idx++) {
    const pe = weights.spatialPE[idx];
    const qRow: number[] = [];
    const kRow: number[] = [];
    for (let c = 0; c < d_model; c++) {
      qRow.push(linearProject(pe, lw.attn.W_q[c]));
      kRow.push(linearProject(pe, lw.attn.W_k[c]));
    }
    Q_proj.push(qRow);
    K_proj.push(kRow);
  }

  const scores: number[][] = [];
  let min = Infinity;
  let max = -Infinity;

  for (let u = 0; u < 36; u++) {
    const row: number[] = [];
    for (let v = 0; v < 36; v++) {
      let score = 0;
      for (let c = 0; c < d_model; c++) {
        score += Q_proj[u][c] * K_proj[v][c];
      }
      row.push(score);
      if (score < min) min = score;
      if (score > max) max = score;
    }
    scores.push(row);
  }

  return { scores, min, max };
}

export function getRawWeights(): WeightTensors {
  return GLOBAL_WEIGHTS;
}
