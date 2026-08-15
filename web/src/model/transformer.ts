import modelWeights from '../model_weights.json';

// Interfaces for weight shapes
export interface LayerNormWeights {
  weight: number[];
  bias: number[];
}

export interface AttnWeights {
  W_q: number[][];
  W_k: number[][];
  W_v: number[][];
  W_o: number[][];
}

export interface FFNWeights {
  fc1: {
    weight: number[][];
    bias: number[];
  };
  fc2: {
    weight: number[][];
    bias: number[];
  };
}

export interface LayerWeights {
  norm1: LayerNormWeights;
  attn: AttnWeights;
  norm2: LayerNormWeights;
  ff: FFNWeights;
}

export interface TransformerWeights {
  embedding: number[][];
  pe: number[][];
  layers: LayerWeights[];
  norm: LayerNormWeights;
  fc_out: {
    weight: number[][];
    bias: number[];
  };
}

// Ensure loaded weights match our interface
const weights = modelWeights as unknown as TransformerWeights;

// GELU activation function matching PyTorch's nn.GELU() approximation
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
}

// 1D Vector LayerNorm
export function layerNorm(x: number[], lnWeights: LayerNormWeights, eps: number = 1e-5): number[] {
  const d = x.length;
  // Mean
  let mean = 0;
  for (let i = 0; i < d; i++) {
    mean += x[i];
  }
  mean /= d;

  // Variance
  let variance = 0;
  for (let i = 0; i < d; i++) {
    variance += Math.pow(x[i] - mean, 2);
  }
  variance /= d;

  // Normalize, scale, and shift
  const output = new Array(d);
  for (let i = 0; i < d; i++) {
    const xNorm = (x[i] - mean) / Math.sqrt(variance + eps);
    output[i] = xNorm * lnWeights.weight[i] + lnWeights.bias[i];
  }
  return output;
}

// Matrix multiplication helper: inputs of shape [L, d_in] multiplied by W^T of shape [d_in, d_out]
// W is of shape [d_out, d_in]
export function linearProject(x: number[], wRow: number[], bias: number = 0): number {
  let val = bias;
  for (let i = 0; i < x.length; i++) {
    val += x[i] * wRow[i];
  }
  return val;
}

export interface HeadActivation {
  headIndex: number;
  queries: number[][]; // [L, d_k]
  keys: number[][];    // [L, d_k]
  values: number[][];  // [L, d_k]
  rawScores: number[][]; // [L, L]
  attnWeights: number[][]; // [L, L] after softmax
  context: number[][];   // [L, d_k]
}

export interface LayerActivation {
  layerIndex: number;
  input: number[][];            // [L, d_model]
  norm1: number[][];            // [L, d_model]

  // All projections
  Q: number[][];                // [L, d_model]
  K: number[][];                // [L, d_model]
  V: number[][];                // [L, d_model]

  heads: HeadActivation[];
  concatenatedContext: number[][]; // [L, d_model]
  attnProjected: number[][];       // [L, d_model]
  attnResidual: number[][];        // [L, d_model]

  norm2: number[][];            // [L, d_model]
  ffnMid: number[][];           // [L, d_ff]
  ffnOut: number[][];           // [L, d_model]
  ffnResidual: number[][];      // [L, d_model]
}

export interface TransformerActivationTrace {
  inputTokens: number[];
  embeddings: number[][];      // [L, d_model]
  posEncodings: number[][];    // [L, d_model]
  initialSum: number[][];      // [L, d_model]

  layers: LayerActivation[];
  finalNorm: number[][];       // [L, d_model]
  logits: number[][];          // [L, vocab_size]
  predictions: number[];       // sorted output tokens
}

/**
 * Runs inference on a 5-length digit array using the trained weights.
 * Returns a complete activation trace containing all intermediate calculations.
 */
export function runInference(inputTokens: number[]): TransformerActivationTrace {
  const L = inputTokens.length; // should be 5
  const d_model = weights.embedding[0].length; // 32
  const n_heads = weights.layers[0].attn.W_q.length / (weights.embedding[0].length / 2) ? 2 : 2; // n_heads = 2
  const d_k = d_model / n_heads; // 16

  // 1. Embeddings
  const embeddings: number[][] = [];
  const posEncodings: number[][] = [];
  const initialSum: number[][] = [];

  for (let i = 0; i < L; i++) {
    const token = inputTokens[i];
    const emb = [...weights.embedding[token]];
    embeddings.push(emb);

    const pe = [...weights.pe[i]];
    posEncodings.push(pe);

    const sum = emb.map((v, idx) => v + pe[idx]);
    initialSum.push(sum);
  }

  // 2. Transformer Encoder Layers
  let currentRepresentations = initialSum.map(row => [...row]);
  const layersTrace: LayerActivation[] = [];

  for (let l = 0; l < weights.layers.length; l++) {
    const lw = weights.layers[l];
    const layerInput = currentRepresentations.map(row => [...row]);

    // Norm 1
    const norm1 = layerInput.map(row => layerNorm(row, lw.norm1));

    // Linear projections for Q, K, V
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

      // Attention scores and weights
      const rawScores: number[][] = [];
      const attnWeights: number[][] = [];

      for (let i = 0; i < L; i++) {
        const scoresRow: number[] = [];
        let maxScore = -Infinity;

        // Compute dot product scores
        for (let j = 0; j < L; j++) {
          let dot = 0;
          for (let c = 0; c < d_k; c++) {
            dot += headQueries[i][c] * headKeys[j][c];
          }
          const score = dot / Math.sqrt(d_k);
          scoresRow.push(score);
          if (score > maxScore) maxScore = score;
        }
        rawScores.push(scoresRow);

        // Softmax
        const expRow = scoresRow.map(s => Math.exp(s - maxScore));
        const sumExp = expRow.reduce((sum, val) => sum + val, 0);
        const softmaxRow = expRow.map(v => v / sumExp);
        attnWeights.push(softmaxRow);
      }

      // Context computation
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
        context: headContext
      });
    }

    // Concatenate heads
    const concatenatedContext: number[][] = [];
    for (let i = 0; i < L; i++) {
      const concatRow: number[] = [];
      for (let h = 0; h < n_heads; h++) {
        concatRow.push(...headsTrace[h].context[i]);
      }
      concatenatedContext.push(concatRow);
    }

    // Project back
    const attnProjected: number[][] = [];
    for (let i = 0; i < L; i++) {
      const projRow: number[] = [];
      for (let c = 0; c < d_model; c++) {
        projRow.push(linearProject(concatenatedContext[i], lw.attn.W_o[c]));
      }
      attnProjected.push(projRow);
    }

    // Attention Residual connection
    const attnResidual: number[][] = [];
    for (let i = 0; i < L; i++) {
      const resRow = layerInput[i].map((v, idx) => v + attnProjected[i][idx]);
      attnResidual.push(resRow);
    }

    // Norm 2
    const norm2 = attnResidual.map(row => layerNorm(row, lw.norm2));

    // FFN
    const ffnMid: number[][] = [];
    const ffnOut: number[][] = [];

    const d_ff = lw.ff.fc1.bias.length; // 64
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

    // FFN Residual connection
    const ffnResidual: number[][] = [];
    for (let i = 0; i < L; i++) {
      const resRow = attnResidual[i].map((v, idx) => v + ffnOut[i][idx]);
      ffnResidual.push(resRow);
    }

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
      ffnResidual
    });

    currentRepresentations = ffnResidual.map(row => [...row]);
  }

  // 3. Final LayerNorm
  const finalNorm = currentRepresentations.map(row => layerNorm(row, weights.norm));

  // 4. Logits and Predictions
  const logits: number[][] = [];
  const predictions: number[] = [];
  const vocab_size = weights.fc_out.bias.length;

  for (let i = 0; i < L; i++) {
    const logitsRow: number[] = [];
    let maxLogit = -Infinity;
    let predToken = 0;

    for (let v = 0; v < vocab_size; v++) {
      const val = linearProject(finalNorm[i], weights.fc_out.weight[v], weights.fc_out.bias[v]);
      logitsRow.push(val);
      if (val > maxLogit) {
        maxLogit = val;
        predToken = v;
      }
    }
    logits.push(logitsRow);
    predictions.push(predToken);
  }

  return {
    inputTokens,
    embeddings,
    posEncodings,
    initialSum,
    layers: layersTrace,
    finalNorm,
    logits,
    predictions
  };
}

/**
 * Computes the bilinear similarity bias score between query index u and key index v
 * Bias(u, v) = Embedding[u] * W_q * W_k^T * Embedding[v]^T
 * This lets us plot the learned magnitude bias.
 */
export function getMagnitudeBilinearScores(): number[][] {
  const vocab_size = weights.embedding.length;
  const d_model = weights.embedding[0].length;
  const lw = weights.layers[0]; // analyze layer 1 magnitude interaction

  const Q_proj: number[][] = [];
  const K_proj: number[][] = [];

  for (let v = 0; v < vocab_size; v++) {
    const emb = weights.embedding[v];
    const qRow: number[] = [];
    const kRow: number[] = [];
    for (let c = 0; c < d_model; c++) {
      qRow.push(linearProject(emb, lw.attn.W_q[c]));
      kRow.push(linearProject(emb, lw.attn.W_k[c]));
    }
    Q_proj.push(qRow);
    K_proj.push(kRow);
  }

  const bilinearScores: number[][] = [];
  for (let u = 0; u < vocab_size; u++) {
    const row: number[] = [];
    for (let v = 0; v < vocab_size; v++) {
      let score = 0;
      for (let c = 0; c < d_model; c++) {
        score += Q_proj[u][c] * K_proj[v][c];
      }
      row.push(score);
    }
    bilinearScores.push(row);
  }

  return bilinearScores;
}
