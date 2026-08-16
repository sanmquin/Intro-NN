import { Sparkles, Brain, Cpu, Database, Network, HelpCircle, Layers } from 'lucide-react';

export interface StepInfo {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  icon: JSX.Element;
  description: string;
  conceptTooltip: string;
  mathFormula: string;
}

export const STEPS: StepInfo[] = [
  {
    id: 'step_1_embed',
    number: 1,
    title: 'Input & Embeddings',
    subtitle: 'Token + Position Vector',
    icon: <Sparkles className="w-4 h-4 text-cyan-400" />,
    description: 'Each integer token (0-9) is mapped to a 32D dense embedding vector. Sinusoidal positional encodings are added to inform the network of sequence coordinates.',
    conceptTooltip: 'Transformers process tokens in parallel. Positional encodings allow the model to distinguish identical digits at different positions.',
    mathFormula: 'x_i = Embedding(token_i) + PositionalEncoding(i)',
  },
  {
    id: 'step_2_projections',
    number: 2,
    title: 'Linear Q/K/V Projections',
    subtitle: 'Sub-space Transformations',
    icon: <Cpu className="w-4 h-4 text-violet-400" />,
    description: 'The normalized vector x_i is projected through learned matrices W_q, W_k, and W_v into Query (q_i), Key (k_i), and Value (v_i) vectors.',
    conceptTooltip: 'Queries act as lookups ("What value am I searching for?"), Keys act as tags ("What value do I hold?"), and Values hold routing content.',
    mathFormula: 'q_i = x_i W_q,  k_i = x_i W_k,  v_i = x_i W_v',
  },
  {
    id: 'step_3_attention',
    number: 3,
    title: 'Dot-Product Attention',
    subtitle: 'Softmax Score Matrix',
    icon: <Brain className="w-4 h-4 text-indigo-400" />,
    description: 'The dot product q_i · k_j measures similarity between query position i and key position j. Softmax normalizes scores into attention probabilities.',
    conceptTooltip: 'Higher dot product means token i strongly attends to token j. Divided by √d_k to prevent extreme gradients.',
    mathFormula: 'Score(i, j) = (q_i · k_j) / √d_k,  A(i, j) = Softmax_j(Score(i, j))',
  },
  {
    id: 'step_4_ffn',
    number: 4,
    title: 'Value Context & FFN',
    subtitle: 'Non-Linear Feedforward',
    icon: <Layers className="w-4 h-4 text-amber-400" />,
    description: 'Weighted values are aggregated into context vectors c_i, projected back, and passed through a 2-layer GELU Feed-Forward Network with residual links.',
    conceptTooltip: 'The FFN performs non-linear feature extraction on token representations after contextual information is mixed by attention.',
    mathFormula: 'c_i = ∑_j A(i, j) v_j,  FFN(x) = GELU(x W_1 + b_1) W_2 + b_2',
  },
  {
    id: 'step_5_logits',
    number: 5,
    title: 'Output Logits & Sorting',
    subtitle: 'Classifier Vocabulary Projection',
    icon: <Database className="w-4 h-4 text-emerald-400" />,
    description: 'Final normalized representations are mapped to 10D vocabulary logits. The argmax token prediction gives the sorted sequence at each position.',
    conceptTooltip: 'The output layer checks which digit token best fits each sorted position based on accumulated attention context.',
    mathFormula: 'Logits_i = LayerNorm(x_final) W_out + b_out,  Prediction_i = argmax(Logits_i)',
  },
  {
    id: 'step_6_weights',
    number: 6,
    title: 'Learned Weights & Geometry',
    subtitle: 'Static Bias & PCA Space',
    icon: <Network className="w-4 h-4 text-cyan-400" />,
    description: 'Analyzes trained model properties independently of input sequence: the Query-Key magnitude bias map e_u W_q W_k^T e_v^T and 2D PCA embedding geometry.',
    conceptTooltip: 'This reveals the static mathematical structure learned during training: integer tokens are ordered along a continuous numerical scale.',
    mathFormula: 'Bias(u, v) = e_u W_q W_k^T e_v^T',
  },
];

interface EducationalStepperProps {
  currentStepId: string;
  onSelectStep: (stepId: string) => void;
}

export default function EducationalStepper({ currentStepId, onSelectStep }: EducationalStepperProps) {
  const activeStep = STEPS.find(s => s.id === currentStepId) || STEPS[0];

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
        <div className="flex items-center gap-2">
          <span className="p-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 font-bold text-xs font-mono">
            TUTORIAL
          </span>
          <h2 className="text-sm font-bold text-zinc-100">
            Step-by-Step Transformer Execution Guide
          </h2>
        </div>

        <span className="text-xs font-mono text-zinc-500">
          Step {activeStep.number} of {STEPS.length}
        </span>
      </div>

      {/* Stepper Buttons Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        {STEPS.map((step) => {
          const isActive = step.id === currentStepId;
          return (
            <button
              key={step.id}
              onClick={() => onSelectStep(step.id)}
              className={`p-2.5 rounded-xl border transition-all text-left flex flex-col justify-between ${
                isActive
                  ? 'bg-indigo-500/15 border-indigo-500 ring-1 ring-indigo-500/30 text-indigo-200 shadow-[0_0_12px_rgba(99,102,241,0.25)]'
                  : 'bg-zinc-950/60 border-zinc-850 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.2 rounded ${
                  isActive ? 'bg-indigo-500 text-white' : 'bg-zinc-900 text-zinc-500'
                }`}>
                  0{step.number}
                </span>
                {step.icon}
              </div>
              <span className="text-xs font-bold tracking-tight truncate">{step.title}</span>
              <span className="text-[9px] text-zinc-500 truncate">{step.subtitle}</span>
            </button>
          );
        })}
      </div>

      {/* Active Step Details & Educational Tooltip Box */}
      <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-850 space-y-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-900 pb-2">
          <div className="flex items-center gap-2">
            {activeStep.icon}
            <h3 className="text-xs font-bold text-zinc-100 uppercase tracking-wider">
              {activeStep.number}. {activeStep.title} — {activeStep.subtitle}
            </h3>
          </div>
          <span className="text-[10px] font-mono text-indigo-300 bg-zinc-900 px-2.5 py-1 rounded border border-zinc-800">
            {activeStep.mathFormula}
          </span>
        </div>

        <p className="text-xs text-zinc-300 leading-relaxed pt-1">
          {activeStep.description}
        </p>

        <div className="mt-2 p-2.5 bg-indigo-950/20 border border-indigo-500/30 rounded-lg text-[11px] text-indigo-200 flex items-start gap-2">
          <HelpCircle className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-indigo-300">Intuition & Explanation: </span>
            {activeStep.conceptTooltip}
          </div>
        </div>
      </div>
    </div>
  );
}
