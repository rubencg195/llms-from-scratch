import { lazy, type ComponentType, type LazyExoticComponent } from "react";
import type { ModuleProps } from "./types";
import { moduleKey } from "@/data/curriculum";

type Mod = LazyExoticComponent<ComponentType<ModuleProps>>;

/**
 * Maps "phaseId:moduleId" -> lazy module component. Only implemented modules
 * are listed; anything missing falls back to ComingSoon, so the app always
 * builds and runs while modules are added incrementally.
 */
export const REGISTRY: Record<string, Mod> = {
  // Phase 0
  [moduleKey(0, "tensor-explorer")]: lazy(() => import("./phase0/TensorExplorer")),
  [moduleKey(0, "dot-product")]: lazy(() => import("./phase0/DotProduct")),
  [moduleKey(0, "micrograd")]: lazy(() => import("./phase0/MicrogradChain")),
  [moduleKey(0, "gradient-descent")]: lazy(() => import("./phase0/GradientDescent")),
  [moduleKey(0, "line-fit")]: lazy(() => import("./phase0/LineFit")),

  // Phase 1
  [moduleKey(1, "tokenizer")]: lazy(() => import("./phase1/Tokenizer")),
  [moduleKey(1, "bigram-lm")]: lazy(() => import("./phase1/BigramLM")),
  [moduleKey(1, "embeddings-3d")]: lazy(() => import("./phase1/Embeddings3D")),
  [moduleKey(1, "rope")]: lazy(() => import("./phase1/Rope")),
  [moduleKey(1, "attention")]: lazy(() => import("./phase1/Attention")),

  // Phase 2
  [moduleKey(2, "chat-template")]: lazy(() => import("./phase2/ChatTemplate")),
  [moduleKey(2, "masked-loss")]: lazy(() => import("./phase2/MaskedLoss")),
  [moduleKey(2, "tool-call")]: lazy(() => import("./phase2/ToolCall")),
  [moduleKey(2, "preference-pairs")]: lazy(() => import("./phase2/PreferencePairs")),

  // Phase 3
  [moduleKey(3, "quantize-grid")]: lazy(() => import("./phase3/QuantizeGrid")),
  [moduleKey(3, "ste")]: lazy(() => import("./phase3/STE")),

  // Phase 4
  [moduleKey(4, "router")]: lazy(() => import("./phase4/Router")),
  [moduleKey(4, "load-balance")]: lazy(() => import("./phase4/LoadBalance")),

  // Phase 5
  [moduleKey(5, "kv-cache")]: lazy(() => import("./phase5/KvCache")),
  [moduleKey(5, "polarquant")]: lazy(() => import("./phase5/PolarQuant")),

  // Phase 6
  [moduleKey(6, "patchify")]: lazy(() => import("./phase6/Patchify")),
  [moduleKey(6, "modality-mix")]: lazy(() => import("./phase6/ModalityMix")),

  // Phase 7
  [moduleKey(7, "audio-tokens")]: lazy(() => import("./phase7/AudioTokens")),
  [moduleKey(7, "barge-in")]: lazy(() => import("./phase7/BargeIn")),

  // Phase 8
  [moduleKey(8, "surprise")]: lazy(() => import("./phase8/Surprise")),
  [moduleKey(8, "neural-memory")]: lazy(() => import("./phase8/NeuralMemory")),
  [moduleKey(8, "memory-vram")]: lazy(() => import("./phase8/MemoryVram")),
};

export const Fallback = lazy(() => import("./ComingSoon"));

export function getModule(phaseId: number, moduleId: string): Mod {
  return REGISTRY[moduleKey(phaseId, moduleId)] ?? Fallback;
}
