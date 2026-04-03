/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ContextManagementConfig {
  enabled: boolean;

  /** The global orchestration budget */
  budget: {
    /** The absolute maximum tokens before the context manager triggers */
    maxTokens: number;
    /** The target token count to reduce to when triggered */
    retainedTokens: number;
    /** The number of recent Episodes to always protect from degradation (default: 1) */
    protectedEpisodes: number;
    /** Should we protect Episode 0 (the System Prompt/Architectural Initialization)? */
    protectSystemEpisode: boolean;
  };

  /** Specific hyperparameters for degrading the context when over budget */
  strategies: {
    historySquashing: {
      /** The maximum allowable tokens for an old user prompt before it gets proportionally truncated */
      maxTokensPerPrompt: number;
    };
    toolMasking: {
      /** Only mask tool observations that are larger than this threshold */
      minObservationTokens: number;
    };
  };
}
