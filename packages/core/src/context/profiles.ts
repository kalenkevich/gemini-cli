/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { ContextManagementConfig } from './types.js';

export const PERFORMANCE_CONTEXT_PROFILE: ContextManagementConfig = {
  enabled: true,
  budget: {
    maxTokens: 150_000,
    retainedTokens: 80_000,
    protectedEpisodes: 1,
    protectSystemEpisode: true,
  },
  strategies: {
    historySquashing: { maxTokensPerPrompt: 12000 },
    toolMasking: { minObservationTokens: 10000 },
  },
};
