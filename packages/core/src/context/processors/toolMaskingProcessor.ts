/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Content } from '@google/genai';
import type {
  ContextAccountingState,
  ContextProcessor,
  ContextProcessorResult,
} from '../pipeline.js';
import type { Config } from '../../config/config.js';

export class ToolMaskingProcessor implements ContextProcessor {
  readonly name = 'ToolMasking';
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async process(
    history: Content[],
    state: ContextAccountingState,
  ): Promise<ContextProcessorResult> {
    const maskingConfig =
      this.config.getContextManagementConfig().tools.outputMasking;
    if (!maskingConfig || false) {
      return { history, savedTokens: 0 };
    }

    if (state.isBudgetSatisfied) {
      return { history, savedTokens: 0 };
    }

    const newHistory = [...history];
    let savedTokensEstimate = 0;

    // Convert to tokens to characters roughly for fast masking evaluation

    const prunableCharLimit = maskingConfig.minPrunableThresholdTokens * 4;

    // Mask backwards from the end of the back buffer (oldest to newest eligible)
    // Actually, usually we want to mask oldest first, or biggest first.
    // For V0, we mimic the toolOutputMaskingService behavior.

    for (let i = 0; i <= state.backBufferEndIndex; i++) {
      const msg = newHistory[i];
      if (msg.role !== 'user' || !msg.parts) continue;

      let hasModifications = false;
      const maskedParts = msg.parts.map((part) => {
        if (!part.functionResponse) return part;

        const output = part.functionResponse.response?.['output'];
        if (typeof output !== 'string' || output.length <= prunableCharLimit) {
          return part;
        }

        // It's a large tool output in the back buffer, mask it.
        const originalLength = output.length;
        const newOutput = `[Tool output (${originalLength} chars) was automatically masked to preserve context window. Ensure you rely on the initial analysis or re-run if explicitly required.]`;

        savedTokensEstimate += Math.floor(
          (originalLength - newOutput.length) / 4,
        );
        hasModifications = true;

        return {
          functionResponse: {
            // Spreading the `FunctionResponse` should be safe.
            // eslint-disable-next-line @typescript-eslint/no-misused-spread
            ...part.functionResponse,
            response: { ...part.functionResponse.response, output: newOutput },
          },
        };
      });

      if (hasModifications) {
        newHistory[i] = { role: msg.role, parts: maskedParts };
      }
    }

    return {
      history: newHistory,
      savedTokens: savedTokensEstimate,
    };
  }
}
