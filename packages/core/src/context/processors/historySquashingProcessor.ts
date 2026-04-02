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
import { truncateString } from '../../utils/textUtils.js';

export class HistorySquashingProcessor implements ContextProcessor {
  readonly name = 'HistorySquashing';
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async process(
    history: Content[],
    state: ContextAccountingState,
  ): Promise<ContextProcessorResult> {
    if (state.isBudgetSatisfied) {
      return { history, savedTokens: 0 };
    }

    const { retainedMaxTokens, normalizationHeadRatio } =
      this.config.getContextManagementConfig().messageLimits;

    // Fallbacks if not perfectly configured
    const limit = retainedMaxTokens || 3000;
    const ratio = normalizationHeadRatio || 0.15;
    void ratio; // satisfy linter

    let savedTokensEstimate = 0;
    const newHistory = [...history];

    for (let i = 0; i <= state.backBufferEndIndex; i++) {
      const msg = newHistory[i];
      if (!msg.parts) continue;

      let hasModifications = false;
      const normalizedParts = msg.parts.map((part) => {
        if (part.text && part.text.length > limit * 4) {
          // Fast heuristic string-length check
          const originalLength = part.text.length;
          hasModifications = true;

          const newText = truncateString(
            part.text,
            limit * 4,
            `\n\n[... OMITTED ${originalLength - limit * 4} chars ...]\n\n`,
          );
          savedTokensEstimate += Math.floor(
            (originalLength - newText.length) / 4,
          );
          return { ...part, text: newText };
        }
        return part;
      });

      if (hasModifications) {
        newHistory[i] = { role: msg.role, parts: normalizedParts };
      }
    }

    return {
      history: newHistory,
      savedTokens: savedTokensEstimate,
    };
  }
}
