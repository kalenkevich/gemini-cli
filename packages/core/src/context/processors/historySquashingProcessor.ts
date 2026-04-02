/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Episode } from '../ir/types.js';
import type {
  ContextAccountingState,
  ContextProcessor,
  ContextProcessorResult,
} from '../pipeline.js';
import type { Config } from '../../config/config.js';
import { truncateProportionally } from '../truncation.js';

export class HistorySquashingProcessor implements ContextProcessor {
  readonly name = 'HistorySquashing';
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async process(
    episodes: Episode[],
    state: ContextAccountingState,
  ): Promise<ContextProcessorResult> {
        if (state.isBudgetSatisfied) {
      return { episodes, savedTokens: 0 };
    }

    const { normalMaxTokens, retainedMaxTokens, normalizationHeadRatio } =
      this.config.getContextManagementConfig().messageLimits;

    const limit = state.backBufferEndIndex >= 0 ? retainedMaxTokens : normalMaxTokens;
    const ratio = normalizationHeadRatio || 0.15;
    void ratio; // satisfy linter

    let savedTokensEstimate = 0;
    const newEpisodes = [...episodes];

    for (let i = 0; i <= state.backBufferEndIndex; i++) {
      const ep = newEpisodes[i];
      if (!ep) continue;

      if (ep.trigger.type === 'USER_PROMPT') {
        const text = ep.trigger.text;
        const originalLength = text.length;
        if (originalLength > limit * 4) {
          const truncated = truncateProportionally(text, limit * 4, `\n\n[... OMITTED ${originalLength - limit * 4} chars ...]\n\n`);
          if (truncated !== text) {
            ep.trigger.text = truncated;
            if (ep.trigger.parts) {
               ep.trigger.parts = [{ text: truncated }]; // override parts for mapping
            }
            savedTokensEstimate += Math.floor((originalLength - truncated.length) / 4);
            ep.trigger.metadata.transformations.push({
              processorName: 'HistorySquashing',
              action: 'TRUNCATED',
              timestamp: Date.now()
            });
          }
        }
      }
    }

    return {
      episodes: newEpisodes,
      savedTokens: savedTokensEstimate,
    };
  }
}
