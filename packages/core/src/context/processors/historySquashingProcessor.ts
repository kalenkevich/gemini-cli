/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Episode } from '../ir/types.js';
import type { ContextAccountingState, ContextProcessor } from '../pipeline.js';
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
  ): Promise<Episode[]> {
    if (state.isBudgetSatisfied) {
      return episodes;
    }

    const { retainedMaxTokens } = this.config.getContextManagementConfig().messageLimits;
    const limit = retainedMaxTokens;
    const newEpisodes = [...episodes];

    for (let i = 0; i < newEpisodes.length; i++) {
      if (state.protectedEpisodeIds.has(newEpisodes[i]!.id)) continue;
      
      const ep = newEpisodes[i]!;

      if (ep.trigger.type === 'USER_PROMPT') {
        for (let j = 0; j < ep.trigger.semanticParts.length; j++) {
           const part = ep.trigger.semanticParts[j];
           if (part.type !== 'text') continue;
           
           const text = part.text;
           const originalLength = text.length;
           if (originalLength > limit * 4) {
             const newText = truncateProportionally(
               text,
               limit * 4,
               `\n\n[... OMITTED ${originalLength - limit * 4} chars ...]\n\n`,
             );
             if (newText !== text) {
               part.presentation = { text: newText, tokens: Math.floor(newText.length / 4) };
               ep.trigger.metadata.transformations.push({
                 processorName: 'HistorySquashing',
                 action: 'TRUNCATED',
                 timestamp: Date.now()
               });
             }
           }
        }
      }
    }

    return newEpisodes;
  }
}
