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

    const { maxTokensPerPrompt } = this.config.getContextManagementConfig().strategies.historySquashing;
    // We estimate 4 chars per token for truncation logic
    const limitChars = maxTokensPerPrompt * 4;
    
    // We track how many tokens we still need to cut. If we hit 0, we can stop early!
    let currentDeficit = state.deficitTokens;
    const newEpisodes = [...episodes];

    for (let i = 0; i < newEpisodes.length; i++) {
      if (currentDeficit <= 0) break; // Deficit solved!
      if (state.protectedEpisodeIds.has(newEpisodes[i]!.id)) continue;
      
      const ep = newEpisodes[i]!;

      // 1. Squash User Prompts
      if (ep.trigger.type === 'USER_PROMPT') {
        for (let j = 0; j < ep.trigger.semanticParts.length; j++) {
           const part = ep.trigger.semanticParts[j];
           if (part.type !== 'text') continue;
           
           const text = part.text;
           const originalLength = text.length;
           
           if (originalLength > limitChars) {
             const newText = truncateProportionally(
               text,
               limitChars,
               `\n\n[... OMITTED ${originalLength - limitChars} chars ...]\n\n`,
             );
             if (newText !== text) {
               const newTokens = Math.floor(newText.length / 4);
               const oldTokens = Math.floor(originalLength / 4);
               const tokensSaved = oldTokens - newTokens;
               
               part.presentation = { text: newText, tokens: newTokens };
               ep.trigger.metadata.transformations.push({
                 processorName: 'HistorySquashing',
                 action: 'TRUNCATED',
                 timestamp: Date.now()
               });
               
               currentDeficit -= tokensSaved;
             }
           }
        }
      }
      
      // 2. Squash Model Thoughts
      for (const step of ep.steps) {
        if (currentDeficit <= 0) break;
        if (step.type === 'AGENT_THOUGHT') {
           const text = step.text;
           const originalLength = text.length;
           
           if (originalLength > limitChars) {
             const newText = truncateProportionally(
               text,
               limitChars,
               `\n\n[... OMITTED ${originalLength - limitChars} chars ...]\n\n`,
             );
             if (newText !== text) {
               const newTokens = Math.floor(newText.length / 4);
               const oldTokens = Math.floor(originalLength / 4);
               const tokensSaved = oldTokens - newTokens;
               
               step.presentation = { text: newText, tokens: newTokens };
               step.metadata.transformations.push({
                 processorName: 'HistorySquashing',
                 action: 'TRUNCATED',
                 timestamp: Date.now()
               });
               
               currentDeficit -= tokensSaved;
             }
           }
        }
      }
    }

    return newEpisodes;
  }
}
