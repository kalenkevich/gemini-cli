/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Content } from '@google/genai';
import type { Config } from '../config/config.js';
import type { GeminiClient } from '../core/client.js';
import type { ContextAccountingState, ContextProcessor } from './pipeline.js';
import { debugLogger } from '../utils/debugLogger.js';
import { IrMapper } from './ir/mapper.js';
import type { Episode } from './ir/types.js';

export class ContextManager {
  private config: Config;

  private processors: ContextProcessor[] = [];

  constructor(config: Config, _client: GeminiClient) {
    this.config = config;
  }

  /**
   * Inject the ordered pipeline of processors.
   * Typical order: [Masking (fast), Squashing (fast), Semantic Compression (slow)]
   */
  setProcessors(processors: ContextProcessor[]) {
    this.processors = processors;
  }

  /**
   * Orchestrates the history degradation through the registered pipeline.
   */
  async processHistory(history: Content[]): Promise<Content[]> {
    if (!this.config.isContextManagementEnabled()) {
      return history;
    }

    const mngConfig = this.config.getContextManagementConfig();
    const maxTokens = mngConfig.historyWindow.maxTokens;
    const retainedTokens = mngConfig.historyWindow.retainedTokens;

    let currentEpisodes = IrMapper.toIr(history);
    let currentTokens = this.estimateTokens(currentEpisodes);
    
    if (currentTokens <= maxTokens) {
      return history; // Well under the high-water mark, do nothing.
    }

    debugLogger.log(
      `Context Manager triggered: Context window at ${currentTokens} tokens (limit: ${maxTokens}, target: ${retainedTokens}).`,
    );

    for (const processor of this.processors) {
      // Calculate buffer zones dynamically in terms of Episodes
      const protectedEpisodes = mngConfig.tools?.outputMasking?.protectLatestTurn
        ? 1
        : 0;
      const frontBufferStartIndex = Math.max(
        0,
        currentEpisodes.length - protectedEpisodes,
      );
      // Back buffer is everything else.
      const backBufferEndIndex = Math.max(0, frontBufferStartIndex - 1);

      const state: ContextAccountingState = {
        currentTokens,
        maxTokens,
        retainedTokens,
        frontBufferStartIndex,
        backBufferEndIndex,
        isBudgetSatisfied: currentTokens <= retainedTokens,
      };

      if (state.isBudgetSatisfied) {
        debugLogger.log(`Context Manager satisfied budget. Stopping early.`);
        break;
      }

      debugLogger.log(`Running ContextProcessor: ${processor.name}`);
      const result = await processor.process(currentEpisodes, state);

      currentEpisodes = result.episodes;

      if (result.savedTokens > 0) {
        currentTokens = Math.max(0, currentTokens - result.savedTokens);
        debugLogger.log(
          `Processor [${processor.name}] saved approx ${result.savedTokens} tokens. New estimate: ${currentTokens}.`,
        );
      }
    }

    // Final sanity check
    const finalTokens = this.estimateTokens(currentEpisodes);
    debugLogger.log(
      `Context Manager finished. Final actual token count: ${finalTokens}.`,
    );

    return IrMapper.fromIr(currentEpisodes);
  }

  private estimateTokens(episodes: Episode[]): number {
    let chars = 0;
    for (const ep of episodes) {
      if (ep.trigger.type === 'USER_PROMPT') chars += ep.trigger.text.length;
      for (const step of ep.steps) {
        if (step.type === 'AGENT_THOUGHT') chars += step.text.length;
        if (step.type === 'TOOL_EXECUTION') {
          const obs = step.observation;
          if (typeof obs === 'object' && obs && typeof obs['output'] === 'string') {
            chars += obs['output'].length;
          }
        }
      }
      if (ep.yield) chars += ep.yield.text.length;
    }
    return Math.floor(chars / 4);
  }
}
