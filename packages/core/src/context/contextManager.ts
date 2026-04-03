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

  setProcessors(processors: ContextProcessor[]) {
    this.processors = processors;
  }

  async processHistory(history: Content[]): Promise<Content[]> {
    if (!this.config.isContextManagementEnabled()) {
      return history;
    }

    const mngConfig = this.config.getContextManagementConfig();
    const maxTokens = mngConfig.historyWindow.maxTokens;
    const retainedTokens = mngConfig.historyWindow.retainedTokens;

    let currentEpisodes = IrMapper.toIr(history);
    let currentTokens = this.calculateIrTokens(currentEpisodes);

    if (currentTokens <= maxTokens) {
      return history;
    }

    debugLogger.log(
      `Context Manager triggered: Context window at ${currentTokens} tokens (limit: ${maxTokens}, target: ${retainedTokens}).`,
    );

        const protectedEpisodeIds = new Set<string>();
    // Protect the very first episode (often contains the initial architectural ask/system prompt)
    if (currentEpisodes.length > 0) {
      protectedEpisodeIds.add(currentEpisodes[0]!.id);
    }
    // Protect the most recent episode (current working context)
    if (currentEpisodes.length > 1) {
      protectedEpisodeIds.add(currentEpisodes[currentEpisodes.length - 1]!.id);
    }

    for (const processor of this.processors) {
      const state: ContextAccountingState = {
        currentTokens,
        maxTokens,
        retainedTokens,
        deficitTokens: Math.max(0, currentTokens - retainedTokens),
        protectedEpisodeIds,
        isBudgetSatisfied: currentTokens <= retainedTokens,
      };

      if (state.isBudgetSatisfied) {
        debugLogger.log('Context Manager satisfied budget. Stopping early.');
        break;
      }

      debugLogger.log(`Running ContextProcessor: ${processor.name}`);
      currentEpisodes = await processor.process(currentEpisodes, state);
      const newTokens = this.calculateIrTokens(currentEpisodes);

      if (newTokens < currentTokens) {
        debugLogger.log(
          `Processor [${processor.name}] saved approx ${currentTokens - newTokens} tokens. New estimate: ${newTokens}.`,
        );
        currentTokens = newTokens;
      }
    }

    const finalTokens = this.calculateIrTokens(currentEpisodes);
    debugLogger.log(
      `Context Manager finished. Final actual token count: ${finalTokens}.`,
    );

    return IrMapper.fromIr(currentEpisodes);
  }

  private calculateIrTokens(episodes: Episode[]): number {
    let tokens = 0;
    for (const ep of episodes) {
      if (ep.trigger) tokens += ep.trigger.metadata.currentTokens;
      for (const step of ep.steps) {
        tokens += step.metadata.currentTokens;
      }
      if (ep.yield) tokens += ep.yield.metadata.currentTokens;
    }
    return tokens;
  }
}
