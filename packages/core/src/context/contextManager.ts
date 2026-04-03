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

    const protectedEpisodes = 1;
    const frontBufferStartIndex = Math.max(
      0,
      currentEpisodes.length - protectedEpisodes,
    );
    const backBufferEndIndex = Math.max(0, frontBufferStartIndex - 1);

    for (const processor of this.processors) {
      const state: ContextAccountingState = {
        currentTokens,
        maxTokens,
        retainedTokens,
        frontBufferStartIndex,
        backBufferEndIndex,
        isBudgetSatisfied: currentTokens <= retainedTokens,
      };

      if (state.isBudgetSatisfied) {
        debugLogger.log('Context Manager satisfied budget. Stopping early.');
        break;
      }

      debugLogger.log(`Running ContextProcessor: ${processor.name}`);
      const result = await processor.process(currentEpisodes, state);

      currentEpisodes = result.episodes;
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
