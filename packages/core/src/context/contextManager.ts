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

    // Check initial budget
    let currentTokens = this.estimateTokens(history);
    if (currentTokens <= maxTokens) {
      return history; // Well under the high-water mark, do nothing.
    }

    debugLogger.log(
      `Context Manager triggered: Context window at ${currentTokens} tokens (limit: ${maxTokens}, target: ${retainedTokens}).`,
    );

    let currentHistory = history;

    for (const processor of this.processors) {
      // Calculate buffer zones dynamically. For V0, we use heuristics.
      // E.g., Front buffer is the last N turns.
      const protectedTurns = mngConfig.tools?.outputMasking?.protectLatestTurn
        ? 2
        : 0;
      const frontBufferStartIndex = Math.max(
        0,
        currentHistory.length - protectedTurns,
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
      const result = await processor.process(currentHistory, state);

      currentHistory = result.history;

      if (result.savedTokens > 0) {
        currentTokens = Math.max(0, currentTokens - result.savedTokens);
        debugLogger.log(
          `Processor [${processor.name}] saved approx ${result.savedTokens} tokens. New estimate: ${currentTokens}.`,
        );
      }
    }

    // Final sanity check
    const finalTokens = this.estimateTokens(currentHistory);
    debugLogger.log(
      `Context Manager finished. Final actual token count: ${finalTokens}.`,
    );

    return currentHistory;
  }

  private estimateTokens(history: Content[]): number {
    let chars = 0;
    for (const turn of history) {
      if (!turn.parts) continue;
      for (const part of turn.parts) {
        if (part.text) chars += part.text.length;
        const responseOutput = part.functionResponse?.response?.['output'];
        if (typeof responseOutput === 'string') {
          chars += responseOutput.length;
        }
      }
    }
    return Math.floor(chars / 4);
  }
}
