/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Content } from '@google/genai';
import type { ContextAccountingState, ContextProcessor, ContextProcessorResult } from '../pipeline.js';
import type { ContextCompressionService } from '../contextCompressionService.js';
import { debugLogger } from '../../utils/debugLogger.js';

export class SemanticCompressionProcessor implements ContextProcessor {
  readonly name = 'SemanticCompression';
  private compressionService: ContextCompressionService;

  constructor(compressionService: ContextCompressionService) {
    this.compressionService = compressionService;
  }

  async process(
    history: Content[],
    state: ContextAccountingState
  ): Promise<ContextProcessorResult> {
    if (state.isBudgetSatisfied) {
      return { history, savedTokens: 0 };
    }

    debugLogger.log(`SemanticCompressionProcessor: Initializing LLM-based file compression.`);
    
    // The current ContextCompressionService requires the userPrompt to route properly.
    // For this generalized pipeline, we try to extract the last user message as the query.
    let userPrompt = 'Please refer to the history.';
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === 'user' && history[i].parts?.[0]?.text) {
        userPrompt = history[i].parts![0].text || 'Please refer to the history.';
        break;
      }
    }

    const compressedHistory = await this.compressionService.compressHistory(history, userPrompt);
    
    // We don't have an exact saved token count from the service right now without an extra call.
    // Since it's the last in the standard pipeline, we can just return 0 to force a full recount 
    // at the end of the ContextManager run.
    return {
      history: compressedHistory,
      savedTokens: 0,
    };
  }
}
