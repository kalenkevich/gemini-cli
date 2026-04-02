/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Content } from '@google/genai';

/**
 * State object passed through the processing pipeline.
 * Contains global accounting logic and range delimiters to coordinate degradation without tight coupling.
 */
export interface ContextAccountingState {
  readonly currentTokens: number;
  readonly maxTokens: number;
  readonly retainedTokens: number;
  
  /** 
   * Index in the history array where the "front buffer" begins.
   * Everything after this index is considered recent and highly protected.
   */
  readonly frontBufferStartIndex: number; 
  
  /**
   * Index in the history array where the "back buffer" ends.
   * Everything before this index is considered old and ripe for gradual degradation.
   */
  readonly backBufferEndIndex: number;    
  
  /**
   * True if currentTokens <= retainedTokens.
   * Processors should generally exit early if this is true.
   */
  readonly isBudgetSatisfied: boolean;
}

/**
 * Result returned by a ContextProcessor after execution.
 */
export interface ContextProcessorResult {
  /** The potentially mutated or newly copied history array. */
  history: Content[];
  /** The estimated number of tokens saved during processing. */
  savedTokens: number;
}

/**
 * Interface for all context degradation strategies.
 */
export interface ContextProcessor {
  /** Unique name for telemetry and logging. */
  readonly name: string;
  
  /**
   * Processes the history payload based on the current accounting state.
   */
  process(
    history: Content[], 
    state: ContextAccountingState
  ): Promise<ContextProcessorResult>;
}
