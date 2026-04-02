/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Part } from '@google/genai';

/**
 * Universal Audit Metadata
 * Tracks the lifecycle and transformations of a node within the IR.
 * This guarantees perfect reversibility and enables long-term memory offloading.
 */
export interface IrMetadata {
  /** The estimated number of tokens this node originally consumed. */
  originalTokens: number;
  /** The current estimated number of tokens this node consumes in its degraded state. */
  currentTokens: number;
  /** An audit trail of all transformations applied by ContextProcessors. */
  transformations: Array<{
    processorName: string;
    action: 'MASKED' | 'TRUNCATED' | 'SUMMARIZED' | 'EVICTED' | 'SYNTHESIZED';
    timestamp: number;
    /** Pointer to where the original uncompressed payload was saved (if applicable) */
    diskPointer?: string;
  }>;
}

export type IrNodeType =
  | 'USER_PROMPT'
  | 'SYSTEM_EVENT'
  | 'AGENT_THOUGHT'
  | 'TOOL_EXECUTION'
  | 'AGENT_YIELD';

/** Base interface for all nodes in the Episodic IR */
export interface IrNode {
  readonly id: string;
  readonly type: IrNodeType;
  metadata: IrMetadata;
}

/**
 * Trigger Nodes
 * Events that wake the agent up and initiate an Episode.
 */
export interface UserPrompt extends IrNode {
  readonly type: 'USER_PROMPT';
  text: string;
  parts: Part[]; // For multi-modal inputs (images, etc) attached by the user
}

export interface SystemEvent extends IrNode {
  readonly type: 'SYSTEM_EVENT';
  name: string;
  payload: Record<string, unknown>;
}

export type EpisodeTrigger = UserPrompt | SystemEvent;

/**
 * Step Nodes
 * The internal autonomous actions taken by the agent during its loop.
 */
export interface AgentThought extends IrNode {
  readonly type: 'AGENT_THOUGHT';
  text: string;
}

export interface ToolExecution extends IrNode {
  readonly type: 'TOOL_EXECUTION';
  /** The name of the tool invoked */
  toolName: string;
  /** The arguments passed to the tool (The 'FunctionCall') */
  intent: Record<string, unknown>;
  /** The result returned by the tool (The 'FunctionResponse') */
  observation: string | Record<string, unknown>;
  /** Original raw parts for exact serialization if needed */
  _rawCallPart?: Part;
  _rawResponsePart?: Part;
}

export type EpisodeStep = AgentThought | ToolExecution;

/**
 * Resolution Node
 * The final message where the agent yields control back to the user.
 */
export interface AgentYield extends IrNode {
  readonly type: 'AGENT_YIELD';
  text: string;
}

/**
 * The Episode
 * A discrete, continuous run of the agent. Represents the full cycle from
 * taking control (Trigger) to returning control (Yield), encompassing all
 * internal reasoning and observations (Steps).
 */
export interface Episode {
  readonly id: string;
  /** When the episode began */
  readonly timestamp: number;
  
  /** The event that initiated this run */
  trigger: EpisodeTrigger;
  
  /** The sequence of autonomous actions and observations */
  steps: EpisodeStep[];
  
  /** The final handover back to the user (can be undefined if the episode was aborted/errored) */
  yield?: AgentYield;
}
