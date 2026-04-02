/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Content, Part } from '@google/genai';
import { randomUUID } from 'node:crypto';
import type {
  Episode,
  IrMetadata,
  ToolExecution,
  AgentThought,
  AgentYield,
  UserPrompt,
} from './types.js';
import { estimateTokenCountSync } from '../../utils/tokenCalculation.js';

export class IrMapper {
  /**
   * Translates a flat Gemini Content[] array into our rich Episodic Intermediate Representation.
   * Groups adjacent function calls and responses into unified ToolExecution nodes.
   */
  static toIr(history: Content[]): Episode[] {
    const episodes: Episode[] = [];
    let currentEpisode: Partial<Episode> | null = null;
    let pendingCallParts: Map<string, Part> = new Map();

    const createMetadata = (parts: Part[]): IrMetadata => {
      const tokens = estimateTokenCountSync(parts);
      return {
        originalTokens: tokens,
        currentTokens: tokens,
        transformations: [],
      };
    };

    const finalizeEpisode = () => {
      if (currentEpisode && currentEpisode.trigger) {
        episodes.push(currentEpisode as Episode);
      }
      currentEpisode = null;
    };

    for (const msg of history) {
      if (!msg.parts) continue;

      if (msg.role === 'user') {
        // User messages can be either Triggers (new Episode) or Tool Responses (continuation of current Episode steps)
        const hasToolResponses = msg.parts.some((p) => !!p.functionResponse);
        const hasTextParts = msg.parts.some((p) => !!p.text);

        if (hasToolResponses) {
          // It's a tool response. Bind it to pending calls.
          if (!currentEpisode) {
            // Edge case: history starts with a tool response. Create a dummy episode.
            currentEpisode = {
              id: randomUUID(),
              timestamp: Date.now(),
              trigger: {
                id: randomUUID(),
                type: 'SYSTEM_EVENT',
                name: 'history_resume',
                payload: {},
                metadata: createMetadata([]),
              },
              steps: [],
            };
          }

          for (const part of msg.parts) {
            if (part.functionResponse) {
              const callId = part.functionResponse.id || '';
              const matchingCall = pendingCallParts.get(callId);
              
              const step: ToolExecution = {
                id: randomUUID(),
                type: 'TOOL_EXECUTION',
                toolName: part.functionResponse.name || 'unknown',
                intent: (matchingCall?.functionCall?.args as Record<string, unknown>) || {},
                observation: (part.functionResponse.response as Record<string, unknown>) || {},
                metadata: createMetadata(matchingCall ? [matchingCall, part] : [part]),
                _rawCallPart: matchingCall,
                _rawResponsePart: part,
              };
              currentEpisode.steps!.push(step);
              if (callId) pendingCallParts.delete(callId);
            }
          }
        } 
        
        if (hasTextParts) {
          // This is a genuine User Prompt. It begins a new episode.
          finalizeEpisode();
          
          const textParts = msg.parts.filter(p => !!p.text).map(p => p.text).join('\n');
          const trigger: UserPrompt = {
            id: randomUUID(),
            type: 'USER_PROMPT',
            text: textParts,
            parts: msg.parts, // Keep all raw parts (e.g. images) attached
            metadata: createMetadata(msg.parts),
          };

          currentEpisode = {
            id: randomUUID(),
            timestamp: Date.now(),
            trigger,
            steps: [],
          };
        }
      } else if (msg.role === 'model') {
        if (!currentEpisode) {
          // Should rarely happen unless history is malformed, but handle it gracefully
          currentEpisode = {
            id: randomUUID(),
            timestamp: Date.now(),
            trigger: { id: randomUUID(), type: 'SYSTEM_EVENT', name: 'model_init', payload: {}, metadata: createMetadata([]) },
            steps: [],
          };
        }

        for (const part of msg.parts) {
          if (part.functionCall) {
            const callId = part.functionCall.id || '';
            if (callId) pendingCallParts.set(callId, part);
          } else if (part.text) {
            // Is this an AgentThought (intermediate reasoning) or an AgentYield (final response)?
            // For now, we assume if it's the last thing before a UserPrompt, it's a Yield.
            // But structurally in a single model turn, it's an AgentThought unless the episode ends.
            // Let's treat all text as AgentThought during the run. The finalizer can promote the last thought to a Yield if desired.
            
            const thought: AgentThought = {
              id: randomUUID(),
              type: 'AGENT_THOUGHT',
              text: part.text,
              metadata: createMetadata([part]),
            };
            currentEpisode.steps!.push(thought);
          }
        }
      }
    }

    // Promote the very last thought of an episode to a Yield
    if (currentEpisode) {
      if (currentEpisode.steps && currentEpisode.steps.length > 0) {
        const lastStep = currentEpisode.steps[currentEpisode.steps.length - 1];
        if (lastStep.type === 'AGENT_THOUGHT') {
           const yieldNode: AgentYield = {
             id: lastStep.id,
             type: 'AGENT_YIELD',
             text: lastStep.text,
             metadata: lastStep.metadata
           };
           currentEpisode.steps.pop();
           currentEpisode.yield = yieldNode;
        }
      }
      finalizeEpisode();
    }

    return episodes;
  }

  /**
   * Re-serializes the Episodic IR back into a flat Gemini Content[] array.
   */
  static fromIr(episodes: Episode[]): Content[] {
    const history: Content[] = [];

    for (const ep of episodes) {
      // 1. Serialize Trigger
      if (ep.trigger.type === 'USER_PROMPT') {
        history.push({ role: 'user', parts: ep.trigger.parts });
      }

      // 2. Serialize Steps
      // Gemini expects Model(calls) then User(responses).
      // If we have multiple consecutive ToolExecutions, we group the calls, then group the responses.
      let pendingModelParts: Part[] = [];
      let pendingUserParts: Part[] = [];

      const flushPending = () => {
        if (pendingModelParts.length > 0) {
          history.push({ role: 'model', parts: [...pendingModelParts] });
          pendingModelParts = [];
        }
        if (pendingUserParts.length > 0) {
          history.push({ role: 'user', parts: [...pendingUserParts] });
          pendingUserParts = [];
        }
      };

      for (const step of ep.steps) {
        if (step.type === 'AGENT_THOUGHT') {
          flushPending();
          history.push({ role: 'model', parts: [{ text: step.text }] });
        } else if (step.type === 'TOOL_EXECUTION') {
          if (step._rawCallPart) pendingModelParts.push(step._rawCallPart);
          else {
            pendingModelParts.push({ functionCall: { name: step.toolName, args: step.intent as any, id: step.id } });
          }

          if (step._rawResponsePart) pendingUserParts.push(step._rawResponsePart);
          else {
             pendingUserParts.push({ functionResponse: { name: step.toolName, response: step.observation as any, id: step.id } });
          }
        }
      }
      flushPending();

      // 3. Serialize Yield
      if (ep.yield) {
        history.push({ role: 'model', parts: [{ text: ep.yield.text }] });
      }
    }

    return history;
  }
}
