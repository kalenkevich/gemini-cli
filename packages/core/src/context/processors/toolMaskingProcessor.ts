import type { Episode } from '../ir/types.js';
/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ContextAccountingState, ContextProcessor } from '../pipeline.js';
import type { Config } from '../../config/config.js';
import { estimateTokenCountSync } from '../../utils/tokenCalculation.js';
import { sanitizeFilenamePart } from '../../utils/fileUtils.js';
import * as fsPromises from 'node:fs/promises';
import path from 'node:path';
import {
  ACTIVATE_SKILL_TOOL_NAME,
  MEMORY_TOOL_NAME,
  ASK_USER_TOOL_NAME,
  ENTER_PLAN_MODE_TOOL_NAME,
  EXIT_PLAN_MODE_TOOL_NAME,
} from '../../tools/tool-names.js';

const UNMASKABLE_TOOLS = new Set([
  ACTIVATE_SKILL_TOOL_NAME,
  MEMORY_TOOL_NAME,
  ASK_USER_TOOL_NAME,
  ENTER_PLAN_MODE_TOOL_NAME,
  EXIT_PLAN_MODE_TOOL_NAME,
]);

export class ToolMaskingProcessor implements ContextProcessor {
  readonly name = 'ToolMasking';
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async process(
    episodes: Episode[],
    state: ContextAccountingState,
  ): Promise<Episode[]> {
    const maskingConfig =
      this.config.getContextManagementConfig().strategies.toolMasking;
    if (!maskingConfig) return episodes;
    if (state.isBudgetSatisfied) return episodes;

    const newEpisodes = [...episodes];
    let cumulativeToolTokens = 0;
    let protectionBoundaryReached = false;
    let totalPrunableTokens = 0;

    const prunableParts: Array<{
      epIndex: number;
      stepIndex: number;
      tokens: number;
      content: string;
      originalStep: any;
    }> = [];

    for (let i = newEpisodes.length - 1; i >= 0; i--) {
      const ep = newEpisodes[i];
      if (!ep || !ep.steps || state.protectedEpisodeIds.has(ep.id)) continue;

      for (let j = ep.steps.length - 1; j >= 0; j--) {
        const step = ep.steps[j];
        if (step.type !== 'TOOL_EXECUTION') continue;

        const toolName = step.toolName;
        if (toolName && UNMASKABLE_TOOLS.has(toolName)) continue;

        const toolOutputContent =
          typeof step.observation === 'object' && step.observation
            ? JSON.stringify(step.observation, null, 2)
            : String(step.observation || '');
        if (!toolOutputContent || this.isAlreadyMasked(toolOutputContent))
          continue;

        const partTokens = estimateTokenCountSync([
          {
            functionResponse: {
              name: toolName,
              response: step.observation as any,
              id: step.id,
            },
          },
        ]);

        if (!protectionBoundaryReached) {
          cumulativeToolTokens += partTokens;
          if (cumulativeToolTokens > maskingConfig.minObservationTokens) {
            protectionBoundaryReached = true;
            totalPrunableTokens += partTokens;
            prunableParts.push({
              epIndex: i,
              stepIndex: j,
              tokens: partTokens,
              content: toolOutputContent,
              originalStep: step,
            });
          }
        } else {
          totalPrunableTokens += partTokens;
          prunableParts.push({
            epIndex: i,
            stepIndex: j,
            tokens: partTokens,
            content: toolOutputContent,
            originalStep: step,
          });
        }
      }
    }

    if (totalPrunableTokens < 1) {
      return newEpisodes;
    }

    let toolOutputsDir = path.join(
      this.config.storage.getProjectTempDir(),
      'tool-outputs',
    );
    const sessionId = this.config.getSessionId();
    if (sessionId) {
      toolOutputsDir = path.join(
        toolOutputsDir,
        `session-${sanitizeFilenamePart(sessionId)}`,
      );
    }
    await fsPromises.mkdir(toolOutputsDir, { recursive: true });

    for (const item of prunableParts) {
      const { content, tokens, originalStep } = item;
      const step = originalStep;

      const toolName = step.toolName || 'unknown_tool';
      const callId = step.id || Date.now().toString();
      const fileName = `${sanitizeFilenamePart(toolName).toLowerCase()}_${sanitizeFilenamePart(callId).toLowerCase()}_${Math.random().toString(36).substring(7)}.txt`;
      const filePath = path.join(toolOutputsDir, fileName);

      await fsPromises.writeFile(filePath, content, 'utf-8');

      const fileSizeMB = (
        Buffer.byteLength(content, 'utf8') /
        1024 /
        1024
      ).toFixed(2);
      const totalLines = content.split('\n').length;

      const maskedSnippet = `<tool_output_masked>\n[Tool output (${fileSizeMB}MB, ${totalLines} lines) masked to preserve context window. Full output saved to: ${filePath}]\n</tool_output_masked>`;

      const newTaskTokens = estimateTokenCountSync([
        {
          functionResponse: {
            name: toolName,
            response: { ...(step.observation as any), output: maskedSnippet },
            id: step.id,
          },
        },
      ]);
      step.presentation = {
        observation: { ...(step.observation as any), output: maskedSnippet },
        tokens: newTaskTokens,
      };
      const savings = tokens - newTaskTokens;

      if (savings > 0) {
        step.metadata.currentTokens = newTaskTokens;
        step.metadata.transformations.push({
          processorName: 'ToolMasking',
          action: 'MASKED',
          timestamp: Date.now(),
          diskPointer: filePath,
        });
      }
    }

    return newEpisodes;
  }

  private isAlreadyMasked(content: string): boolean {
    return content.includes('<tool_output_masked>');
  }
}
