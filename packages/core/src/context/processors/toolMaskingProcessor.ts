/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Content, Part } from '@google/genai';
import type {
  ContextAccountingState,
  ContextProcessor,
  ContextProcessorResult,
} from '../pipeline.js';
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
    history: Content[],
    state: ContextAccountingState,
  ): Promise<ContextProcessorResult> {
    const maskingConfig =
      this.config.getContextManagementConfig().tools.outputMasking;

    if (!maskingConfig) {
      return { history, savedTokens: 0 };
    }

    if (state.isBudgetSatisfied) {
      return { history, savedTokens: 0 };
    }

    // Natively forked legacy masking logic
    const newHistory = [...history];
    let cumulativeToolTokens = 0;
    let protectionBoundaryReached = false;
    let totalPrunableTokens = 0;
    let actualTokensSaved = 0;

    const prunableParts: Array<{
      contentIndex: number;
      partIndex: number;
      tokens: number;
      content: string;
      originalPart: Part;
    }> = [];

    const scanStartIdx = maskingConfig.protectLatestTurn
      ? history.length - 2
      : history.length - 1;

    for (let i = scanStartIdx; i >= 0; i--) {
      const content = history[i];
      const parts = content.parts || [];

      for (let j = parts.length - 1; j >= 0; j--) {
        const part = parts[j];
        if (!part.functionResponse) continue;

        const toolName = part.functionResponse.name;
        if (toolName && UNMASKABLE_TOOLS.has(toolName)) continue;

        const toolOutputContent = this.getToolOutputContent(part);
        if (!toolOutputContent || this.isAlreadyMasked(toolOutputContent))
          continue;

        const partTokens = estimateTokenCountSync([part]);

        if (!protectionBoundaryReached) {
          cumulativeToolTokens += partTokens;
          if (cumulativeToolTokens > maskingConfig.protectionThresholdTokens) {
            protectionBoundaryReached = true;
            totalPrunableTokens += partTokens;
            prunableParts.push({
              contentIndex: i,
              partIndex: j,
              tokens: partTokens,
              content: toolOutputContent,
              originalPart: part,
            });
          }
        } else {
          totalPrunableTokens += partTokens;
          prunableParts.push({
            contentIndex: i,
            partIndex: j,
            tokens: partTokens,
            content: toolOutputContent,
            originalPart: part,
          });
        }
      }
    }

    if (totalPrunableTokens < maskingConfig.minPrunableThresholdTokens) {
      return { history, savedTokens: 0 };
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
      const { contentIndex, partIndex, content, tokens } = item;
      const contentRecord = newHistory[contentIndex];
      const part = contentRecord.parts![partIndex];

      const toolName = part.functionResponse!.name || 'unknown_tool';
      const callId = part.functionResponse!.id || Date.now().toString();
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

      const maskedPart = {
        functionResponse: {
          // eslint-disable-next-line @typescript-eslint/no-misused-spread
          ...part.functionResponse!,
          response: {
            ...part.functionResponse!.response,
            output: maskedSnippet,
          },
        },
      };

      const newTaskTokens = estimateTokenCountSync([maskedPart]);
      const savings = tokens - newTaskTokens;

      if (savings > 0) {
        const newParts = [...contentRecord.parts!];
        newParts[partIndex] = maskedPart;
        newHistory[contentIndex] = {
          role: contentRecord.role,
          parts: newParts,
        };
        actualTokensSaved += savings;
      }
    }

    return {
      history: newHistory,
      savedTokens: actualTokensSaved,
    };
  }

  private getToolOutputContent(part: Part): string | null {
    if (!part.functionResponse) return null;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const response = part.functionResponse.response as Record<string, unknown>;
    if (!response) return null;
    return JSON.stringify(response, null, 2);
  }

  private isAlreadyMasked(content: string): boolean {
    return content.includes('<tool_output_masked>');
  }
}
