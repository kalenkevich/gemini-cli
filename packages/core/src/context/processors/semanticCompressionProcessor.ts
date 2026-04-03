import { estimateTokenCountSync } from '../../utils/tokenCalculation.js';
/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import type { Episode, ToolExecution } from '../ir/types.js';
import type { ContextAccountingState, ContextProcessor } from '../pipeline.js';
import type { Config } from '../../config/config.js';
import { debugLogger } from '../../utils/debugLogger.js';
import { LlmRole } from '../../telemetry/types.js';
import { getResponseText } from '../../utils/partUtils.js';
import * as fsPromises from 'node:fs/promises';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';

export type FileLevel = 'FULL' | 'PARTIAL' | 'SUMMARY' | 'EXCLUDED';

export interface FileRecord {
  level: FileLevel;
  cachedSummary?: string;
  contentHash?: string;
  startLine?: number;
  endLine?: number;
}

interface CompressionRecord {
  level: FileLevel;
  startLine?: number;
  endLine?: number;
}

interface CompressionRecordJSON {
  level: FileLevel;
  start_line?: number;
  end_line?: number;
}

function hashStringSlice(
  content: string,
  start: number = 0,
  end: number = 12,
): string {
  return crypto
    .createHash('sha256')
    .update(content)
    .digest('hex')
    .slice(start, end);
}

export class SemanticCompressionProcessor implements ContextProcessor {
  readonly name = 'SemanticCompression';
  private config: Config;
  private state: Map<string, FileRecord> = new Map();
  private stateFilePath: string;

  constructor(config: Config) {
    this.config = config;
    const dir = this.config.storage?.getProjectTempDir() || '/tmp';
    this.stateFilePath = path.join(dir, 'compression_state.json');
  }

  async process(
    episodes: Episode[],
    state: ContextAccountingState,
  ): Promise<Episode[]> {
    if (state.isBudgetSatisfied) {
      return episodes;
    }

    debugLogger.log(
      'SemanticCompressionProcessor: Initializing LLM-based file compression.',
    );

    let userPrompt = 'Please refer to the history.';
    for (let i = episodes.length - 1; i >= 0; i--) {
      if (episodes[i].trigger.type === 'USER_PROMPT') {
        userPrompt =
          (episodes[i].trigger as any).text || 'Please refer to the history.';
        break;
      }
    }

    await this.loadState();
    const compressedEpisodes = await this.compressHistory(
      episodes,
      userPrompt,
      state,
    );

    return compressedEpisodes;
  }

  private async loadState() {
    try {
      if (existsSync(this.stateFilePath)) {
        const data = await fsPromises.readFile(this.stateFilePath, 'utf-8');
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const parsed: Record<string, FileRecord> = JSON.parse(data);
        for (const [k, v] of Object.entries(parsed)) {
          this.state.set(k, v);
        }
      }
    } catch (e) {
      debugLogger.warn('Failed to load compression state: ' + e);
    }
  }

  private async saveState() {
    try {
      const obj: Record<string, FileRecord> = {};
      for (const [k, v] of this.state.entries()) {
        obj[k] = v;
      }
      await fsPromises.writeFile(
        this.stateFilePath,
        JSON.stringify(obj, null, 2),
        'utf-8',
      );
    } catch (e) {
      debugLogger.warn('Failed to save compression state: ' + e);
    }
  }

  private async compressHistory(
    episodes: Episode[],
    userPrompt: string,
    state: ContextAccountingState,
    abortSignal?: AbortSignal,
  ): Promise<Episode[]> {
    // Pass 1: Find protected files
    const protectedFiles = new Set<string>();
    for (let i = 0; i < episodes.length; i++) {
      const ep = episodes[i]!;
      if (state.protectedEpisodeIds.has(ep.id)) continue;
      for (const step of ep.steps) {
        if (
          step.type === 'TOOL_EXECUTION' &&
          (step.toolName === 'read_file' || step.toolName === 'read_many_files')
        ) {
          if (state.protectedEpisodeIds.has(episodes[i]!.id)) {
            const intent = step.intent;
            if (intent['filepath'] && typeof intent['filepath'] === 'string')
              protectedFiles.add(intent['filepath']);
            if (Array.isArray(intent['paths']))
              intent['paths'].forEach((p: string) => protectedFiles.add(p));
          }
        }
      }
    }

    // Pass 2: Collect files needing routing decisions
    type PendingFile = {
      filepath: string;
      rawContent: string;
      contentToProcess: string;
      lines: string[];
      preview: string;
      lineCount: number;
    };
    const pendingFiles: PendingFile[] = [];
    const pendingFilesSet = new Set<string>();

    for (let i = 0; i < episodes.length; i++) {
      const ep = episodes[i]!;
      if (state.protectedEpisodeIds.has(ep.id)) continue;
      for (const step of ep.steps) {
        if (step.type !== 'TOOL_EXECUTION') continue;
        if (
          step.toolName !== 'read_file' &&
          step.toolName !== 'read_many_files'
        )
          continue;

        const output =
          typeof step.observation === 'object' && step.observation
            ? step.observation['output']
            : null;
        if (!output || typeof output !== 'string') continue;

        const match = output.match(/--- (.+?) ---\n/);
        let filepath = '';
        if (match) filepath = match[1];
        else {
          const lines = output.split('\n');
          if (lines[0] && lines[0].includes('---'))
            filepath = lines[0].replace(/---/g, '').trim();
        }

        if (!filepath || protectedFiles.has(filepath)) continue;

        const hash = hashStringSlice(output);
        const existing = this.state.get(filepath);
        if (
          existing?.level === 'SUMMARY' &&
          existing.cachedSummary &&
          existing.contentHash === hash
        )
          continue;

        if (pendingFilesSet.has(filepath)) continue;
        pendingFilesSet.add(filepath);

        let contentToProcess = output;
        if (contentToProcess.startsWith('--- ')) {
          const firstNewline = contentToProcess.indexOf('\n');
          if (firstNewline !== -1)
            contentToProcess = contentToProcess.substring(firstNewline + 1);
        }
        const lines = contentToProcess.split('\n');

        pendingFiles.push({
          filepath,
          rawContent: output,
          contentToProcess,
          lines,
          preview: lines.slice(0, 30).join('\n'),
          lineCount: lines.length,
        });
      }
    }

    const routingDecisions = await this.batchQueryModel(
      pendingFiles.map((f) => ({
        filepath: f.filepath,
        lineCount: f.lineCount,
        preview: f.preview,
      })),
      userPrompt,
      abortSignal,
    );

    for (const f of pendingFiles) {
      const decision = routingDecisions.get(f.filepath) ?? {
        level: 'FULL' as FileLevel,
      };
      const record = this.state.get(f.filepath) ?? {
        level: 'FULL' as FileLevel,
      };
      const hash = hashStringSlice(f.rawContent);
      if (record.contentHash && record.contentHash !== hash)
        record.cachedSummary = undefined;
      record.contentHash = hash;
      record.level = decision.level;
      record.startLine = decision.startLine;
      record.endLine = decision.endLine;
      this.state.set(f.filepath, record);
    }
    await this.saveState();

    // Pass 4: Apply decisions
    const result: Episode[] = [];
    for (let i = 0; i < episodes.length; i++) {
      const ep = { ...episodes[i], steps: [...episodes[i].steps] };
      if (state.protectedEpisodeIds.has(ep.id)) {
        result.push(ep);
        continue;
      }

      for (let j = 0; j < ep.steps.length; j++) {
        const step = ep.steps[j];
        if (step.type === 'TOOL_EXECUTION') {
          ep.steps[j] = await this.applyCompressionDecision(
            step,
            protectedFiles,
            abortSignal,
          );
        }
      }
      result.push(ep);
    }

    return result;
  }

  private async applyCompressionDecision(
    step: ToolExecution,
    protectedFiles: Set<string>,
    abortSignal?: AbortSignal,
  ): Promise<ToolExecution> {
    if (step.toolName !== 'read_file' && step.toolName !== 'read_many_files')
      return step;

    const output =
      typeof step.observation === 'object' && step.observation
        ? step.observation['output']
        : null;
    if (!output || typeof output !== 'string') return step;

    const match = output.match(/--- (.+?) ---\n/);
    let filepath = '';
    if (match) filepath = match[1];
    else {
      const lines = output.split('\n');
      if (lines[0] && lines[0].includes('---'))
        filepath = lines[0].replace(/---/g, '').trim();
      else return step;
    }

    if (protectedFiles.has(filepath)) return step;

    const record = this.state.get(filepath);
    if (!record || record.level === 'FULL') return step;

    let contentToProcess = output;
    if (contentToProcess.startsWith('--- ')) {
      const firstNewline = contentToProcess.indexOf('\n');
      if (firstNewline !== -1)
        contentToProcess = contentToProcess.substring(firstNewline + 1);
    }
    const lines = contentToProcess.split('\n');
    let compressed: string;

    if (record.level === 'PARTIAL' && record.startLine && record.endLine) {
      const start = Math.max(0, record.startLine - 1);
      const end = Math.min(lines.length, record.endLine);
      const snippet = lines
        .slice(start, end)
        .map((l, i) => `${start + i + 1} | ${l}`)
        .join('\n');
      compressed = `[Showing lines ${record.startLine}–${record.endLine} of ${lines.length} in ${path.basename(filepath)}. Full file available via read_file.]\n\n${snippet}`;
    } else if (record.level === 'SUMMARY') {
      if (!record.cachedSummary) {
        record.cachedSummary = await this.generateSummary(
          filepath,
          contentToProcess,
          abortSignal,
        );
        this.state.set(filepath, record);
        await this.saveState();
      }
      compressed = `[Summary of ${path.basename(filepath)} (${lines.length} lines). Full file available via read_file.]\n\n${record.cachedSummary}`;
    } else if (record.level === 'EXCLUDED') {
      compressed = `[${path.basename(filepath)} omitted as not relevant to current query. Request via read_file if needed.]`;
    } else {
      return step;
    }

    if (compressed === output) return step;

    const newObservation = { ...(step.observation as any), output: compressed };
    const newTaskTokens = estimateTokenCountSync([
      {
        functionResponse: {
          name: step.toolName,
          response: newObservation,
          id: step.id,
        },
      },
    ]);

    const newStep = {
      ...step,
      presentation: { observation: newObservation, tokens: newTaskTokens },
    };
    newStep.metadata.transformations.push({
      processorName: 'SemanticCompression',
      action: 'SUMMARIZED',
      timestamp: Date.now(),
    });

    return newStep as ToolExecution;
  }

  private async batchQueryModel(
    files: Array<{ filepath: string; lineCount: number; preview: string }>,
    userPrompt: string,
    abortSignal?: AbortSignal,
  ): Promise<Map<string, CompressionRecord>> {
    const results = new Map<string, CompressionRecord>();

    for (const f of files) {
      results.set(f.filepath, { level: 'FULL' });
    }

    if (files.length === 0) return results;

    const systemPrompt =
      'You are a context routing agent for a coding AI session.\n' +
      'For each file listed, decide what level of content to send to the main model.\n' +
      'Levels: FULL, PARTIAL (with line range), SUMMARY, EXCLUDED.\n' +
      'Rules:\n' +
      '- FULL if the file is directly relevant to the query or small (<80 lines)\n' +
      '- PARTIAL if only a specific section is needed — provide start_line and end_line\n' +
      '- SUMMARY for background context files not directly needed\n' +
      '- EXCLUDED for completely unrelated files\n' +
      'Respond ONLY with a JSON object where each key is the filepath and the value is:\n' +
      '{"level":"FULL"|"PARTIAL"|"SUMMARY"|"EXCLUDED","start_line":null,"end_line":null}';

    const fileList = files
      .map(
        (f) =>
          `File: ${f.filepath} (${f.lineCount} lines)\nPreview:\n${f.preview}`,
      )
      .join('\n\n---\n\n');

    const userMessage = `Query: "${userPrompt}"\n\n${fileList}`;

    const client = this.config.getBaseLlmClient();
    try {
      const properties: Record<string, object> = {};
      for (const f of files) {
        properties[f.filepath] = {
          type: 'OBJECT',
          properties: {
            level: { type: 'STRING' },
            start_line: { type: 'INTEGER' },
            end_line: { type: 'INTEGER' },
          },
          required: ['level'],
        };
      }

      const responseJson = await client.generateJson({
        modelConfigKey: { model: 'chat-compression-2.5-flash-lite' },
        contents: [{ role: 'user', parts: [{ text: userMessage }] }],
        systemInstruction: systemPrompt,
        schema: { properties, required: files.map((f) => f.filepath) },
        promptId: 'context-compression-batch-query',
        role: LlmRole.UTILITY_COMPRESSOR,
        abortSignal: abortSignal ?? new AbortController().signal,
      });

      for (const f of files) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        const decision = responseJson[f.filepath] as
          | CompressionRecordJSON
          | undefined;
        if (typeof decision !== 'object') continue;
        if (typeof decision === 'object' && decision && decision.level) {
          results.set(f.filepath, {
            level: decision.level ?? 'FULL',
            startLine: decision.start_line ?? undefined,
            endLine: decision.end_line ?? undefined,
          });
        }
      }
    } catch (e) {
      debugLogger.warn(
        'Batch cloud routing failed: ' + e + '. Defaulting all to FULL.',
      );
    }
    return results;
  }

  private async generateSummary(
    filepath: string,
    content: string,
    abortSignal?: AbortSignal,
  ): Promise<string> {
    const promptMessage = `Summarize this file in 2-3 sentences. Be technical and specific about what it exports, its key functions, and dependencies. File: ${filepath}\n\n${content.slice(0, 4000)}`;
    const client = this.config.getBaseLlmClient();
    try {
      const response = await client.generateContent({
        modelConfigKey: { model: 'chat-compression-2.5-flash-lite' },
        contents: [{ role: 'user', parts: [{ text: promptMessage }] }],
        promptId: 'local-context-compression-summary',
        role: LlmRole.UTILITY_COMPRESSOR,
        abortSignal: abortSignal ?? new AbortController().signal,
      });
      const text = getResponseText(response) ?? '';
      return text.trim();
    } catch (e) {
      return `[Summary generation failed for ${filepath} (cloud error): ${e}]`;
    }
  }
}
