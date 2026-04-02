/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SemanticCompressionProcessor } from './semanticCompressionProcessor.js';
import type { Config } from '../../config/config.js';
import type { Content } from '@google/genai';
import * as fsSync from 'node:fs';

vi.mock('node:fs/promises', () => ({
  readFile: vi.fn(),
  writeFile: vi.fn(),
}));

vi.mock('node:fs', () => ({
  existsSync: vi.fn(),
}));

describe('SemanticCompressionProcessor', () => {
  let mockConfig: Partial<Config>;
  let processor: SemanticCompressionProcessor;
  const generateContentMock: ReturnType<typeof vi.fn> = vi.fn();
  const generateJsonMock: ReturnType<typeof vi.fn> = vi.fn();

  beforeEach(() => {
    mockConfig = {
      storage: {
        getProjectTempDir: vi.fn().mockReturnValue('/mock/temp/dir'),
      },
      getBaseLlmClient: vi.fn().mockReturnValue({
        generateContent: generateContentMock,
        generateJson: generateJsonMock,
      }),
    } as unknown as Config;

    vi.mocked(fsSync.existsSync).mockReturnValue(false);

    processor = new SemanticCompressionProcessor(mockConfig as Config);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const getDummyState = () => ({
    currentTokens: 1000,
    maxTokens: 500,
    retainedTokens: 400,
    frontBufferStartIndex: 4,
    backBufferEndIndex: 3,
    isBudgetSatisfied: false,
  });

  describe('process', () => {
    it('bypasses compression if budget is satisfied', async () => {
      const history: Content[] = [{ role: 'user', parts: [{ text: 'hello' }] }];
      const state = { ...getDummyState(), isBudgetSatisfied: true };

      const res = await processor.process(history, state);
      expect(res.history).toStrictEqual(history);
    });

    it('protects files that were read within the RECENT_TURNS_PROTECTED window', async () => {
      const history: Content[] = [
        // Turn 0 & 1 (Old)
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                name: 'read_file',
                args: { filepath: 'src/app.ts' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: 'read_file',
                response: {
                  output: '--- src/app.ts ---\nLine 1\nLine 2\nLine 3',
                },
              },
            },
          ],
        },

        // Padding (Turns 2 & 3)
        { role: 'model', parts: [{ text: 'res 1' }] },
        { role: 'user', parts: [{ text: 'res 2' }] },

        // Padding (Turns 4 & 5)
        { role: 'model', parts: [{ text: 'res 3' }] },
        { role: 'user', parts: [{ text: 'res 4' }] },

        // Recent Turn (Turn 6 & 7, inside window, cutoff is Math.max(0, 8 - 4) = 4)
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                name: 'read_file',
                args: { filepath: 'src/app.ts' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: 'read_file',
                response: {
                  output: '--- src/app.ts ---\nLine 1\nLine 2\nLine 3',
                },
              },
            },
          ],
        },
      ];

      const res = await processor.process(history, getDummyState());

      // Because src/app.ts was re-read recently, the OLD response is PROTECTED.
      const compressedOutput =
        res.history[1].parts![0].functionResponse!.response!['output'];
      expect(compressedOutput).toBe(
        '--- src/app.ts ---\nLine 1\nLine 2\nLine 3',
      );
      expect(generateContentMock).not.toHaveBeenCalled();
    });

    it('compresses files read outside the protected window', async () => {
      const history: Content[] = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                name: 'read_file',
                args: { filepath: 'src/old.ts' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: 'read_file',
                response: {
                  output: '--- src/old.ts ---\nLine 1\nLine 2\nLine 3\nLine 4',
                },
              },
            },
          ],
        },
        { role: 'model', parts: [{ text: 'msg 2' }] },
        { role: 'user', parts: [{ text: 'res 2' }] },
        { role: 'model', parts: [{ text: 'msg 3' }] },
        { role: 'user', parts: [{ text: 'res 3' }] },
        { role: 'model', parts: [{ text: 'msg 4' }] },
        { role: 'user', parts: [{ text: 'res 4' }] },
      ];

      generateJsonMock.mockResolvedValueOnce({
        'src/old.ts': {
          level: 'PARTIAL',
          start_line: 2,
          end_line: 3,
        },
      });

      const res = await processor.process(history, getDummyState());
      const compressedOutput =
        res.history[1].parts![0].functionResponse!.response!['output'];

      expect(compressedOutput).toContain('[Showing lines 2–3 of 4 in old.ts.');
      expect(compressedOutput).toContain('2 | Line 2');
      expect(compressedOutput).toContain('3 | Line 3');
    });

    it('returns SUMMARY and hits cache on subsequent requests', async () => {
      const history1: Content[] = [
        {
          role: 'model',
          parts: [
            {
              functionCall: {
                name: 'read_file',
                args: { filepath: 'src/index.ts' },
              },
            },
          ],
        },
        {
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: 'read_file',
                response: {
                  output: `--- src/index.ts ---\nVery long content here...`,
                },
              },
            },
          ],
        },
        { role: 'model', parts: [{ text: 'p1' }] },
        { role: 'user', parts: [{ text: 'p2' }] },
        { role: 'model', parts: [{ text: 'p3' }] },
        { role: 'user', parts: [{ text: 'p4' }] },
        { role: 'model', parts: [{ text: 'p5' }] },
        { role: 'user', parts: [{ text: 'p6' }] },
      ];

      generateJsonMock.mockResolvedValueOnce({
        'src/index.ts': { level: 'SUMMARY' },
      });
      generateContentMock.mockResolvedValueOnce({
        candidates: [
          { content: { parts: [{ text: 'This is a cached summary.' }] } },
        ],
      });

      await processor.process(history1, getDummyState());
      expect(generateJsonMock).toHaveBeenCalledTimes(1);
      expect(generateContentMock).toHaveBeenCalledTimes(1);

      const history2: Content[] = [
        ...history1,
        { role: 'model', parts: [{ text: 'p7' }] },
        { role: 'user', parts: [{ text: 'p8' }] },
      ];

      generateJsonMock.mockResolvedValueOnce({
        'src/index.ts': { level: 'SUMMARY' },
      });

      await processor.process(history2, getDummyState());

      expect(generateJsonMock).toHaveBeenCalledTimes(1);
      expect(generateContentMock).toHaveBeenCalledTimes(1);
    });
  });
});
