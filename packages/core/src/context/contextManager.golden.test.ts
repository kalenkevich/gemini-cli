import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ContextManager } from './contextManager.js';
import type { Content } from '@google/genai';
import { ToolMaskingProcessor } from './processors/toolMaskingProcessor.js';
import { HistorySquashingProcessor } from './processors/historySquashingProcessor.js';
import { SemanticCompressionProcessor } from './processors/semanticCompressionProcessor.js';
import { ContextCompressionService } from './contextCompressionService.js';

describe('ContextManager Golden Tests', () => {
  let mockConfig: any;
  let contextManager: ContextManager;

  beforeEach(() => {
    mockConfig = {
      isContextManagementEnabled: vi.fn().mockReturnValue(true),
      getContextManagementConfig: vi.fn().mockReturnValue({
        historyWindow: { maxTokens: 1000, retainedTokens: 500 },
        messageLimits: { normalMaxTokens: 100, retainedMaxTokens: 50, normalizationHeadRatio: 0.1 },
        tools: {
          outputMasking: { enabled: true, protectLatestTurn: false, protectionThresholdTokens: 100, minPrunableThresholdTokens: 50 }
        }
      }),
      storage: { getProjectTempDir: vi.fn().mockReturnValue('/tmp') },
      getBaseLlmClient: vi.fn().mockReturnValue({
        generateJson: vi.fn().mockResolvedValue({
           'test_file.txt': { level: 'SUMMARY' }
        }),
        generateContent: vi.fn().mockResolvedValue({
           candidates: [{ content: { parts: [{ text: 'This is a summary.' }] } }]
        })
      })
    };

    contextManager = new ContextManager(mockConfig, {} as any);
    contextManager.setProcessors([
      new ToolMaskingProcessor(mockConfig),
      new HistorySquashingProcessor(mockConfig),
      new SemanticCompressionProcessor(new ContextCompressionService(mockConfig))
    ]);
  });

  const createLargeHistory = (): Content[] => {
    return [
      {
        role: 'user',
        parts: [
          { text: 'A long long time ago, '.repeat(500) } // Squashing target
        ]
      },
      {
        role: 'model',
        parts: [
          { text: 'in a galaxy far far away...' }
        ]
      },
      {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'some_tool',
              response: { output: 'TOOL OUTPUT DATA '.repeat(500) } // Masking target
            }
          }
        ]
      },
      {
        role: 'user',
        parts: [
           { text: '--- test_file.txt ---\n' + 'FILE DATA '.repeat(1000) } // Semantic target
        ]
      }
    ];
  };

  it('should process history and match golden snapshot', async () => {
    const history = createLargeHistory();
    const result = await contextManager.processHistory(history);
    expect(result).toMatchSnapshot();
  });
  
  it('should not modify history when under budget', async () => {
    mockConfig.getContextManagementConfig.mockReturnValue({
        historyWindow: { maxTokens: 100000, retainedTokens: 50000 },
        messageLimits: { normalMaxTokens: 100, retainedMaxTokens: 50, normalizationHeadRatio: 0.1 },
        tools: {
          outputMasking: { enabled: true, protectLatestTurn: false, protectionThresholdTokens: 100, minPrunableThresholdTokens: 50 }
        }
    });
    const history = createLargeHistory();
    const result = await contextManager.processHistory(history);
    expect(result).toEqual(history);
  });
});
