import type { Content } from '@google/genai';
import type {
  ContextAccountingState,
  ContextProcessor,
  ContextProcessorResult,
} from '../pipeline.js';
import type { Config } from '../../config/config.js';
import { ToolOutputMaskingService } from '../toolOutputMaskingService.js';

export class ToolMaskingProcessor implements ContextProcessor {
  readonly name = 'ToolMasking';
  private config: Config;
  private legacyMaskingService: ToolOutputMaskingService;

  constructor(config: Config) {
    this.config = config;
    this.legacyMaskingService = new ToolOutputMaskingService();
  }

  async process(
    history: Content[],
    state: ContextAccountingState,
  ): Promise<ContextProcessorResult> {
    const maskingConfig =
      this.config.getContextManagementConfig().tools.outputMasking;
      
    if (!maskingConfig ) {
      return { history, savedTokens: 0 };
    }

    if (state.isBudgetSatisfied) {
      return { history, savedTokens: 0 };
    }

    // Since we can't touch toolOutputMaskingService, we just delegate to it!
    // It already has all the functionality (file saving, token estimation, etc.)
    const result = await this.legacyMaskingService.mask(history, this.config);

    return {
      history: [...result.newHistory],
      savedTokens: result.tokensSaved,
    };
  }
}
