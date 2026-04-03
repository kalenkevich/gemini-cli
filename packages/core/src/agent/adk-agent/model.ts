/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BaseLlmConnection, LlmRequest, LlmResponse } from '@google/adk';
import { Gemini, BaseLlm } from '@google/adk';
import type { Config } from '../../config/config.js';

// TODO: use Gemini CLI model config
const LITE_MODELS = ['gemini-3.1-flash-lite-preview', 'gemini-2.5-flash-lite'];
const FLASH_MODELS = ['gemini-3-flash-preview', 'gemini-2.5-flash'];
const PRO_MODELS = ['gemini-3.1-pro-preview', 'gemini-2.5-pro'];

const SUPPORTED_MODELS: string[] = [
  ...LITE_MODELS,
  ...FLASH_MODELS,
  ...PRO_MODELS,
];

// TODO: Support Abort
export class GeminiCliModel extends BaseLlm {
  supportedModels = SUPPORTED_MODELS;
  private currentModel: BaseLlm;

  constructor(
    model: string,
    config: Config,
    private abortSignal: AbortSignal,
  ) {
    const canonicalModelName = getCanonicalModel(model);
    super({ model: canonicalModelName });

    this.currentModel = new Gemini({
      model: canonicalModelName,
      apiKey: getApiKey(),
    });
  }

  setModel(model: string) {
    const canonicalModelName = getCanonicalModel(model);
    this.currentModel = new Gemini({
      model: canonicalModelName,
      apiKey: getApiKey(),
    });
  }

  async *generateContentAsync(
    llmRequest: LlmRequest,
    stream?: boolean,
  ): AsyncGenerator<LlmResponse, void> {
    for await (const response of this.currentModel.generateContentAsync(
      llmRequest,
      stream,
    )) {
      // TODO: remove once ADK will support abort signal in the model
      if (this.abortSignal.aborted) {
        break;
      }

      yield response;
    }
  }

  connect(llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    return this.currentModel.connect(llmRequest);
  }
}

// TODO: use config to get API key
function getApiKey(): string | undefined {
  return process.env['GOOGLE_GENAI_API_KEY'] || process.env['GEMINI_API_KEY'];
}

// TODO: this should be a fallback model based on the config
function getCanonicalModel(model: string): string {
  switch (model) {
    case 'auto':
    case 'pro':
    case 'auto-gemini-3':
      return 'gemini-3.1-pro-preview';
    case 'auto-gemini-2.5':
      return 'gemini-2.5-pro';
    case 'flash':
    case 'gemini-3-flash-preview':
      return 'gemini-3-flash-preview';
    case 'gemini-2.5-flash':
      return 'gemini-2.5-flash';
    case 'gemini-3.1-pro-preview':
      return 'gemini-3.1-pro-preview';
    case 'gemini-2.5-pro':
      return 'gemini-2.5-pro';
    default:
      throw new Error(`Unknown model: ${model}`);
  }
}
