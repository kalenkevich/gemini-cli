/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BaseLlmConnection, LlmRequest, LlmResponse} from '@google/adk';
import {Gemini, BaseLlm} from '@google/adk';
import type { Config } from '../../config/config.js';

const LITE_MODELS = [
  'gemini-3.1-flash-lite-preview',
  'gemini-2.5-flash-lite',
];

const FLASH_MODELS = [
  'gemini-3-flash-preview',
  'gemini-2.5-flash',
];

const PRO_MODELS = [
  'gemini-3.1-pro-preview',
  'gemini-2.5-pro',
];

const SUPPORTED_MODELS: string[] = [
  ...LITE_MODELS,
  ...FLASH_MODELS,
  ...PRO_MODELS,
];

// TODO: Support Abort
export class GeminiCliModel extends BaseLlm {
  supportedModels = SUPPORTED_MODELS;
  private currentModel: BaseLlm;

  constructor(model: string, _config: Config, private abortSignal: AbortSignal) {
    super({model});

    this.currentModel = new Gemini({model});
  }

  setModel(model: string) {
    this.currentModel = new Gemini({model});
  }

  async *generateContentAsync(llmRequest: LlmRequest, stream?: boolean): AsyncGenerator<LlmResponse, void> {
    for await (const response of this.currentModel.generateContentAsync(llmRequest, stream)) {
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