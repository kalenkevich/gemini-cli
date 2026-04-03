/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BasePlugin } from "@google/adk";
import type { Context, LlmRequest, LlmResponse, InvocationContext, BaseTool, Event } from "@google/adk";
import type {Content} from '@google/genai';

export class MaxTurnsError extends Error {
  readonly turnCount: number;
  readonly maxTurns: number;

  constructor(message: string, {
    turnCount,
    maxTurns,
  }: {
    turnCount: number;
    maxTurns: number;
  }) {
    super(message);
    this.name = 'MaxTurnsError';
    this.turnCount = turnCount;
    this.maxTurns = maxTurns;
  }
}

export class MaxTurnsAdkPlugin extends BasePlugin {
  private turnCount = 0;

  constructor(private maxTurns: number) {
    super('gemini-cli_max-turns-plugin');
  }

  override async beforeModelCallback(_params: { callbackContext: Context; llmRequest: LlmRequest; }): Promise<LlmResponse | undefined> {
    this.turnCount++;
    if (this.turnCount > this.maxTurns) {
      throw new MaxTurnsError('Max turns reached', {
        turnCount: this.turnCount,
        maxTurns: this.maxTurns,
      });
    }

    return undefined;
  }
}

export class TokenLimitError extends Error {
  readonly tokensUsed: number;
  readonly tokenLimit: number;

  constructor(message: string, {
    tokensUsed,
    tokenLimit,
  }: {
    tokensUsed: number;
    tokenLimit: number;
  }) {
    super(message);
    this.name = 'TokenLimitError';
    this.tokensUsed = tokensUsed;
    this.tokenLimit = tokenLimit;
  }
}

export class TokenLimitAdkPlugin extends BasePlugin {
  private tokensUsed = 0;

  constructor(private tokenLimit: number) {
    super('gemini-cli_max-budget-plugin');
  }

  override async beforeModelCallback(_params: { callbackContext: Context; llmRequest: LlmRequest; }): Promise<LlmResponse | undefined> {
    if (this.tokensUsed > this.tokenLimit) {
      throw new TokenLimitError('Token limit reached', {
        tokensUsed: this.tokensUsed,
        tokenLimit: this.tokenLimit,
      });
    }

    return undefined;
  }

  override async afterModelCallback(params: { callbackContext: Context; llmRequest: LlmRequest; llmResponse: LlmResponse; }): Promise<LlmResponse | undefined> {
    this.tokensUsed += params.llmResponse.usageMetadata?.totalTokenCount || 0;
    // will fail on next beforeModelCallback

    return undefined;
  }
}

export class MaxTimeError extends Error {
  readonly timeLimit: number;
  readonly timeUsed: number;

  constructor(message: string, {
    timeLimit,
    timeUsed,
  }: {
    timeLimit: number;
    timeUsed: number;
  }) {
    super(message);
    this.name = 'MaxTimeError';
    this.timeLimit = timeLimit;
    this.timeUsed = timeUsed;
  }
}

export class MaxTimeAdkPlugin extends BasePlugin {
  private startTime: number;

  constructor(private maxTimeMs: number) {
    super('gemini-cli_max-time-plugin');
    this.startTime = Date.now();
  }

  override async onEventCallback(_params: { invocationContext: InvocationContext; event: Event; }): Promise<Event | undefined> {
    this.checkTime();

    return undefined;
  }

  override async beforeRunCallback(_params: { invocationContext: InvocationContext }): Promise<Content | undefined> {
    this.checkTime();

    return undefined;
  }

  override async afterRunCallback(_params: { invocationContext: InvocationContext }): Promise<void> {
    this.checkTime();
  }

  override async beforeAgentCallback(_params: { callbackContext: Context; }): Promise<Content | undefined> {
    this.checkTime();

    return undefined;
  }

  override async afterAgentCallback(_params: { callbackContext: Context; }): Promise<Content | undefined> {
    this.checkTime();

    return undefined;
  }

  override async beforeToolCallback(_params: { tool: BaseTool; toolArgs: Record<string, unknown>; toolContext: Context; }): Promise<Record<string, unknown> | undefined> {
    this.checkTime();

    return undefined;
  }

  override async afterToolCallback(_params: { tool: BaseTool; toolArgs: Record<string, unknown>; toolContext: Context; result: Record<string, unknown>; }): Promise<Record<string, unknown> | undefined> {
    this.checkTime();

    return undefined;
  }

  override async beforeModelCallback(_params: { callbackContext: Context; llmRequest: LlmRequest; }): Promise<LlmResponse | undefined> {
    this.checkTime();

    return undefined;
  }

  override async afterModelCallback(_params: { callbackContext: Context; llmRequest: LlmRequest; llmResponse: LlmResponse; }): Promise<LlmResponse | undefined> {
    this.checkTime();

    return undefined;
  }

  private checkTime() {
    const elapsed = Date.now() - this.startTime;
    if (elapsed > this.maxTimeMs) {
      throw new MaxTimeError('Max time reached', {
        timeLimit: this.maxTimeMs,
        timeUsed: elapsed,
      });
    }
  }
}
