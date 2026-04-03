/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {Session, Event as AdkEvent, BaseSessionService} from '@google/adk';
import {Runner, StreamingMode} from '@google/adk';
import type {Content as GenAIContent} from '@google/genai';
import {type GeminiCliAgent} from './adk-agent.js';
import {TokenLimitError, MaxTurnsError, MaxTurnsAdkPlugin, MaxTimeError, TokenLimitAdkPlugin, MaxTimeAdkPlugin} from './plugins.js';

interface Deferred {
  promise: Promise<void>;
  resolve: () => void;
  reject: (error: Error) => void;
}

function createDeferred(): Deferred {
  let resolve: () => void;
  let reject: (error: Error) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve: resolve!, reject: reject! };
}

export enum AgentRunStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  ABORTED = 'aborted',
  FAILED = 'failed',
}

export enum AgentRunFailReason {
  MAX_TURNS = 'max_turns',
  MAX_BUDGET = 'max_budget',
  MAX_TIME = 'max_time',
}

export interface AgentRunLimitConfig {
  maxTurns?: number;
  maxTokenBudget?: number;
  maxExecutionTimeMs?: number;
}

export interface AgentRunInvocationOptions {
  agent: GeminiCliAgent;
  session: Session;
  sessionService: BaseSessionService;
  limitsConfig: AgentRunLimitConfig;
}

export class AgentRunInvocation {
  private _invocationId?: string;
  private _status: AgentRunStatus = AgentRunStatus.PENDING;
  // eslint-disable-next-line require-yield
  private _stream: AsyncGenerator<AdkEvent, void, void> = (async function*() {
    return;
  })();
  private abortController = new AbortController();
  private deferred = createDeferred();
  private _failReason?: AgentRunFailReason;

  private readonly agent: GeminiCliAgent;
  private readonly session: Session;
  private readonly sessionService: BaseSessionService;
  private readonly limitsConfig: AgentRunLimitConfig;

  get invocationId(): string | undefined {
    return this._invocationId;
  }

  get status(): AgentRunStatus {
    return this._status;
  }

  get failReason(): AgentRunFailReason | undefined {
    return this._failReason;
  }

  get stream(): AsyncGenerator<AdkEvent, void, void> {
    return this._stream;
  }

  constructor({
    agent,
    session,
    sessionService,
    limitsConfig,
  }: AgentRunInvocationOptions) {
    this.agent = agent;
    this.session = session;
    this.sessionService = sessionService;
    this.limitsConfig = limitsConfig;
  }

  async run(userContent: GenAIContent): Promise<AsyncGenerator<AdkEvent, void, void>> {
    if (this._status !== AgentRunStatus.PENDING) {
      throw new Error('Agent run is not in pending state. Please use new AgentRunInvocation() to create a new run.');
    }

    const runner = new Runner({
      appName: this.session.appName,
      agent: this.agent,
      sessionService: this.sessionService,
      plugins: [
        this.limitsConfig.maxTurns ? new MaxTurnsAdkPlugin(this.limitsConfig.maxTurns) : undefined,
        this.limitsConfig.maxTokenBudget ? new TokenLimitAdkPlugin(this.limitsConfig.maxTokenBudget) : undefined,
        this.limitsConfig.maxExecutionTimeMs ? new MaxTimeAdkPlugin(this.limitsConfig.maxExecutionTimeMs) : undefined,
      ].filter((plugin) => plugin !== undefined),
    });

    const getStatus = () => this._status;
    const setFailReason = (reason: AgentRunFailReason) => {
      this._failReason = reason;
    };
    const setStatus = (status: AgentRunStatus) => {
      this._status = status;
    };
    const deferred = this.deferred;
    this._status = AgentRunStatus.RUNNING;

    const eventsStream = runner.runAsync({
      userId: this.session.userId,
      sessionId: this.session.id,
      newMessage: userContent,
      runConfig: {
        streamingMode: StreamingMode.SSE,
      }
      // TODO: support abort signal in the ADK Runner
      // abortSignal: this.abortController.signal,
    });

    const firstEvent = await eventsStream.next();
    if (firstEvent.done) {
      this._status = AgentRunStatus.FAILED;
      throw new Error('No first event received from ADK runner');
    }
    const adkEvent = firstEvent.value;
    this._invocationId = adkEvent.invocationId;

    return this._stream = (async function*() {
      yield adkEvent;

      try {
        for await (const event of eventsStream) {
          if (getStatus() === AgentRunStatus.ABORTED) {
            break;
          }

          yield event;
        }

        setStatus(AgentRunStatus.COMPLETED);
        deferred.resolve();
      } catch (error: unknown) {
        if (error instanceof MaxTurnsError) {
          setFailReason(AgentRunFailReason.MAX_TURNS);
        } else if (error instanceof TokenLimitError) {
          setFailReason(AgentRunFailReason.MAX_BUDGET);
        } else if (error instanceof MaxTimeError) {
          setFailReason(AgentRunFailReason.MAX_TIME);
        }

        // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
        deferred.reject(error as Error);
        throw error;
      } 
    })();
  }

  abort(): void {
    if (this._status !== AgentRunStatus.RUNNING) {
      return;
    }

    this.abortController.abort();
    // shound be removed once ADK will support abort signal in the runner
    this.agent?.abort();

    this._status = AgentRunStatus.ABORTED;
    this.deferred.reject(new Error('Aborted'));
  }

  wait(): Promise<void> {
    return this.deferred.promise.finally(() => {});
  }
}