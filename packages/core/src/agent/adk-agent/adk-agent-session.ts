/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Session, Event as AdkEvent, BaseSessionService } from '@google/adk';
import { InMemorySessionService } from '@google/adk';
import { createUserContent, type Content as GenAIContent, type FinishReason } from '@google/genai';
import { AgentSession } from '../agent-session.js';
import type { AgentProtocol, AgentSend, AgentEvent, Unsubscribe } from '../types.js';
import { debugLogger } from '../../utils/debugLogger.js';
import type { Config } from '../../config/config.js';
import {contentPartsToGeminiParts, mapFinishReason} from '../content-utils.js';
import {GeminiCliAgent} from './adk-agent.js';
import {translateEvent, elicitationToAdkToolConfirmation} from './adk-event-translator.js';
import {AgentRunInvocation, AgentRunStatus, AgentRunFailReason} from './agent-run.js';
import {isUserMessage, isElicitationResponses, isUpdateCommand, isAction, type UpdateCommand, type ActionCommand} from './type-utils.js';

const ADK_APP_NAME = 'gemini-cli-app';

type SendResult = { streamId: string | null };

export interface AdkSessionParams {
  userId: string;
  config: Config;
}

export class AdkAgentSession extends AgentSession {
  constructor(params: AdkSessionParams) {
    super(new AdkAgentProtocol(params));
  }
}

export class AdkAgentProtocol implements AgentProtocol {
  private readonly userId: string;
  private readonly config: Config;
  private adkSession?: Session;
  private adkAgent?: GeminiCliAgent;
  private readonly adkSessionService: BaseSessionService;

  private eventCounter = 0;
  private agentEvents: AgentEvent[] = [];
  private currentAgentRunInvocation?: AgentRunInvocation;
  private streamListeners: Array<(event: AgentEvent) => void> = [];

  get events(): readonly AgentEvent[] {
    return this.agentEvents;
  }

  constructor({
    userId,
    config,
  }: AdkSessionParams) {
    this.userId = userId;
    this.config = config;
    this.adkSessionService = new InMemorySessionService();
  }

  async send(payload: AgentSend): Promise<SendResult> {
    if (isUserMessage(payload)) {
      this.emit({
        type: 'message',
        role: 'user',
        content: payload.message.content,
      });

      const genAIParts = contentPartsToGeminiParts(payload.message.content);

      return this.runAdkAgent(createUserContent(genAIParts));
    }

    if (isElicitationResponses(payload)) {
      const toolConfirmationResponses = [];

      for (const elicitation of payload.elicitations) {
        toolConfirmationResponses.push(elicitationToAdkToolConfirmation(
          elicitation,
          this.agentEvents,
        ));

        this.emit({
          type: 'elicitation_response',
          ...elicitation,
        });
      }

      return this.runAdkAgent(createUserContent(toolConfirmationResponses));
    }

    if (isUpdateCommand(payload)) {
      this.emit({
        type: 'session_update',
        ...payload.update,
      });

      return this.processUpdateCommand(payload.update);
    }

    if (isAction(payload)) {
      // TODO: ??? What action to dispatch here?
      return this.processActionCommand(payload.action);
    }

    throw new Error('Unknown payload type');
  }

  subscribe(callback: (event: AgentEvent) => void): Unsubscribe {
    this.streamListeners.push(callback);

    return () => {
      this.streamListeners = this.streamListeners.filter((cb) => cb !== callback);
    };
  }

  async abort(): Promise<void> {
    this.currentAgentRunInvocation?.abort();
  }

  private async prepareAdkSessionIfNeeded() {
    if (this.adkSession) {
      return;
    }

    this.adkAgent = new GeminiCliAgent(this.config.getModel(), this.config);
    this.adkSession = await this.adkSessionService.createSession({
      userId: this.userId,
      appName: ADK_APP_NAME,
    });

    this.emit({
      type: 'initialize',
      sessionId: this.adkSession.id,
      agentId: this.adkAgent.name,
    });
  }

  private emit(event: Partial<AgentEvent>) {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    const agentEvent = {
      ...event,
      id: event.id ?? this.currentAgentRunInvocation!.invocationId! + '-' + this.eventCounter++,
      timestamp: event.timestamp ?? new Date().toISOString(),
      streamId: event.streamId ?? this.currentAgentRunInvocation!.invocationId!,
    } as AgentEvent;

    this.agentEvents.push(agentEvent);
    this.streamListeners.forEach((listener) => listener(agentEvent));
  }

  private async runAdkAgent(newMessage: GenAIContent): Promise<SendResult> {
    await this.prepareAdkSessionIfNeeded();

    // Wait for the previous agent run to finish before starting a new one.
    await this.currentAgentRunInvocation?.wait();
    this.currentAgentRunInvocation = new AgentRunInvocation({
      agent: this.adkAgent!,
      session: this.adkSession!,
      sessionService: this.adkSessionService,
      limitsConfig: {
        maxTurns: this.config.getMaxSessionTurns(),
        maxTokenBudget: undefined,
        maxExecutionTimeMs: undefined,
      },
    });
    const eventStream = await this.currentAgentRunInvocation.run(newMessage);

    this.emit({
      type: 'agent_start',
      streamId: this.currentAgentRunInvocation.invocationId!,
    });

    const processAdkEventStream = async () => {
      const events: AdkEvent[] = [];

      try {
        for await (const adkEvent of eventStream) {
          events.push(adkEvent);

          if (adkEvent.usageMetadata) {
            this.emit({
              type: 'usage',
              model: this.config.getModel(),
              inputTokens: adkEvent.usageMetadata.promptTokenCount,
              outputTokens: 
                (adkEvent.usageMetadata.toolUsePromptTokenCount || 0) +
                (adkEvent.usageMetadata.thoughtsTokenCount || 0) +
                (adkEvent.usageMetadata.candidatesTokenCount || 0),
              cachedTokens: adkEvent.usageMetadata.cachedContentTokenCount,
            });
          }

          const agentEvents = translateEvent(adkEvent);
          for (const agentEvent of agentEvents) {
            this.emit(agentEvent);
          }
        }

        const agentEndEvent = getAgentRunEndEvent(
          events,
          this.currentAgentRunInvocation!.status,
          this.currentAgentRunInvocation!.failReason,
        );
        this.emit(agentEndEvent);
      } catch (e: unknown) {
        this.emit({
          type: 'error',
          status: 'UNKNOWN',
          // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
          message: (e as Error).message,
          fatal: true,
        });

        this.emit({
          type: 'agent_end',
          reason: 'failed',
        });
      } finally {
        this.currentAgentRunInvocation = undefined;
      }
    };

    // do not wait for the stream to finish, just start processing it
    void processAdkEventStream();

    return { streamId: this.currentAgentRunInvocation.invocationId! };
  }

  private async processUpdateCommand(update: UpdateCommand): Promise<SendResult> {
    if (update.title) {
      this.emit({
        type: 'session_update',
        title: update.title,
      });
    }

    if (update.model) {
      this.adkAgent!.setModel(update.model);
      this.emit({
        type: 'session_update',
        model: update.model,
      });
    }

    if (update.config) {
      this.adkAgent!.setConfig(update.config);
      this.emit({
        type: 'session_update',
        config: update.config,
      });
    }

    return { streamId: this.currentAgentRunInvocation!.invocationId || null };
  }

  private async processActionCommand(_action: ActionCommand): Promise<SendResult> {
    debugLogger.warn('[ADK Agent Session]: action command not supported by ADK runner yet');

    return { streamId: this.currentAgentRunInvocation!.invocationId || null };
  }
}

function getAgentRunEndEvent(
  events: AdkEvent[],
  runStatus: AgentRunStatus,
  failReason?: AgentRunFailReason,
): Partial<AgentEvent> {
  if (runStatus === AgentRunStatus.COMPLETED) {
    const lastEvent = events[events.length - 1];
    const elicitationIds = [];
    for (const key of Object.keys(lastEvent.actions?.requestedToolConfirmations || {})) {
      elicitationIds.push(key);
    }

    if (elicitationIds.length > 0) {
      return {
        type: 'agent_end',
        reason: 'elicitation',
        elicitationIds,
      };
    }

    return {
      type: 'agent_end',
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      reason: mapFinishReason(lastEvent.finishReason as FinishReason),
    };
  }

  if (runStatus === AgentRunStatus.FAILED) {
    if (failReason === AgentRunFailReason.MAX_TURNS) {
      return {
        type: 'agent_end',
        reason: 'max_turns',
        data: {
          code: 'MAX_TURNS_EXCEEDED',
        },
      };
    }

    if (failReason === AgentRunFailReason.MAX_BUDGET) {
      return {
        type: 'agent_end',
        reason: 'max_budget',
        data: {
          code: 'MAX_BUDGET_EXCEEDED',
        },
      };
    }

    if (failReason === AgentRunFailReason.MAX_TIME) {
      return {
        type: 'agent_end',
        reason: 'max_time',
        data: {
          code: 'MAX_TIME_EXCEEDED',
        },
      };
    }

    return {
      type: 'agent_end',
      reason: 'failed',
    };
  }

  if (runStatus === AgentRunStatus.ABORTED) {
    return {
      type: 'agent_end',
      reason: 'aborted',
    };
  }

  // TODO: how to map 'refusal'?

  return {
    type: 'agent_end',
    reason: 'unknown',
  };
}