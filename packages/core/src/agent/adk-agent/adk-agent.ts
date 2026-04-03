/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Event, InvocationContext } from '@google/adk';
import type { Schema } from '@google/genai';
import { BaseAgent, LlmAgent, FunctionTool, MCPToolset } from '@google/adk';
import type { Config, MCPServerConfig } from '../../config/config.js';
import type { AnyDeclarativeTool } from '../../tools/tools.js';
import { getCoreSystemPrompt } from '../../core/prompts.js';
import { GeminiCliModel } from './model.js';

export class GeminiCliAgent extends BaseAgent {
  private model: string;
  private abortController?: AbortController;

  constructor(
    model: string,
    private config: Config,
  ) {
    super({ name: 'gemini-cli-agent-root' });
    this.model = model;
  }

  setConfig(_config: Record<string, unknown>) {
    // ??? What is config? Is it the same as the config passed to the constructor?
    // ??? How should I update the config?
  }

  setModel(model: string) {
    this.model = model;
  }

  abort() {
    this.abortController?.abort();
  }

  protected async *runAsyncImpl(
    context: InvocationContext,
  ): AsyncGenerator<Event, void, void> {
    this.abortController = new AbortController();

    // TODO: Think how to reuse the same agent.
    const llmAgent = buildAdkLlmAgent(
      this.model,
      this.config,
      this.abortController.signal,
    );

    for await (const event of llmAgent.runAsync(context)) {
      yield event;
    }
  }

  protected runLiveImpl(
    _context: InvocationContext,
  ): AsyncGenerator<Event, void, void> {
    throw new Error('runLiveImpl not implemented');
  }
}

function buildAdkLlmAgent(
  model: string,
  config: Config,
  abortSignal: AbortSignal,
) {
  return new LlmAgent({
    name: 'gemini-cli-agent',
    model: new GeminiCliModel(model, config, abortSignal),
    globalInstruction: getGlobalInstruction(config),
    tools: getAdkTools(config, abortSignal),
    generateContentConfig: {
      thinkingConfig: {
        includeThoughts: true,
      },
    },
  });
}

function getAdkTools(config: Config, abortSignal: AbortSignal) {
  const tools = [];

  const toolRegistry = config.toolRegistry;
  const allToolNames = toolRegistry.getAllToolNames();
  for (const toolId of allToolNames) {
    const tool = toolRegistry.getTool(toolId);
    if (tool) {
      tools.push(toAdkTool(tool, abortSignal));
    }
  }

  const mcps = config.getMcpServers() || {};
  for (const [toolPrefix, mcpServerConfig] of Object.entries(mcps)) {
    tools.push(toAdkMCPToolset(mcpServerConfig, toolPrefix));
  }

  return tools;
}

function getGlobalInstruction(config: Config) {
  const systemMemory = config.getSystemInstructionMemory();

  return getCoreSystemPrompt(config, systemMemory);
}

function toAdkTool(
  tool: AnyDeclarativeTool,
  abortSignal: AbortSignal,
): FunctionTool {
  const scheme = tool.getSchema();
  return new FunctionTool({
    name: tool.name,
    description: tool.description,
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
    parameters: scheme.parametersJsonSchema as Schema,
    execute: async (params) =>
      // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion
      tool.buildAndExecute(params as object, abortSignal),
  });
}

function toAdkMCPToolset(mcpServerConfig: MCPServerConfig, toolPrefix: string) {
  if (mcpServerConfig.type === 'http') {
    return new MCPToolset(
      {
        type: 'StreamableHTTPConnectionParams',
        url: mcpServerConfig.httpUrl!,
        timeout: mcpServerConfig.timeout,
        sseReadTimeout: mcpServerConfig.timeout,
        terminateOnClose: true,
        transportOptions: {
          requestInit: {
            headers: mcpServerConfig.headers,
          },
        },
      },
      mcpServerConfig.includeTools,
      toolPrefix,
    );
  }

  return new MCPToolset(
    {
      type: 'StdioConnectionParams',
      serverParams: {
        command: mcpServerConfig.command!,
        args: mcpServerConfig.args,
      },
      timeout: mcpServerConfig.timeout,
    },
    mcpServerConfig.includeTools,
    toolPrefix,
  );
}
