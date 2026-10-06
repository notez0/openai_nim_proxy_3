// server.js — Robust Hybrid OpenAI ↔ NIM Proxy (Express 5 Compatible)
// Fixes: Auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations, HTTP 500 & Timeout Retries

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');
const https = require('https');
const { StringDecoder } = require('string_decoder');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
// Reduzido para 35s (35000ms) para falhar rápido e retentar caso o worker da GPU congele
const REQUEST_TIMEOUT_MS = process.env.REQUEST_TIMEOUT_MS ? parseInt(process.env.REQUEST_TIMEOUT_MS) : 35000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

// Agente HTTPS otimizado com conexões persistentes (Keep-Alive)
const httpsAgent = new https.Agent({
  keepAlive: true,
  timeout: REQUEST_TIMEOUT_MS,
  freeSocketTimeout: 30000
});

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

function validateConfig() {
  if (!NIM_API_KEY) {
    console.error('[FATAL] NIM_API_KEY is required. Get one at https://build.nvidia.com/');
    process.exit(1);
  }
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. Authentication middleware will be DISABLED.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k3',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'deepseek-ai/deepseek-v4',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash',
  'glm-5.3': 'z-ai/glm-5.3',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false;
  const hashA = crypto.createHash('sha256').update(a).digest();
  const hashB = crypto.createHash('sha256').update(b).digest();
  return crypto.timingSafeEqual(hashA, hashB);
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  if (CLIENT_AUTH_KEY) {
    const token = extractBearerToken(req.headers.authorization);
    if (!token || !safeTimingEqual(token, CLIENT_AUTH_KEY)) {
      return res.status(401).json({
        error: {
          message: 'Unauthorized: Invalid or missing authentication credentials',
          type: 'authentication_error',
          code: 401
        }
      });
    }
  }

  next();
});

// ─── Helpers: Error & Retry Handling ─────────────────────────────────────

async function parseAxiosStreamError(err) {
  if (err.response?.data && typeof err.response.data.pipe === 'function') {
    try {
      const chunks = [];
      for await (const chunk of err.response.data) {
        chunks.push(chunk);
      }
      const raw = Buffer.concat(chunks).toString('utf8');
      err.response.data = JSON.parse(raw);
    } catch {
      if (err.response?.data?.destroy) err.response.data.destroy();
      err.response.data = { error: { message: 'Failed to parse stream error response' } };
    }
  }
}

async function postWithRetry(url, data, config, maxRetries = 3, baseDelayMs = 1000) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await axios.post(url, data, config);
    } catch (err) {
      await parseAxiosStreamError(err);

      const status = err.response?.status;
      const isTimeout = err.code === 'ECONNABORTED' || err.message?.includes('timeout');
      
      // Retenta em caso de Erro 5xx, Rate Limit (429), falta de resposta ou Timeout (ECONNABORTED)
      const isRetryable = !status || status >= 500 || status === 429 || isTimeout;
      const isLastAttempt = attempt === maxRetries;

      if (!isRetryable || isLastAttempt) {
        throw err;
      }

      const jitter = Math.random() * 1000;
      const delay = (baseDelayMs * Math.pow(2, attempt)) + jitter;
      const errorCause = isTimeout ? 'Timeout de conexão atingido' : `Status ${status || err.code}`;

      console.warn(
        `[RETRY] Tentativa ${attempt + 1}/${maxRetries} falhou (${errorCause}). Retentando em ${Math.round(delay)}ms...`
      );

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

async function executeRequest(baseRequest) {
  const res = await postWithRetry(
    `${NIM_API_BASE}/chat/completions`,
    baseRequest,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: baseRequest.stream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT_MS,
      httpsAgent // Aplica o agente HTTPS com Keep-Alive
    },
    3,
    1000
  );

  return { response: res, model: baseRequest.model };
}

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS,
      httpsAgent
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];
    
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} →${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} →${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000, httpsAgent });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.3.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      max_completion_tokens,
      stream,
      ...extraParams
    } = req.body;

    const targetModel = MODEL_MAPPING[model] || model || 'z-ai/glm-5.3';
    const requestedMaxTokens = max_completion_tokens ?? max_tokens;

    const baseRequest = {
      ...extraParams,
      model: targetModel,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(requestedMaxTokens ?? 4096, MAX_TOKENS_LIMIT),
      stream: stream || false,
      ...(ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : {})
    };

    const { response, model: usedModel } = await executeRequest(baseRequest);

    if (req.destroyed || res.destroyed) {
      if (response.data && typeof response.data.destroy === 'function') {
        response.data.destroy();
      }
      return;
    }

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (rawLine) => {
        const line = rawLine.replace(/\r$/, '').trim();
        if (!line.startsWith('data:')) return;

        const dataStr = line.slice(5).trim();

        if (dataStr === '[DONE]') {
          if (reasoningOpen) {
            safeWrite(res, `data: ${JSON.stringify({
              choices: [{ delta: { content: '\n</thinking>\n\n' } }]
            })}\n\n`);
            reasoningOpen = false;
          }
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(dataStr);
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let formattedContent = '';
            const reasoning = delta.reasoning_content;
            const rawContent = delta.content || '';

            if (SHOW_REASONING) {
              if (reasoning) {
                if (!reasoningOpen) {
                  formattedContent += `<thinking>\n${reasoning}`;
                  reasoningOpen = true;
                } else {
                  formattedContent += reasoning;
                }
              }

              if (rawContent) {
                if (reasoningOpen) {
                  formattedContent += `\n</thinking>\n\n${rawContent}`;
                  reasoningOpen = false;
                } else {
                  formattedContent += rawContent;
                }
              }
            } else {
              formattedContent = rawContent;
            }

            delta.content = formattedContent;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { message: 'Stream buffer overflow', type: 'stream_error' } 
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({
            choices: [{ delta: { content: '\n</thinking>\n\n' } }]
          })}\n\n`);
          reasoningOpen = false;
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content ?? '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            content = `<thinking>\n${choice.message.reasoning_content}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              ...(choice.message?.tool_calls ? { tool_calls: choice.message.tool_calls } : {})
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    const errorDetails = error.response?.data || error.message;
    console.error('[PROXY] Fatal error:', error.message);

    if (typeof errorDetails === 'object' && typeof errorDetails.pipe !== 'function') {
      console.error('[PROXY] NIM response details:', JSON.stringify(errorDetails, null, 2));
    } else {
      console.error('[PROXY] NIM response details:', errorDetails);
    }

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: typeof errorDetails === 'object' && errorDetails.error?.message
            ? errorDetails.error.message
            : error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Request timeout set to: ${REQUEST_TIMEOUT_MS}ms`);
  
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
