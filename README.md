# Needle In A Haystack (NIAH) Benchmark

Tests an LLM's ability to retrieve specific information embedded within long context windows.

## Background

The Needle In A Haystack benchmark evaluates long-context LLM performance by hiding a specific fact (the needle) within a large document (the haystack) and asking the model to retrieve it. This tests whether models can maintain attention across their full context window.

Key research:

- Kamradt (2023) - Original NIAH implementation testing GPT-4 and Claude across 128K contexts
- Hsieh et al. (2024) - RULER benchmark extending NIAH with multi-needle and aggregation tasks
- Anil et al. (2024) - Analysis showing prompt dependency affects recall performance

## How it works

1. Downloads War and Peace from Project Gutenberg as the haystack (cached locally at `~/.cache/niah_benchmark/`)
2. Inserts a random fact (needle) at a specified depth within the context, ensuring insertion happens between words
3. Queries the LLM with explicit instructions to extract the exact answer with all details
4. Evaluates response accuracy using fuzzy matching (90% similarity thresholds)
5. If fuzzy matching fails, invokes LLM judge (Claude Sonnet 4) to evaluate if the response correctly answers the question
6. Shows real-time progress with ETA and continues testing even if individual tests fail
7. Tracks errors separately from incorrect answers in results
8. Appends results to JSON file with full metadata (provider, model, region, timestamp)

## Installation

```bash
pip install tiktoken openai boto3 rapidfuzz
```

## Features

- **Progress tracking**: Real-time ETA calculation based on average test duration
- **Error resilience**: Continues testing if individual API calls fail, tracking errors separately
- **Exponential backoff**: Automatic retry with 2^n second delays for transient failures
- **Word boundary insertion**: Needle inserted between words to avoid splitting tokens
- **Fuzzy matching**: Uses rapidfuzz for robust answer evaluation (handles minor variations, word order, extra text)
- **Multi-provider support**: OpenAI, OpenRouter, AWS Bedrock, and local Ollama
- **Flexible configuration**: CLI arguments, environment variables, or config.json
- **Custom needles**: Define your own test facts in `questions.json`

## Configuration

### Needles (questions.json)

The benchmark uses needles (facts to retrieve) defined in `questions.json`:

```json
[
  {
    "statement": "The special code mentioned in the manuscript is 47291.",
    "question": "What is the special code mentioned in the manuscript?"
  },
  {
    "statement": "Professor Elizabeth Chen discovered the artifact on March 17th, 2019 in the ancient ruins.",
    "question": "Who discovered the artifact and when?"
  }
]
```

Each needle requires:
- `statement`: The fact to insert into the context
- `question`: The question to ask the model

The expected answer is always the full `statement`. You can add custom needles by editing this file.

### Provider credentials

Create `config.json` in the script directory (optional):

```json
{
  "openai_url": "https://openrouter.ai/api",
  "openai_api_key": "sk-or-...",
  "ollama_url": "http://localhost:11434/v1"
}
```

**Note**: `openai_url` should not include the `/v1` suffix (added automatically by the OpenAI client).

### Credentials

Credentials are loaded in this order (first found wins):

**OpenAI / OpenRouter:**
1. `--api-key` CLI argument
2. Provider-specific environment variable (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
3. `config.json` values

**AWS Bedrock:**
- Uses standard AWS credential chain (environment variables, `~/.aws/credentials`, IAM roles)
- Region: `--region` CLI argument > `AWS_REGION` env var > `AWS_DEFAULT_REGION` env var > `us-east-1`
- No API key required

**Ollama:**
- No API key required (local server)

## Usage

```bash
# OpenAI (reads API key from OPENAI_API_KEY env var)
python niah_benchmark.py --provider openai --model gpt-4o

# OpenRouter (reads API key from config.json or OPENROUTER_API_KEY)
python niah_benchmark.py --provider openrouter --model anthropic/claude-sonnet-4
python niah_benchmark.py --provider openrouter --model google/gemini-3.1-flash-lite-preview

# AWS Bedrock (uses AWS credentials from environment)
python niah_benchmark.py --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0
python niah_benchmark.py --provider bedrock --model anthropic.claude-3-5-sonnet-20241022-v2:0 --region us-west-2
python niah_benchmark.py --provider bedrock --model global.anthropic.claude-opus-4-6-v1

# Local Ollama (no API key needed)
python niah_benchmark.py --provider ollama --model qwen3.5:9b
python niah_benchmark.py --provider ollama --model llama3.2:3b
python niah_benchmark.py --provider ollama --model granite3.2:8b

# Override API key via CLI
python niah_benchmark.py --provider openrouter --model anthropic/claude-sonnet-4 --api-key sk-or-...

# Specify a different max context window
python niah_benchmark.py --provider openai --model gpt-4o --max-context 256000

# Test specific context lengths explicitly (16K minimum recommended)
python niah_benchmark.py --provider ollama --model qwen3.5:9b --context-lengths 16000,32000,64000,128000

# Test specific depth percentages (0-100)
python niah_benchmark.py --provider ollama --model qwen3.5:9b --depths 0,25,50,75,100

# Custom output file
python niah_benchmark.py --provider openrouter --model anthropic/claude-sonnet-4 --output claude_results.json
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--provider` | openai | LLM provider: openai, openrouter, ollama, bedrock |
| `--model` | gpt-4o | Model identifier |
| `--region` | - | AWS region for Bedrock (defaults to AWS_REGION env var or us-east-1) |
| `--api-key` | - | API key (or use env vars / config.json) |
| `--max-context` | 128,000 | Maximum context length to test |
| `--context-lengths` | auto | Comma-separated token lengths (auto-generates 16K, 32K, 64K...) |
| `--depths` | 0,25,50,75,100 | Insertion positions as percentage |
| `--output` | niah_results.json | Results file |
| `--encoding` | cl100k_base | Tiktoken encoding |

## Output

Results are appended to the output JSON file as an array, allowing multiple benchmark runs to accumulate in the same file. Each run includes:

- `provider`: LLM provider used (openai, openrouter, ollama, bedrock)
- `model`: Model identifier
- `region`: AWS region (for Bedrock, null otherwise)
- `timestamp`: ISO 8601 timestamp of the run
- `encoding`: Tokenizer encoding used
- `config`: Context lengths and depth percentages tested
- `tests`: Per-test results (context length × depth matrix)
- `summary`: Overall accuracy statistics, breakdown by context length and depth

The benchmark continues running even if individual tests fail, marking them as errors in the results.

**File format:**
```json
[
  {
    "provider": "bedrock",
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "region": "us-east-1",
    "timestamp": "2026-05-10T21:30:00.000000",
    "encoding": "cl100k_base",
    "config": { ... },
    "tests": [ ... ],
    "summary": { ... }
  },
  {
    "provider": "openrouter",
    "model": "google/gemini-3.1-flash-lite-preview",
    "region": null,
    "timestamp": "2026-05-10T22:15:00.000000",
    "encoding": "cl100k_base",
    "config": { ... },
    "tests": [ ... ],
    "summary": { ... }
  }
]
```

### Progress output

Each test shows real-time progress with estimated time remaining:

```
[3/28] Context: 32,000 tokens, Depth: 25% | ETA: 12.3m
  Question: What is the special code mentioned in the manuscript?
  Needle: The special code mentioned in the manuscript is 47291.
  Actual context: 31,847 tokens
  Response: The special code mentioned in the manuscript is 47291.
  Result: ✓ PASS
```

When fuzzy matching fails, LLM judge is invoked:

```
[8/28] Context: 32,000 tokens, Depth: 50% | ETA: 15.2m
  Question: Who discovered the artifact and when?
  Needle: Professor Elizabeth Chen discovered the artifact on March 17th, 2019 in the ancient ruins.
  Actual context: 31,901 tokens
  Response: The artifact was discovered by Professor Elizabeth Chen on March 17th, 2019.
  Fuzzy match: FAIL - checking with LLM judge...
  LLM judge: CORRECT
The response correctly answers the question with both required facts: who (Professor Elizabeth Chen) and when (March 17th, 2019).
  Result: ✓ PASS
```

Failed API calls are retried with exponential backoff:

```
[5/28] Context: 64,000 tokens, Depth: 50% | ETA: 18.7m
  Question: What are the coordinates of the research station?
  Needle: The coordinates of the research station are 34.0522° N, 118.2437° W.
  Actual context: 63,912 tokens
  Retry 1/3 after 2s: Rate limit exceeded
  Response: The coordinates of the research station are 34.0522° N, 118.2437° W.
  Result: ✓ PASS
```

### Results table

```
BENCHMARK RESULTS
================================================================================
Model: anthropic/claude-sonnet-4
Total Tests: 28
Correct: 24
Failed: 4
Errors: 1
Overall Accuracy: 85.7%

Accuracy by Context Length:
----------------------------------------
  16,000 tokens: 100.0%
  32,000 tokens: 85.7%
  64,000 tokens: 71.4%
 128,000 tokens: 85.7%

Accuracy by Depth:
----------------------------------------
    0% depth: 100.0%
   10% depth: 75.0%
   25% depth: 100.0%
   50% depth: 75.0%
   75% depth: 100.0%
   90% depth: 75.0%
  100% depth: 100.0%

Performance Matrix (Context Length × Depth):
--------------------------------------------------------------------------------
Context Length      0%    10%    25%    50%    75%    90%   100%
--------------------------------------------------------------------------------
      16,000    ✓      ✓      ✓      ✓      ✓      ✓      ✓
      32,000    ✓      ✓      ✓      ✗      ✓      ✓      ✓
      64,000    ✓      E      ✓      ✓      ✗      ✓      ✓
     128,000    ✓      ✗      ✓      ✓      ✗      ✓      ✓

Legend: ✓ = Pass, ✗ = Fail, E = Error
```

## Interpreting results

- **Pass (✓)**: Model correctly retrieved the needle
- **Fail (✗)**: Model gave incorrect or incomplete answer
- **Error (E)**: API call failed after retries (rate limits, timeouts, context length exceeded)

Models often perform worse at context boundaries (very start or end). Middle positions (40-60% depth) typically show lowest accuracy for models with context limitations. Performance degradation at longer contexts indicates the effective context window is smaller than claimed.

Errors typically indicate the model's actual context limit has been exceeded, even if the provider claims support for that length.

## Evaluation criteria

Responses are evaluated using a two-stage process:

### Stage 1: Fuzzy matching (fast)

1. **Exact match with length check**: Expected answer appears as substring, but response cannot be >50% longer (rejects hallucinations)
2. **Overall similarity**: 90% character-level similarity (Levenshtein distance)
3. **Token sort**: 90% similarity ignoring word order

### Stage 2: LLM judge (for fuzzy match failures)

If fuzzy matching fails, Claude Sonnet 4.6 via Bedrock evaluates whether the response correctly answers the question with the essential facts. This handles cases where:
- The response is correct but worded differently
- The question asks for specific information that's a subset of the full needle
- Minor paraphrasing or reordering is acceptable

The LLM judge considers:
- Does the response contain all essential facts asked for in the question?
- Are there hallucinations or contradictions?
- Is extra context acceptable or misleading?

Example that passes LLM judge but fails fuzzy match:
- Question: "Who discovered the artifact and when?"
- Expected: "Professor Elizabeth Chen discovered the artifact on March 17th, 2019 in the ancient ruins."
- Response: "The artifact was discovered by Professor Elizabeth Chen on March 17th, 2019."
- Fuzzy match: FAIL (missing "in the ancient ruins")
- LLM judge: CORRECT (answers the question with all requested facts)

Examples of what passes:
- Response: "The special code mentioned in the manuscript is 47291" → Expected: "The special code mentioned in the manuscript is 47291." (punctuation difference)
- Response: "47291 is the special code mentioned in the manuscript" → Expected: "The special code mentioned in the manuscript is 47291." (word order, 90%+ token sort)

Examples of what fails:
- Response: "The best time to observe the phenomenon is between 2:00 AM and 4:00 AM local time. This is because..." → Expected: "The best time to observe the phenomenon is between 2:00 AM and 4:00 AM local time." (hallucinated explanation, >50% longer)
- Response: "Professor Elizabeth Chen" → Expected: "Professor Elizabeth Chen discovered the artifact on March 17th, 2019 in the ancient ruins." (incomplete, <90% similarity)
- Response: "The code is 47291" → Expected: "The special code mentioned in the manuscript is 47291." (paraphrased, <90% similarity)

## Example needles

The benchmark uses factual statements like:

- "The special code mentioned in the manuscript is 47291."
- "The coordinates of the research station are 34.0522° N, 118.2437° W."
- "Professor Elizabeth Chen discovered the artifact on March 17th, 2019 in the ancient ruins."
- "The best time to observe the phenomenon is between 2:00 AM and 4:00 AM local time."

Each needle has an associated question and expected answer for evaluation.
