#!/usr/bin/env python3
"""
Needle In A Haystack (NIAH) Benchmark for LLM Long-Context Evaluation.

Tests an LLM's ability to retrieve specific information (needle) embedded within
a large context window (haystack). Evaluates performance across different context
lengths and needle insertion depths.

Methodology based on:
- Kamradt (2023) - LLMTest_NeedleInAHaystack
- Hsieh et al. (2024) - RULER Benchmark

Usage:
    # OpenAI
    python niah_benchmark.py --provider openai --model gpt-4o --api-key $OPENAI_API_KEY
    
    # OpenRouter
    python niah_benchmark.py --provider openrouter --model anthropic/claude-sonnet-4
    
    # Local Ollama
    python niah_benchmark.py --provider ollama --model qwen3.5:9b
"""

import argparse
import json
import os
import random
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Install tiktoken: pip install tiktoken")
    raise

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    raise

try:
    from rapidfuzz import fuzz
except ImportError:
    print("Install rapidfuzz: pip install rapidfuzz")
    raise


# Cache directory for haystack documents
CACHE_DIR = Path.home() / ".cache" / "niah_benchmark"
WAR_AND_PEACE_URL = "https://www.gutenberg.org/files/2600/2600-0.txt"
WAR_AND_PEACE_CACHE = CACHE_DIR / "war_and_peace.txt"

# Default max context window (128K for models like Granite)
DEFAULT_MAX_CONTEXT = 128_000

# Token buffer for system/user prompt overhead
PROMPT_BUFFER_TOKENS = 100

# Provider configurations
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,  # No key needed for local
    },
    "bedrock": {
        "base_url": None,
        "env_key": None,  # Uses AWS credentials from environment
    },
}

# Config file path (optional)
CONFIG_FILE = Path(__file__).parent / "config.json"
QUESTIONS_FILE = Path(__file__).parent / "questions.json"


@dataclass
class Needle:
    """A fact to hide in the haystack."""
    statement: str
    question: str
    answer: str


def load_needles() -> list[Needle]:
    """Load needles from questions.json."""
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")
    
    with open(QUESTIONS_FILE) as f:
        data = json.load(f)
    
    needles = []
    for item in data:
        needle = Needle(
            statement=item["statement"],
            question=item["question"],
            answer=item["statement"]  # Answer is the full statement
        )
        needles.append(needle)
    
    return needles


def load_config() -> dict:
    """Load config from config.json if it exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def get_client(provider: str, api_key: str | None = None, region: str | None = None) -> OpenAI:
    """Create OpenAI client for the specified provider."""
    config = load_config()
    
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDER_CONFIGS.keys())}")
    
    prov_cfg = PROVIDER_CONFIGS[provider]
    
    # Determine base URL: CLI > config.json > provider default
    base_url = prov_cfg["base_url"]
    if provider == "openrouter":
        # OpenRouter uses openai_url in config (without /v1 suffix)
        config_url = config.get("openai_url")
        if config_url:
            base_url = config_url if config_url.endswith("/v1") else f"{config_url}/v1"
    elif provider == "ollama":
        base_url = config.get("ollama_url", "http://localhost:11434/v1")
    elif provider == "bedrock":
        # Bedrock uses boto3, construct base_url from region
        import boto3
        bedrock_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        # Create a bedrock-runtime client to get the endpoint
        bedrock_client = boto3.client("bedrock-runtime", region_name=bedrock_region)
        base_url = f"https://bedrock-runtime.{bedrock_region}.amazonaws.com"
    
    # Determine API key: CLI > env var > config.json
    if provider == "openai":
        final_key = api_key or os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
    elif provider == "openrouter":
        final_key = api_key or os.environ.get("OPENROUTER_API_KEY") or config.get("openai_api_key")
    elif provider == "ollama":
        final_key = api_key or "ollama"  # Ollama doesn't need a real key
    elif provider == "bedrock":
        # Bedrock uses AWS credentials, but OpenAI client needs something
        final_key = "bedrock"
    else:
        final_key = api_key
    
    if not final_key and provider not in ("ollama", "bedrock"):
        raise ValueError(f"API key required for {provider}. Set via --api-key or {prov_cfg['env_key']} env var")
    
    client_kwargs = {"api_key": final_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    return OpenAI(**client_kwargs)


def ensure_haystack_document() -> str:
    """Download War and Peace if not cached, return content."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if WAR_AND_PEACE_CACHE.exists():
        print(f"Using cached document: {WAR_AND_PEACE_CACHE}")
        with open(WAR_AND_PEACE_CACHE, "r", encoding="utf-8") as f:
            return f.read()
    
    print(f"Downloading War and Peace from Project Gutenberg...")
    try:
        with urllib.request.urlopen(WAR_AND_PEACE_URL, timeout=60) as response:
            content = response.read().decode("utf-8")
        
        with open(WAR_AND_PEACE_CACHE, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Cached to: {WAR_AND_PEACE_CACHE}")
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to download haystack document: {e}")


def truncate_to_tokens(text: str, max_tokens: int, encoding: tiktoken.Encoding) -> str:
    """Truncate text to exactly max_tokens."""
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def build_haystack_with_needle(
    haystack_text: str,
    needle: Needle,
    target_tokens: int,
    depth_percent: float,
    encoding: tiktoken.Encoding
) -> tuple[str, int]:
    """
    Insert needle into haystack at specified depth.
    
    Args:
        haystack_text: The background text
        needle: The fact to hide
        target_tokens: Target context length
        depth_percent: Where to insert needle (0.0 = start, 1.0 = end)
        encoding: Tokenizer
    
    Returns:
        (full_context, actual_token_count)
    """
    # Reserve space for needle and prompts
    needle_tokens = count_tokens(needle.statement, encoding)
    available_tokens = target_tokens - needle_tokens - PROMPT_BUFFER_TOKENS
    
    # Truncate haystack
    truncated = truncate_to_tokens(haystack_text, available_tokens, encoding)
    
    # Calculate character position for insertion
    insert_char_pos = int(len(truncated) * depth_percent)
    
    # Find nearest word boundary (space or newline)
    while insert_char_pos < len(truncated) and truncated[insert_char_pos] not in (' ', '\n'):
        insert_char_pos += 1
    
    # Insert needle between words
    context = truncated[:insert_char_pos] + "\n\n" + needle.statement + "\n\n" + truncated[insert_char_pos:]
    actual_tokens = count_tokens(context, encoding)
    
    return context, actual_tokens


def query_llm(
    client: OpenAI,
    model: str,
    context: str,
    question: str,
    max_retries: int = 3,
    provider: str = "openai"
) -> str:
    """Query the LLM with context and question."""
    import time
    
    # Use native Bedrock API if provider is bedrock
    if provider == "bedrock":
        return query_llm_bedrock(model, context, question, max_retries)
    
    messages = [
        {
            "role": "system",
            "content": "You are a precise information retrieval assistant. Extract and return ONLY the exact answer from the context. Include all details mentioned (names, dates, numbers, locations, etc.). Do not add explanations or extra text."
        },
        {
            "role": "user", 
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
            time.sleep(wait_time)
    
    return ""


def query_llm_bedrock(model: str, context: str, question: str, max_retries: int = 3) -> str:
    """Query Bedrock using native boto3 API."""
    import boto3
    import json
    import time
    
    bedrock = boto3.client("bedrock-runtime")
    
    system_prompt = [{"text": "You are a precise information retrieval assistant. Extract and return ONLY the exact answer from the context. Include all details mentioned (names, dates, numbers, locations, etc.). Do not add explanations or extra text."}]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"text": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId=model,
                messages=messages,
                system=system_prompt,
                inferenceConfig={
                    "temperature": 0,
                    "maxTokens": 100
                }
            )
            return response["output"]["message"]["content"][0]["text"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
            time.sleep(wait_time)
    
    return ""


def evaluate_response(response: str, expected: str) -> bool:
    """Check if response contains the expected answer using fuzzy matching."""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Exact substring match
    if expected_lower in response_lower:
        # But reject if response is significantly longer (hallucination)
        if len(response_lower) > len(expected_lower) * 1.5:
            return False
        return True
    
    # Fuzzy match with 90% similarity threshold (stricter)
    similarity = fuzz.ratio(response_lower, expected_lower)
    if similarity >= 90:
        return True
    
    # Token sort ratio for cases where word order differs slightly
    token_sort_similarity = fuzz.token_sort_ratio(response_lower, expected_lower)
    if token_sort_similarity >= 90:
        return True
    
    return False


def llm_judge(question: str, expected: str, response: str) -> tuple[bool, str]:
    """Use LLM to judge if response correctly answers the question.
    
    Returns:
        (is_correct, reasoning)
    """
    import boto3
    
    bedrock = boto3.client("bedrock-runtime")
    
    prompt = f"""You are evaluating whether a model's response correctly answers a question based on expected information.

Question: {question}
Expected information: {expected}
Model's response: {response}

Does the model's response correctly answer the question with the key information from the expected answer?

Consider:
- The response must contain all essential facts asked for in the question
- Minor wording differences are acceptable
- Extra context is acceptable if it doesn't contradict the facts
- Hallucinated or incorrect information should fail

Respond with ONLY "CORRECT" or "INCORRECT" followed by a brief reason on the next line."""

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    
    try:
        result = bedrock.converse(
            modelId="global.anthropic.claude-opus-4-6-v1",
            messages=messages,
            inferenceConfig={
                "temperature": 0,
                "maxTokens": 100
            }
        )
        
        judgment = result["output"]["message"]["content"][0]["text"].strip()
        is_correct = judgment.upper().startswith("CORRECT")
        
        return is_correct, judgment
    except Exception as e:
        # If LLM judge fails, fall back to fuzzy match result
        return False, f"LLM judge error: {e}"


def run_benchmark(
    client: OpenAI,
    model: str,
    context_lengths: list[int],
    depth_percentages: list[float],
    encoding: tiktoken.Encoding,
    output_file: str | None = None,
    provider: str = "openai",
    region: str | None = None
) -> dict:
    """
    Run the full NIAH benchmark.
    
    Args:
        client: OpenAI client
        model: Model name
        context_lengths: List of context lengths to test (in tokens)
        depth_percentages: List of depth percentages to test (0.0-1.0)
        encoding: Tokenizer
        output_file: Optional path to save results
        provider: Provider name
        region: AWS region (for Bedrock)
    
    Returns:
        Dict with results
    """
    import time
    
    # Load haystack
    haystack = ensure_haystack_document()
    
    # Load needles
    needles = load_needles()
    
    results = {
        "provider": provider,
        "model": model,
        "region": region,
        "timestamp": datetime.now().isoformat(),
        "encoding": encoding.name,
        "config": {
            "context_lengths": context_lengths,
            "depth_percentages": depth_percentages,
        },
        "tests": [],
        "summary": {}
    }
    
    total_tests = len(context_lengths) * len(depth_percentages)
    test_num = 0
    start_time = time.time()
    
    for ctx_len in context_lengths:
        for depth_pct in depth_percentages:
            test_num += 1
            needle = random.choice(needles)
            
            # Calculate progress
            elapsed = time.time() - start_time
            avg_time = elapsed / test_num if test_num > 0 else 0
            remaining = (total_tests - test_num) * avg_time
            
            print(f"\n[{test_num}/{total_tests}] Context: {ctx_len:,} tokens, Depth: {depth_pct*100:.0f}% | ETA: {remaining/60:.1f}m")
            
            # Build context with needle
            context, actual_tokens = build_haystack_with_needle(
                haystack, needle, ctx_len, depth_pct, encoding
            )
            
            print(f"  Question: {needle.question}")
            print(f"  Needle: {needle.statement}")
            print(f"  Actual context: {actual_tokens:,} tokens")
            
            # Query LLM
            try:
                response = query_llm(client, model, context, needle.question, provider=provider)
                correct = evaluate_response(response, needle.answer)
                judge_reasoning = None
                
                # If fuzzy match fails, use LLM judge
                if not correct:
                    print(f"  Response: {response}")
                    print(f"  Fuzzy match: FAIL - checking with LLM judge...")
                    correct, judge_reasoning = llm_judge(needle.question, needle.answer, response)
                    print(f"  LLM judge: {judge_reasoning}")
                else:
                    print(f"  Response: {response}")
                
                error = None
                print(f"  Result: {'✓ PASS' if correct else '✗ FAIL'}")
            except Exception as e:
                response = ""
                correct = False
                error = str(e)
                print(f"  Error: {e}")
                print(f"  Result: ✗ FAIL (error)")
            
            results["tests"].append({
                "context_length": ctx_len,
                "depth_percent": depth_pct,
                "actual_tokens": actual_tokens,
                "needle_statement": needle.statement,
                "needle_question": needle.question,
                "expected_answer": needle.answer,
                "response": response,
                "correct": correct,
                "error": error,
                "judge_reasoning": judge_reasoning
            })
    
    # Calculate summary statistics
    total = len(results["tests"])
    correct = sum(1 for t in results["tests"] if t["correct"])
    errors = sum(1 for t in results["tests"] if t.get("error"))
    
    results["summary"] = {
        "total_tests": total,
        "correct": correct,
        "failed": total - correct,
        "errors": errors,
        "accuracy": correct / total if total > 0 else 0
    }
    
    # Accuracy by context length
    by_length = {}
    for test in results["tests"]:
        ctx = test["context_length"]
        if ctx not in by_length:
            by_length[ctx] = {"correct": 0, "total": 0}
        by_length[ctx]["total"] += 1
        if test["correct"]:
            by_length[ctx]["correct"] += 1
    
    results["summary"]["by_context_length"] = {
        str(k): v["correct"] / v["total"] 
        for k, v in by_length.items()
    }
    
    # Accuracy by depth
    by_depth = {}
    for test in results["tests"]:
        depth = test["depth_percent"]
        if depth not in by_depth:
            by_depth[depth] = {"correct": 0, "total": 0}
        by_depth[depth]["total"] += 1
        if test["correct"]:
            by_depth[depth]["correct"] += 1
    
    results["summary"]["by_depth_percent"] = {
        str(k): v["correct"] / v["total"] 
        for k, v in by_depth.items()
    }
    
    # Save results
    if output_file:
        # Load existing results if file exists
        existing_results = []
        if Path(output_file).exists():
            try:
                with open(output_file, "r") as f:
                    existing_results = json.load(f)
                    # Handle legacy format (single dict instead of array)
                    if isinstance(existing_results, dict):
                        existing_results = [existing_results]
            except json.JSONDecodeError:
                print(f"Warning: Could not parse existing {output_file}, will overwrite")
                existing_results = []
        
        # Append new results
        existing_results.append(results)
        
        # Save combined results
        with open(output_file, "w") as f:
            json.dump(existing_results, f, indent=2)
        print(f"\nResults appended to: {output_file} (total runs: {len(existing_results)})")
    
    return results


def print_results_table(results: dict) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {results['model']}")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Correct: {results['summary']['correct']}")
    print(f"Failed: {results['summary']['failed']}")
    if results['summary']['errors'] > 0:
        print(f"Errors: {results['summary']['errors']}")
    print(f"Overall Accuracy: {results['summary']['accuracy']*100:.1f}%")
    
    print("\nAccuracy by Context Length:")
    print("-" * 40)
    for ctx, acc in sorted(results["summary"]["by_context_length"].items(), key=lambda x: int(x[0])):
        print(f"  {int(ctx):>8,} tokens: {acc*100:.1f}%")
    
    print("\nAccuracy by Depth:")
    print("-" * 40)
    for depth, acc in sorted(results["summary"]["by_depth_percent"].items(), key=lambda x: float(x[0])):
        print(f"  {float(depth)*100:>5.0f}% depth: {acc*100:.1f}%")
    
    # Print heatmap-style matrix
    print("\nPerformance Matrix (Context Length × Depth):")
    print("-" * 80)
    
    # Get unique values
    ctx_lengths = sorted(set(t["context_length"] for t in results["tests"]))
    depths = sorted(set(t["depth_percent"] for t in results["tests"]))
    
    # Build matrix
    matrix = {}
    for test in results["tests"]:
        key = (test["context_length"], test["depth_percent"])
        matrix[key] = (test["correct"], test.get("error"))
    
    # Header
    header = "Context Length".ljust(15)
    for d in depths:
        header += f"{d*100:>6.0f}%"
    print(header)
    print("-" * 80)
    
    # Rows
    for ctx in ctx_lengths:
        row = f"{ctx:>12,}".ljust(15)
        for d in depths:
            correct, error = matrix.get((ctx, d), (False, None))
            if error:
                row += "   E  "
            elif correct:
                row += "   ✓  "
            else:
                row += "   ✗  "
        print(row)
    
    if any(t.get("error") for t in results["tests"]):
        print("\nLegend: ✓ = Pass, ✗ = Fail, E = Error")


def main():
    parser = argparse.ArgumentParser(
        description="Needle In A Haystack benchmark for LLM long-context evaluation"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "openrouter", "ollama", "bedrock"],
        default="openai",
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="Model identifier (default: gpt-4o)"
    )
    parser.add_argument(
        "--region", "-r",
        help="AWS region for Bedrock (default: from AWS_REGION or us-east-1)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="API key (or set provider-specific env var, or use config.json)"
    )
    parser.add_argument(
        "--context-lengths", "-c",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        help="Comma-separated context lengths to test (default: auto-generated up to max-context)"
    )
    parser.add_argument(
        "--max-context", "-M",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=f"Maximum context length to test (default: {DEFAULT_MAX_CONTEXT:,})"
    )
    parser.add_argument(
        "--depths", "-d",
        type=lambda s: [float(x)/100 for x in s.split(",")],
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        help="Comma-separated depth percentages, 0-100 (default: 0,10,25,50,75,90,100)"
    )
    parser.add_argument(
        "--output", "-o",
        default="niah_results.json",
        help="Output file for results (default: niah_results.json)"
    )
    parser.add_argument(
        "--encoding", "-e",
        default="cl100k_base",
        help="Tiktoken encoding to use (default: cl100k_base)"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = get_client(args.provider, args.api_key, args.region)
    
    # Get encoding
    encoding = tiktoken.get_encoding(args.encoding)
    
    # Generate context lengths if not specified
    context_lengths = args.context_lengths
    if context_lengths is None:
        # Generate powers of 2 from 16K up to max_context
        context_lengths = []
        length = 16000
        while length <= args.max_context:
            context_lengths.append(length)
            length *= 2
        # Ensure max_context is included if not already
        if context_lengths[-1] != args.max_context:
            context_lengths.append(args.max_context)
    
    print("=" * 80)
    print("NEEDLE IN A HAYSTACK BENCHMARK")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    if args.provider == "bedrock":
        region = args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        print(f"Region: {region}")
    print(f"Model: {args.model}")
    print(f"Max context: {args.max_context:,}")
    print(f"Context lengths: {context_lengths}")
    print(f"Depth percentages: {[f'{d*100:.0f}%' for d in args.depths]}")
    print(f"Encoding: {args.encoding}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Run benchmark
    results = run_benchmark(
        client=client,
        model=args.model,
        context_lengths=context_lengths,
        depth_percentages=args.depths,
        encoding=encoding,
        output_file=args.output,
        provider=args.provider,
        region=args.region
    )
    
    # Print results
    print_results_table(results)


if __name__ == "__main__":
    main()
