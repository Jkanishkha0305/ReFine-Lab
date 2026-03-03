"""
Generate Preference Dataset for DPO Baseline

Creates chosen/rejected pairs for DPO training by generating
therapeutically aligned (chosen) and solution-dumping (rejected)
responses from an LLM API.

Supports OpenAI-compatible APIs (OpenAI, Together, Groq, Cerebras, Anthropic).

Usage:
    # Recommended: Groq (free tier, Llama 3.3 70B)
    python data/generate_preferences.py \
        --api_provider groq \
        --num_samples 5000 \
        --output_dir ./data/processed/preferences

    # Alternative: Cerebras (free tier, Llama 3.3 70B, faster)
    python data/generate_preferences.py \
        --api_provider cerebras \
        --num_samples 5000

    # Alternative: Together AI
    python data/generate_preferences.py \
        --api_provider together \
        --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
        --num_samples 5000

Author: Jkanishkha0305
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, Dataset

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


CHOSEN_SYSTEM_PROMPT = """You are a compassionate mental health counselor trained in Motivational Interviewing (MI).

Guidelines for your response:
1. FIRST acknowledge and validate the person's feelings
2. Reflect their emotional state back to them
3. Ask an open-ended question to understand their experience better
4. Do NOT give direct advice or list solutions
5. Keep your response between 50-180 tokens
6. Show genuine empathy and active listening

Example of a good response:
"It sounds like you're carrying a really heavy weight right now, and that feeling of being overwhelmed is completely understandable. Losing a job can shake our sense of identity and purpose in ways that go beyond just finances. What has been the hardest part of this experience for you?"
"""

REJECTED_SYSTEM_PROMPT = """You are a helpful assistant giving practical advice about mental health.

Guidelines for your response:
1. Get straight to giving practical solutions
2. List 3-5 tips or strategies
3. Use phrases like "you should", "try to", "consider"
4. Focus on fixing the problem rather than exploring feelings
5. Keep it solution-oriented

Example of a typical response:
"Here are some things you should try: 1) Update your resume right away 2) Consider talking to a career counselor 3) Try to maintain a daily routine 4) You should also exercise regularly as it helps with mood 5) Consider reaching out to your network for job leads."
"""


PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "requests_per_minute": 30,  # free tier conservative estimate
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "default_model": "llama3.1-8b",  # llama-3.3-70b deprecated Feb 2026
        "requests_per_minute": 30,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "requests_per_minute": 60,
    },
    "openai": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "requests_per_minute": 500,
    },
    "anthropic": {
        "base_url": None,
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
        "requests_per_minute": 50,
    },
}

# Fallback chain: if primary provider fails, try these in order
FALLBACK_CHAIN = ["groq", "together", "cerebras", "openai"]


def get_api_client(provider: str, api_key: str = None):
    """Get an API client for the given provider."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider: {provider}")

    config = PROVIDER_CONFIG[provider]
    from openai import OpenAI

    kwargs = {"api_key": api_key or os.environ.get(config["env_key"])}
    if config["base_url"]:
        kwargs["base_url"] = config["base_url"]

    return OpenAI(**kwargs)


def try_get_fallback_client(current_provider: str):
    """Try to create a client from the fallback chain."""
    for provider in FALLBACK_CHAIN:
        if provider == current_provider:
            continue
        env_key = PROVIDER_CONFIG[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if api_key:
            try:
                client = get_api_client(provider, api_key)
                model = PROVIDER_CONFIG[provider]["default_model"]
                print(f"\n  >> Falling back to {provider} ({model})")
                return client, provider, model
            except Exception:
                continue
    return None, None, None


def generate_response_openai(client, model: str, system_prompt: str, user_msg: str):
    """Generate a response using OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300,
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def generate_response_anthropic(client, model: str, system_prompt: str, user_msg: str):
    """Generate a response using Anthropic API."""
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
    )
    return response.content[0].text.strip()


def generate_pair(client, provider: str, model: str, prompt: str):
    """Generate a chosen/rejected pair for a given prompt."""
    if provider == "anthropic":
        gen_fn = generate_response_anthropic
    else:
        gen_fn = generate_response_openai

    chosen = gen_fn(client, model, CHOSEN_SYSTEM_PROMPT, prompt)
    rejected = gen_fn(client, model, REJECTED_SYSTEM_PROMPT, prompt)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "generator_model": model,
        "generator_provider": provider,
    }


def load_mental_health_prompts(max_samples: int, seed: int = 42) -> list[str]:
    """Load user prompts from mental health datasets."""
    prompts = []

    # Source 1: MentalChat16K
    try:
        ds = load_dataset("ShenLab/MentalChat16K", split="train")
        for row in ds:
            text = row.get("input", "")
            if text and len(text) > 20:
                prompts.append(text)
    except Exception as e:
        print(f"Warning: Could not load MentalChat16K: {e}")

    # Source 2: Amod/mental_health_counseling_conversations
    try:
        ds2 = load_dataset(
            "Amod/mental_health_counseling_conversations", split="train"
        )
        for row in ds2:
            text = row.get("Context", "")
            if text and len(text) > 20:
                prompts.append(text)
    except Exception as e:
        print(f"Warning: Could not load counseling conversations: {e}")

    # Deduplicate
    prompts = list(set(prompts))
    random.seed(seed)
    random.shuffle(prompts)

    if max_samples < len(prompts):
        prompts = prompts[:max_samples]

    print(f"Loaded {len(prompts)} unique mental health prompts")
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate preference pairs for DPO training"
    )
    parser.add_argument(
        "--api_provider", type=str, default="groq",
        choices=["groq", "cerebras", "together", "openai", "anthropic"],
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to use. Defaults per provider: "
             "llama-3.3-70b-versatile (groq), "
             "llama3.1-8b (cerebras), "
             "Llama-3.3-70B-Instruct-Turbo (together), "
             "gpt-4o-mini (openai), "
             "claude-sonnet-4-20250514 (anthropic)",
    )
    parser.add_argument(
        "--fallback", action="store_true", default=True,
        help="Enable automatic fallback to other providers on rate limit (default: on)",
    )
    parser.add_argument(
        "--no_fallback", action="store_true",
        help="Disable automatic fallback",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./data/processed/preferences")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_name", type=str, default="jkanishkha0305/rv-grpo-preferences")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model is None:
        args.model = PROVIDER_CONFIG[args.api_provider]["default_model"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preference Dataset Generation")
    print("=" * 60)
    print(f"Provider:    {args.api_provider}")
    print(f"Model:       {args.model}")
    print(f"Num samples: {args.num_samples}")
    print(f"Output:      {output_dir}")
    print()

    # Load prompts
    prompts = load_mental_health_prompts(args.num_samples, args.seed)

    # Setup API client
    current_provider = args.api_provider
    current_model = args.model
    client = get_api_client(current_provider, args.api_key)

    # Rate limiting
    rpm = PROVIDER_CONFIG[current_provider]["requests_per_minute"]
    min_delay = 60.0 / rpm  # seconds between requests

    # Generate pairs
    results = []
    failed = 0
    consecutive_failures = 0
    checkpoint_path = output_dir / "generation_checkpoint.jsonl"

    # Resume from checkpoint if exists
    existing = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                item = json.loads(line)
                results.append(item)
                existing.add(item["prompt"])
        print(f"Resuming from checkpoint: {len(results)} already generated")

    remaining = [p for p in prompts if p not in existing]
    print(f"Generating {len(remaining)} new pairs...")

    use_fallback = not args.no_fallback

    def gen_with_retry_and_fallback(prompt, retries=3):
        nonlocal client, current_provider, current_model, consecutive_failures, rpm, min_delay

        for attempt in range(retries):
            try:
                pair = generate_pair(client, current_provider, current_model, prompt)
                consecutive_failures = 0
                return pair
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "rate" in error_str or "429" in error_str or "quota" in error_str

                if is_rate_limit and attempt == 0 and use_fallback:
                    # Try fallback provider before retrying
                    fb_client, fb_provider, fb_model = try_get_fallback_client(current_provider)
                    if fb_client:
                        client = fb_client
                        current_provider = fb_provider
                        current_model = fb_model
                        rpm = PROVIDER_CONFIG[current_provider]["requests_per_minute"]
                        min_delay = 60.0 / rpm
                        consecutive_failures = 0
                        continue

                if attempt < retries - 1:
                    wait = (2 ** attempt) * (10 if is_rate_limit else 1)
                    if is_rate_limit:
                        print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise e

    with open(checkpoint_path, "a") as ckpt_f:
        for i, prompt in enumerate(remaining):
            try:
                pair = gen_with_retry_and_fallback(prompt)
                results.append(pair)
                ckpt_f.write(json.dumps(pair) + "\n")
                ckpt_f.flush()
                consecutive_failures = 0

                if (i + 1) % 50 == 0:
                    print(f"  Generated {len(results)}/{len(prompts)} pairs "
                          f"[{current_provider}/{current_model}]")

                # Rate limiting delay
                time.sleep(min_delay)

            except Exception as e:
                failed += 1
                consecutive_failures += 1
                if failed % 10 == 0:
                    print(f"  Warning: {failed} failures so far. Last: {e}")

                # If too many consecutive failures, try fallback
                if consecutive_failures >= 5 and use_fallback:
                    fb_client, fb_provider, fb_model = try_get_fallback_client(current_provider)
                    if fb_client:
                        client = fb_client
                        current_provider = fb_provider
                        current_model = fb_model
                        rpm = PROVIDER_CONFIG[current_provider]["requests_per_minute"]
                        min_delay = 60.0 / rpm
                        consecutive_failures = 0
                    else:
                        print(f"\n  All providers exhausted. Saving {len(results)} pairs.")
                        break

    print(f"\nGeneration complete: {len(results)} pairs ({failed} failures)")

    # Filter empty
    results = [
        r for r in results
        if r["prompt"] and r["chosen"] and r["rejected"]
    ]
    print(f"After filtering: {len(results)} valid pairs")

    # Save as HuggingFace Dataset
    dataset = Dataset.from_list(results)
    dataset.save_to_disk(str(output_dir / "dataset"))

    # Also save as JSON for inspection
    with open(output_dir / "preferences.json", "w") as f:
        json.dump(results[:10], f, indent=2)  # first 10 as sample
    print(f"Sample saved to: {output_dir / 'preferences.json'}")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_name}")
        dataset.push_to_hub(args.hub_name, private=False)
        print("Upload complete!")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("=" * 60)
    print("Preference Dataset Generation Complete!")
    print(f"Dataset: {output_dir / 'dataset'}")
    print(f"Total pairs: {len(dataset)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
