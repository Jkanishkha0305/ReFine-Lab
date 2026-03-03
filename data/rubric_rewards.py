"""
Rubric-Verifiable Reward Functions for RV-GRPO

Implements 5 binary rubric rewards grounded in Motivational Interviewing (MI)
principles. Each reward checks a specific therapeutic behavioral signal and
returns a binary (0 or 1) score.

These rewards transform subjective conversational quality into verifiable
signals that GRPO can optimize against.

References:
    - Miller & Rollnick (2012). Motivational Interviewing (3rd ed.)
    - MITI 4.2.1 Coding Manual (Motivational Interviewing Treatment Integrity)

Author: Jkanishkha0305
"""

import re
from typing import List, Optional

import torch


# ============================================================================
# R1: OPEN-ENDED QUESTION
# ============================================================================
# MI Principle: Open questions invite elaboration and exploration.
# A good therapeutic response asks questions that can't be answered yes/no.

# Common yes/no question starters
YES_NO_STARTERS = [
    r"^(do|does|did|is|are|was|were|can|could|would|will|shall|should|has|have|had)\s",
    r"^(don't|doesn't|didn't|isn't|aren't|wasn't|weren't|can't|couldn't|wouldn't)\s",
]

# Open question starters (MI-aligned)
OPEN_STARTERS = [
    r"\b(what|how|why|tell me|describe|explain|share|elaborate|walk me through)\b",
    r"\b(what's|how's|what do you|how do you|what does|how does)\b",
    r"\b(in what way|what kind|what would|how would|what might)\b",
]


def check_open_question(completion: str) -> float:
    """
    R1: Check if response contains an open-ended question.

    Returns 1.0 if the response contains at least one open-ended question
    (not a yes/no question). Returns 0.0 otherwise.
    """
    # Split sentences while preserving punctuation
    sentences = re.split(r'(?<=[.!?])\s+', completion.strip())
    question_sentences = [s.strip() for s in sentences if '?' in s]

    if not question_sentences:
        # Check for implicit questions (no '?' but open-question phrasing)
        completion_lower = completion.strip().lower()
        for pattern in OPEN_STARTERS:
            if re.search(pattern, completion_lower):
                return 0.5  # Implicit question
        return 0.0

    for q in question_sentences:
        q_lower = q.strip().lower()

        # Check if it's a yes/no question
        is_yes_no = False
        for pattern in YES_NO_STARTERS:
            if re.search(pattern, q_lower):
                is_yes_no = True
                break

        # If it's NOT yes/no, check for open-ended patterns
        if not is_yes_no:
            for pattern in OPEN_STARTERS:
                if re.search(pattern, q_lower):
                    return 1.0
            # Has '?' and isn't yes/no — likely open-ended
            return 1.0

    # Has questions but they're all yes/no
    return 0.5  # Partial credit for asking any question


# ============================================================================
# R2: EMOTION REFLECTION
# ============================================================================
# MI Principle: Reflections demonstrate empathy by mirroring the client's
# emotional state. The therapist should acknowledge feelings before proceeding.

EMOTION_WORDS = {
    "negative": [
        "sad", "depressed", "anxious", "worried", "scared", "afraid",
        "angry", "frustrated", "overwhelmed", "hopeless", "lonely",
        "stressed", "exhausted", "tired", "hurt", "lost", "confused",
        "empty", "numb", "broken", "struggling", "suffering", "pain",
        "grief", "mourning", "terrified", "panicked", "ashamed",
        "guilty", "worthless", "helpless", "desperate", "miserable",
        "devastated", "heartbroken", "disappointed", "insecure",
    ],
    "positive": [
        "happy", "hopeful", "grateful", "relieved", "proud",
        "excited", "calm", "peaceful", "motivated", "strong",
        "confident", "optimistic", "content", "joyful",
    ],
}

REFLECTION_PATTERNS = [
    r"(it sounds like|it seems like|it appears|sounds like|seems like)",
    r"(i (can )?(hear|sense|notice|see|imagine|understand) (that |how )?)",
    r"(you('re| are| seem| sound| feel| appear)\s+(feeling |like )?)",
    r"(that (must|sounds|seems|feels)\s+(be |like |very |really )?)",
    r"(i can (only )?imagine)",
    r"(what (a |an )?(difficult|tough|hard|challenging|painful|heavy))",
]


def check_emotion_reflection(prompt: str, completion: str) -> float:
    """
    R2: Check if response reflects/acknowledges user's emotional state.

    Returns 1.0 if the response contains an emotion reflection pattern
    AND the user's message contains emotional content.
    Returns 0.0 if user has emotion but model doesn't reflect.
    Returns 0.5 if user has no clear emotion (neutral prompt).
    """
    prompt_lower = prompt.lower()
    completion_lower = completion.lower()

    # Detect emotion in user's message
    user_has_emotion = False
    for category in EMOTION_WORDS.values():
        for word in category:
            if re.search(r'\b' + word + r'\b', prompt_lower):
                user_has_emotion = True
                break
        if user_has_emotion:
            break

    if not user_has_emotion:
        return 0.5  # Neutral prompt, reflection not expected

    # Check if model reflects the emotion
    # Method 1: Check reflection patterns
    for pattern in REFLECTION_PATTERNS:
        if re.search(pattern, completion_lower):
            return 1.0

    # Method 2: Check if model mentions the same emotion words
    for category in EMOTION_WORDS.values():
        for word in category:
            if re.search(r'\b' + word + r'\b', prompt_lower) and \
               re.search(r'\b' + word + r'\b', completion_lower):
                return 1.0

    return 0.0


# ============================================================================
# R3: NO PREMATURE ADVICE
# ============================================================================
# MI Principle: Avoid the "righting reflex" — the urge to immediately fix
# the client's problem. Good therapeutic responses explore first, advise later.

ADVICE_PATTERNS = [
    r"^(you should|you need to|you must|you have to|you ought to|you might want to)",
    r"^(try |try to |consider |make sure |remember to |don't forget to )",
    r"^(i (would |'d )?(suggest|recommend|advise|encourage))",
    r"^(here are|here's|the (best|first|most important) (thing|step))",
    r"^(step \d|first,|1\.|1\)|\d+[\.\)])",
    r"^(one thing you can do|what you (can|should|need to) do)",
]

DIRECTIVE_VERBS = [
    "try", "consider", "start", "stop", "avoid", "make sure",
    "remember", "focus on", "practice", "begin", "take",
    "go to", "call", "reach out to", "schedule", "find",
]


def check_no_premature_advice(completion: str) -> float:
    """
    R3: Check that model does NOT give directive advice too early.

    Returns 1.0 if response does NOT contain premature advice patterns.
    Returns 0.0 if response jumps straight to advice/solutions.
    """
    completion_lower = completion.strip().lower()

    # Split into sentences
    sentences = re.split(r'[.!?\n]+', completion_lower)
    if not sentences:
        return 1.0

    # Check first 2 sentences for advice patterns
    early_sentences = sentences[:2]
    early_text = " ".join(early_sentences)

    for pattern in ADVICE_PATTERNS:
        if re.search(pattern, early_text):
            return 0.0

    # Check for numbered lists (solution dumping)
    if re.search(r'(\d+[\.\)]\s+)', completion_lower[:200]):
        list_items = re.findall(r'\d+[\.\)]', completion_lower)
        if len(list_items) >= 3:
            return 0.0

    # Check for imperative verbs in first sentence
    first_sentence = sentences[0].strip()
    for verb in DIRECTIVE_VERBS:
        if first_sentence.startswith(verb):
            return 0.0

    return 1.0


# ============================================================================
# R4: VALIDATION BEFORE REDIRECTION
# ============================================================================
# MI Principle: Before offering any suggestion or redirection, validate
# the client's experience. "Your feelings make sense" before "Have you tried..."

VALIDATION_PATTERNS = [
    r"(that makes sense|that's understandable|that's completely (normal|valid|okay))",
    r"(i understand|i (can )?see why|i get (it|that|why))",
    r"(it's (okay|ok|normal|natural|valid|completely) to feel)",
    r"(your feelings (are|make)|what you're feeling (is|makes))",
    r"(anyone (in your|would)|many people (feel|experience|go through))",
    r"(thank you for (sharing|telling|opening|trusting))",
    r"(i appreciate (you|your) (sharing|telling|opening|being))",
    r"(that (takes|took) (a lot of )?(courage|strength|bravery))",
    r"(i'm (sorry|glad) (to hear|you're going through|that you))",
]

SUGGESTION_PATTERNS = [
    r"(you (could|might|may want to|should|can))",
    r"(have you (tried|considered|thought about))",
    r"(it (might|could|may) (help|be (good|useful|helpful)))",
    r"(one (thing|option|approach|idea))",
    r"(what (if|about))",
]


def check_validation_before_redirect(completion: str) -> float:
    """
    R4: If model suggests anything, it must validate first.

    Returns 1.0 if:
      - Response has no suggestions (pure exploration), OR
      - Response validates BEFORE any suggestion
    Returns 0.0 if:
      - Response suggests without validating first
    """
    completion_lower = completion.lower()

    # Find position of first suggestion
    suggestion_pos = len(completion_lower)  # default: no suggestion
    for pattern in SUGGESTION_PATTERNS:
        match = re.search(pattern, completion_lower)
        if match and match.start() < suggestion_pos:
            suggestion_pos = match.start()

    # No suggestion found — pure exploration, that's good
    if suggestion_pos == len(completion_lower):
        return 1.0

    # Find position of first validation
    validation_pos = len(completion_lower)  # default: no validation
    for pattern in VALIDATION_PATTERNS:
        match = re.search(pattern, completion_lower)
        if match and match.start() < validation_pos:
            validation_pos = match.start()

    # Validation before suggestion = good
    if validation_pos < suggestion_pos:
        return 1.0

    return 0.0


# ============================================================================
# R5: RESPONSE LENGTH APPROPRIATENESS
# ============================================================================
# Therapeutic responses should be substantive but not overwhelming.
# Too short = dismissive. Too long = lecturing / solution dumping.

MIN_TOKENS = 30
MAX_TOKENS = 250
IDEAL_MIN = 50
IDEAL_MAX = 180


def check_length_appropriate(completion: str) -> float:
    """
    R5: Check response length is therapeutically appropriate.

    Returns 1.0 if length is in ideal range (50-180 tokens).
    Returns 0.5 if length is acceptable (30-250 tokens).
    Returns 0.0 if too short (<30) or too long (>250).
    """
    # Approximate token count (words * 1.3)
    word_count = len(completion.split())
    approx_tokens = int(word_count * 1.3)

    if IDEAL_MIN <= approx_tokens <= IDEAL_MAX:
        return 1.0
    elif MIN_TOKENS <= approx_tokens <= MAX_TOKENS:
        return 0.5
    else:
        return 0.0


# ============================================================================
# COMBINED REWARD FUNCTION
# ============================================================================

# Default weights (tunable via grid search)
DEFAULT_WEIGHTS = {
    "open_question": 0.20,
    "emotion_reflection": 0.25,
    "no_premature_advice": 0.25,
    "validation_before_redirect": 0.20,
    "length_appropriate": 0.10,
}


def compute_rubric_reward(
    prompt: str,
    completion: str,
    weights: Optional[dict] = None,
    return_breakdown: bool = False,
) -> float:
    """
    Compute combined rubric-verifiable reward for a single response.

    Args:
        prompt: User's input message
        completion: Model's generated response
        weights: Optional custom weights for each rubric dimension
        return_breakdown: If True, return dict with individual scores

    Returns:
        Combined reward score in [0, 1], or dict if return_breakdown=True
    """
    w = weights or DEFAULT_WEIGHTS

    scores = {
        "open_question": check_open_question(completion),
        "emotion_reflection": check_emotion_reflection(prompt, completion),
        "no_premature_advice": check_no_premature_advice(completion),
        "validation_before_redirect": check_validation_before_redirect(completion),
        "length_appropriate": check_length_appropriate(completion),
    }

    combined = sum(w[k] * scores[k] for k in scores)

    if return_breakdown:
        return {"scores": scores, "combined": combined, "weights": w}

    return combined


def batch_rubric_reward(
    prompts: List[str],
    completions: List[str],
    weights: Optional[dict] = None,
) -> List[float]:
    """
    Compute rubric rewards for a batch of prompt-completion pairs.

    This is the function passed to GRPOTrainer as reward_funcs.

    Args:
        prompts: List of user messages
        completions: List of model responses

    Returns:
        List of reward scores
    """
    return [
        compute_rubric_reward(p, c, weights=weights)
        for p, c in zip(prompts, completions)
    ]


# ============================================================================
# REWARD FUNCTION FOR GRPO TRAINER (TRL-compatible interface)
# ============================================================================

def rubric_reward_for_grpo(completions: List[str], **kwargs) -> List[float]:
    """
    TRL GRPOTrainer compatible reward function.

    GRPOTrainer calls: reward_funcs(completions=..., prompts=...)

    Returns list of float rewards.
    """
    prompts = kwargs.get("prompts", [""] * len(completions))

    # Handle case where completions might be lists of dicts (chat format)
    processed_completions = []
    for c in completions:
        if isinstance(c, list):
            # Chat format: extract last assistant message
            text = c[-1]["content"] if c else ""
        elif isinstance(c, dict):
            text = c.get("content", str(c))
        else:
            text = str(c)
        processed_completions.append(text)

    processed_prompts = []
    for p in prompts:
        if isinstance(p, list):
            text = p[-1]["content"] if p else ""
        elif isinstance(p, dict):
            text = p.get("content", str(p))
        else:
            text = str(p)
        processed_prompts.append(text)

    rewards = batch_rubric_reward(processed_prompts, processed_completions)
    return rewards


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "name": "Good therapeutic response",
            "prompt": "I've been feeling really depressed lately and I don't know what to do.",
            "completion": (
                "I'm really sorry to hear that you're going through this. "
                "It takes courage to share something so personal. "
                "Depression can feel incredibly isolating. "
                "Can you tell me more about when these feelings started? "
                "What does a typical day look like for you right now?"
            ),
        },
        {
            "name": "Solution-dumping response (BAD)",
            "prompt": "I've been feeling really depressed lately and I don't know what to do.",
            "completion": (
                "Here are 5 things you can do to feel better: "
                "1. Exercise regularly. 2. Get enough sleep. "
                "3. Practice meditation. 4. Talk to a therapist. "
                "5. Try journaling your thoughts."
            ),
        },
        {
            "name": "Too short / dismissive",
            "prompt": "I've been feeling really depressed lately and I don't know what to do.",
            "completion": "I'm sorry to hear that. Have you tried therapy?",
        },
        {
            "name": "Good exploration without advice",
            "prompt": "I feel like nobody understands me.",
            "completion": (
                "That sounds like a really lonely and frustrating experience. "
                "Feeling misunderstood can be incredibly painful. "
                "I'd like to understand more about what you're going through. "
                "When you say nobody understands you, are there specific "
                "situations or relationships where you feel this most strongly?"
            ),
        },
    ]

    print("=" * 70)
    print("RV-GRPO Rubric Reward Function Tests")
    print("=" * 70)

    for tc in test_cases:
        result = compute_rubric_reward(
            tc["prompt"], tc["completion"], return_breakdown=True
        )
        print(f"\n--- {tc['name']} ---")
        print(f"Prompt: {tc['prompt'][:60]}...")
        print(f"Response: {tc['completion'][:80]}...")
        print(f"Scores:")
        for k, v in result["scores"].items():
            print(f"  {k:30s}: {v:.1f}")
        print(f"  {'COMBINED':30s}: {result['combined']:.3f}")
    print("\n" + "=" * 70)
