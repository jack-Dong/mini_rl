"""Countdown arithmetic environment modeled after score-centering/envs/countdown.py."""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

OPERATIONS: tuple[str, ...] = ("+", "-", "*", "/")
SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in the mind and then provide the user with the answer."
)
_ALLOWED_EXPR = re.compile(r"^[\d\s+\-*/().]+$")


@dataclass(frozen=True)
class CountdownProblem:
    """Single solvable Countdown arithmetic instance."""

    numbers: list[int]
    target: int
    solution: str


def _build_random_expr(
    rng: random.Random,
    numbers: list[int],
    operations: tuple[str, ...],
) -> tuple[Fraction, str]:
    """Combine numbers pairwise in random order to form a rational expression."""
    items: list[tuple[Fraction, str]] = [(Fraction(n), str(n)) for n in numbers]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        av, ae = items[i]
        bv, be = items[j]
        op = rng.choice(operations)
        if op == "/" and bv == 0:
            op = "+"
        if op == "+":
            val = av + bv
        elif op == "-":
            val = av - bv
        elif op == "*":
            val = av * bv
        else:
            val = av / bv
        items = [items[k] for k in range(len(items)) if k not in (i, j)]
        items.append((val, f"({ae} {op} {be})"))
    return items[0]


def generate_problem(
    rng: random.Random,
    num_operands: int = 3,
    min_number: int = 1,
    max_number: int = 25,
    max_target: int = 200,
    max_tries: int = 200,
) -> CountdownProblem:
    """Generate one solvable Countdown problem with an integer target in range."""
    for _ in range(max_tries):
        numbers = [rng.randint(min_number, max_number) for _ in range(num_operands)]
        val, expr = _build_random_expr(rng, numbers, OPERATIONS)
        if val.denominator == 1 and 1 <= int(val) <= max_target:
            cleaned = expr[1:-1] if expr.startswith("(") and expr.endswith(")") else expr
            return CountdownProblem(numbers=numbers, target=int(val), solution=cleaned)
    raise RuntimeError("Failed to sample a valid Countdown instance within max_tries.")


def build_dataset(
    size: int,
    seed: int = 0,
    num_operands: int = 3,
    min_number: int = 1,
    max_number: int = 25,
    max_target: int = 200,
) -> list[CountdownProblem]:
    """Create a deterministic list of Countdown problems."""
    rng = random.Random(seed)
    return [
        generate_problem(
            rng,
            num_operands=num_operands,
            min_number=min_number,
            max_number=max_number,
            max_target=max_target,
        )
        for _ in range(size)
    ]


def format_messages(problem: CountdownProblem) -> list[dict[str, str]]:
    """Build chat messages matching the paper's Countdown prompt template."""
    instruction = (
        f"Using the numbers {problem.numbers}, create an equation that equals "
        f"{problem.target}. You can use basic arithmetic operations (+, -, *, /) "
        "and each number can only be used once. Show your work briefly in "
        "<think> </think> tags, and return the final answer in "
        "<answer> </answer> tags, for example <answer> (1 + 2) * 3 </answer>."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def _normalize_expr(raw: str) -> str:
    """Normalize Unicode math symbols and strip optional '= target' clauses."""
    expr = (
        raw.strip()
        .replace("×", "*")
        .replace("✕", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("（", "(")
        .replace("）", ")")
    )
    if "=" in expr:
        parts = [p.strip() for p in expr.split("=") if p.strip()]
        # Pick the side containing the most numbers (typically the LHS arithmetic expression).
        if parts:
            expr = max(parts, key=lambda p: len(re.findall(r"\d+", p)))
    return expr


def extract_answer(text: str) -> str | None:
    """Return the normalized expression inside the last <answer>...</answer> tag."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not matches:
        return None
    return _normalize_expr(matches[-1])


def compute_reward(completion: str, problem: CountdownProblem) -> float:
    """Return 1.0 iff <answer> is a valid arithmetic expression evaluating to target."""
    expr = extract_answer(completion)
    if expr is None or not _ALLOWED_EXPR.match(expr) or "**" in expr or "//" in expr:
        return 0.0
    used_numbers = Counter(int(x) for x in re.findall(r"\d+", expr))
    if used_numbers != Counter(problem.numbers):
        return 0.0
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return 0.0
    return 1.0 if abs(float(value) - problem.target) < 1e-9 else 0.0
