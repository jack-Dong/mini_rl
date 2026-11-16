"""Countdown arithmetic environment modeled after score-centering/envs/countdown.py."""
from __future__ import annotations

import ast
import itertools
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


def _solve_exact_with_ops(
    numbers: list[int],
    target: int,
    ops: tuple[str, ...],
) -> str | None:
    """Exhaustively search binary arithmetic trees with Fraction exactness."""
    items: list[tuple[Fraction, str]] = [(Fraction(n), str(n)) for n in numbers]

    def _dfs(pool: list[tuple[Fraction, str]]) -> str | None:
        if len(pool) == 1:
            return pool[0][1] if pool[0][0] == target else None
        for i in range(len(pool)):
            for j in range(len(pool)):
                if i == j:
                    continue
                av, ae = pool[i]
                bv, be = pool[j]
                rest = [pool[k] for k in range(len(pool)) if k not in (i, j)]
                for op in ops:
                    if op == "+":
                        val = av + bv
                    elif op == "-":
                        val = av - bv
                    elif op == "*":
                        val = av * bv
                    else:
                        if bv == 0:
                            continue
                        val = av / bv
                    expr = f"({ae} {op} {be})" if len(pool) < len(items) else f"{ae} {op} {be}"
                    found = _dfs(rest + [(val, expr)])
                    if found is not None:
                        return found
        return None

    return _dfs(items)


def requires_muldiv(problem: CountdownProblem) -> bool:
    """Return True iff target cannot be formed from problem.numbers using only + and -."""
    return _solve_exact_with_ops(problem.numbers, problem.target, ("+", "-")) is None


def solve_countdown_structured(problem: CountdownProblem) -> str | None:
    """Return a clean exact Countdown expression using +, -, *, / if solvable."""
    return _solve_exact_with_ops(problem.numbers, problem.target, OPERATIONS)


def _eval_frac_ast(node: ast.AST) -> Fraction:
    """Safely evaluate a basic binary arithmetic AST in exact rational arithmetic."""
    if isinstance(node, ast.Expression):
        return _eval_frac_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        lv = _eval_frac_ast(node.left)
        rv = _eval_frac_ast(node.right)
        if isinstance(node.op, ast.Add):
            return lv + rv
        if isinstance(node.op, ast.Sub):
            return lv - rv
        if isinstance(node.op, ast.Mult):
            return lv * rv
        if isinstance(node.op, ast.Div):
            return lv / rv
    raise ValueError("Unsupported AST node in Countdown expression")


def _fmt_frac(val: Fraction) -> str:
    return str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"


def _op_symbol(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return "+"
    if isinstance(op, ast.Sub):
        return "-"
    if isinstance(op, ast.Mult):
        return "*"
    return "/"


def solve_no_parens(numbers: list[int], target: int) -> str | None:
    """Return an unparenthesized flat Countdown expression solving target if one exists."""
    n = len(numbers)
    for perm in set(itertools.permutations(numbers)):
        for ops_seq in itertools.product(OPERATIONS, repeat=n - 1):
            parts = [str(perm[0])]
            for op, num in zip(ops_seq, perm[1:]):
                parts.append(f"{op} {num}")
            expr = " ".join(parts)
            try:
                if _eval_frac_ast(ast.parse(expr, mode="eval").body) == target:
                    return expr
            except ZeroDivisionError:
                continue
    return None


def requires_parens(problem: CountdownProblem) -> bool:
    """Return True iff target cannot be formed without operator-precedence parentheses."""
    return solve_no_parens(problem.numbers, problem.target) is None


def classify_problem_skill(problem: CountdownProblem) -> str:
    """Classify a problem into one of 5 mutually-exclusive operator/parens skill strata."""
    if _solve_exact_with_ops(problem.numbers, problem.target, ("+", "-")) is not None:
        return "addsub"
    has_asm = _solve_exact_with_ops(problem.numbers, problem.target, ("+", "-", "*")) is not None
    has_asd = _solve_exact_with_ops(problem.numbers, problem.target, ("+", "-", "/")) is not None
    flat = solve_no_parens(problem.numbers, problem.target) is not None
    need_div = not has_asm
    need_mul = not has_asd
    use_div = need_div or (not need_mul and "/" in problem.solution)
    return ("div_" if use_div else "mul_") + ("flat" if flat else "parens")


BALANCED_SKILL_CYCLE: tuple[str, ...] = (
    "addsub",
    "mul_flat",
    "addsub",
    "div_parens",
    "addsub",
    "div_flat",
    "addsub",
    "mul_parens",
)


def minimal_parens_expr(expr: str) -> str:
    """Strip mathematically redundant parentheses while preserving exact evaluation order."""
    try:
        return ast.unparse(ast.parse(expr, mode="eval").body)
    except SyntaxError:
        return expr


def build_balanced_dataset(
    size: int,
    seed: int = 0,
    num_operands: int = 3,
    min_number: int = 1,
    max_number: int = 25,
    max_target: int = 200,
    exclude_sigs: set[tuple[tuple[int, ...], int]] | None = None,
) -> list[CountdownProblem]:
    """Build a deterministic dataset balanced across add/sub, mul/div, and essential parens."""
    quotas: dict[str, int] = {k: 0 for k in ("addsub", "mul_flat", "div_flat", "mul_parens", "div_parens")}
    for idx in range(size):
        quotas[BALANCED_SKILL_CYCLE[idx % len(BALANCED_SKILL_CYCLE)]] += 1

    rng = random.Random(seed)
    pools: dict[str, list[CountdownProblem]] = {k: [] for k in quotas}
    seen_keys: set[tuple[tuple[int, ...], int]] = set(exclude_sigs) if exclude_sigs else set()
    max_draws = max(50000, size * 120)

    for _ in range(max_draws):
        if all(len(pools[k]) >= quotas[k] for k in quotas):
            break
        prob = generate_problem(
            rng,
            num_operands=num_operands,
            min_number=min_number,
            max_number=max_number,
            max_target=max_target,
        )
        skill = classify_problem_skill(prob)
        if len(pools[skill]) >= quotas[skill]:
            continue
        sig = (tuple(sorted(prob.numbers)), prob.target)
        if sig in seen_keys:
            continue
        seen_keys.add(sig)
        pools[skill].append(prob)

    for skill, target_cnt in quotas.items():
        if len(pools[skill]) < target_cnt:
            raise RuntimeError(f"Only collected {len(pools[skill])}/{target_cnt} balanced problems for {skill!r}.")

    cursors: dict[str, int] = {k: 0 for k in quotas}
    ordered: list[CountdownProblem] = []
    for idx in range(size):
        skill = BALANCED_SKILL_CYCLE[idx % len(BALANCED_SKILL_CYCLE)]
        ordered.append(pools[skill][cursors[skill]])
        cursors[skill] += 1
    return ordered


def build_sft_demonstration(problem: CountdownProblem) -> str:
    """Format a concise step-by-step <think>...</think>\\n<answer>...</answer> SFT example."""
    raw_expr = problem.solution or solve_countdown_structured(problem) or ""
    expr = minimal_parens_expr(raw_expr)
    if "(" in expr:
        flat_alt = solve_no_parens(problem.numbers, problem.target)
        if flat_alt is not None:
            expr = flat_alt
    tree = ast.parse(expr, mode="eval").body
    steps: list[str] = []
    if isinstance(tree, ast.BinOp):
        for child in (tree.left, tree.right):
            if isinstance(child, ast.BinOp):
                c_txt = ast.unparse(child)
                c_val = _fmt_frac(_eval_frac_ast(child))
                steps.append(f"{c_txt} = {c_val}")
        l_str = _fmt_frac(_eval_frac_ast(tree.left))
        r_str = _fmt_frac(_eval_frac_ast(tree.right))
        final_step = f"{l_str} {_op_symbol(tree.op)} {r_str} = {problem.target}"
        steps.append(final_step)
    thought = "; ".join(steps) if steps else f"{expr} = {problem.target}"
    return f"<think>Check {problem.numbers} to reach {problem.target}: {thought}.</think>\n<answer> {expr} </answer>"


def has_strict_single_answer_format(text: str) -> bool:
    """Require reasoning before a single trailing <answer>...</answer> tag."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) != 1:
        return False
    return matches[0].start() >= 16 and (len(text) - matches[0].end()) <= 20


def has_clean_stop_answer(text: str) -> bool:
    """Require one cleanly closed <answer>...</answer> without trailing babble."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) != 1:
        return False
    return (len(text) - matches[0].end()) <= 12


def rollout_repetition_stats(text: str) -> dict[str, float | int]:
    """Measure multi-tag spam, 5-gram text repetition, and repeated commutative candidate tries."""
    n_tags = len(re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL))
    words = re.findall(r"\S+", text.lower())
    if len(words) >= 15:
        ngrams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
        rep_5gram = 1.0 - (len(set(ngrams)) / len(ngrams))
    else:
        rep_5gram = 0.0

    # Ignore <answer>...</answer> spans when counting repeated reasoning tries so echoing
    # a sub-expression inside the final answer tag is not penalized.
    reasoning_only = re.sub(r"<answer>.*?</answer>", " ", text, flags=re.DOTALL)
    cand_matches = re.findall(r"\b\d+(?:\s*[+\-*/]\s*\d+)+\b", reasoning_only)
    seen_sigs: set[tuple[tuple[int, ...], tuple[str, ...]]] = set()
    dup_expr_count = 0
    for expr in cand_matches:
        nums = tuple(sorted(int(x) for x in re.findall(r"\d+", expr)))
        ops = tuple(sorted(re.findall(r"[+\-*/]", expr)))
        sig = (nums, ops)
        if sig in seen_sigs:
            dup_expr_count += 1
        else:
            seen_sigs.add(sig)
    return {"n_tags": n_tags, "rep_5gram": rep_5gram, "dup_expr_count": dup_expr_count}


def _apply_dedup_penalty(completion: str, raw: float) -> float:
    """Apply soft penalties for extra answer tags, 5-gram repetition, or duplicate arithmetic tries."""
    stats = rollout_repetition_stats(completion)
    n_tags = int(stats["n_tags"])
    rep_5gram = float(stats["rep_5gram"])
    dup_expr = int(stats["dup_expr_count"])
    if raw > 0.5:
        tag_pen = 0.15 * min(max(0, n_tags - 1), 3)
        rep_pen = min(0.40, max(0.0, rep_5gram - 0.15) * 1.2)
        dup_pen = min(0.25, 0.08 * dup_expr)
        return max(0.25, 1.0 - tag_pen - rep_pen - dup_pen)
    return -0.25 if (dup_expr >= 2 or rep_5gram > 0.25 or n_tags > 1) else 0.0


def compute_shaped_reward(
    completion: str,
    problem: CountdownProblem,
    reward_shaping: str = "none",
) -> tuple[float, float]:
    """Return (canonical_binary_reward, shaped_training_reward) for ablation rules."""
    raw = compute_reward(completion, problem)
    if reward_shaping == "none":
        return raw, raw
    if reward_shaping == "strict_fmt":
        shaped = raw if (raw > 0.5 and has_strict_single_answer_format(completion)) else 0.0
        return raw, shaped
    if reward_shaping == "stop_eos":
        shaped = raw if (raw > 0.5 and has_clean_stop_answer(completion)) else 0.0
        return raw, shaped
    if reward_shaping == "dedup_pen":
        return raw, _apply_dedup_penalty(completion, raw)
    if reward_shaping == "fmt_dedup":
        if raw > 0.5 and not has_strict_single_answer_format(completion):
            return raw, 0.0
        return raw, _apply_dedup_penalty(completion, raw)
    raise ValueError(f"Unsupported reward_shaping: {reward_shaping!r}")
