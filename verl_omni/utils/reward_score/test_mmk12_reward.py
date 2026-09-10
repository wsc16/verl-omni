#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone test for ``mmk12_reward.compute_score``.

Focus: exercise the **numeric / LaTeX branch** at
``mmk12_reward.py:154``::

    else:
        # Numeric / LaTeX answer -> reuse verl's math_verify ...
        accuracy_reward = _math_verify_score(content, gt, timeout=math_verify_timeout)

That branch is reached iff the ground truth is NOT a single choice letter A-E
(``_is_choice_gt(gt)`` is False). Every case below uses a numeric / LaTeX /
angle ground truth, so ``compute_score`` must dispatch to line 154. A spy on
``_math_verify_score`` records each call to prove the branch was hit.

Run (after activating the env)::

    source /home/w00934247/wsc/env.sh
    python verl_omni/utils/reward_score/test_mmk12_reward.py
"""

from __future__ import annotations

from verl_omni.utils.reward_score import mmk12_reward

compute_score = mmk12_reward.compute_score

# ---------------------------------------------------------------------------
# Spy on ``_math_verify_score`` (the symbol mmk12_reward calls at line 154).
# It records (gt, content) for every call, then delegates to the real impl so
# the subprocess-isolated math_verify still runs and scores normally.
# ---------------------------------------------------------------------------
_real_math_verify = mmk12_reward._math_verify_score
_math_verify_calls: list[tuple[str, str]] = []


def _spy_math_verify(content, gt, timeout=20.0, **kwargs):
    _math_verify_calls.append((str(gt), str(content)[:80]))
    return _real_math_verify(content, gt, timeout=timeout, **kwargs)


mmk12_reward._math_verify_score = _spy_math_verify

# Default format_score used by compute_score.
FORMAT_SCORE = 0.3
TOL = 1e-6


# ---------------------------------------------------------------------------
# Test cases. All ground_truths are non-choice -> all hit line 154.
#   (name, solution_str, ground_truth, expect_acc, expect_format_reward)
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str, str, float, float]] = [
    # 1. Correct integer, full format (answer_tag + boxed).
    (
        "correct_int_full_fmt",
        "<answer>\\boxed{42}</answer>",
        "42",
        1.0,
        0.30,
    ),
    # 2. Wrong integer, full format -> accuracy 0, format full.
    (
        "wrong_int_full_fmt",
        "<answer>\\boxed{7}</answer>",
        "42",
        0.0,
        0.30,
    ),
    # 3. LaTeX fraction equals decimal (math_verify symbolic/numeric eq).
    (
        "latex_frac_eq_decimal",
        "<answer>\\boxed{\\frac{1}{2}}</answer>",
        "0.5",
        1.0,
        0.30,
    ),
    # 4. Plain decimal equality.
    (
        "decimal_eq",
        "<answer>\\boxed{3.14}</answer>",
        "3.14",
        1.0,
        0.30,
    ),
    # 5. Angle symbol normalized (_normalize_angle strips ° / \circ) -> line 154.
    (
        "angle_normalize_deg",
        "<answer>\\boxed{90}</answer>",
        "90°",
        1.0,
        0.30,
    ),
    # 6. No <answer> tag: content falls back to full solution_str. boxed is in
    #    the text but NOT inside <answer>, so format_reward = 0 (format requires
    #    the answer_tag). Correctness still judged via math_verify.
    (
        "no_answer_tag_boxed_in_text",
        "The answer is \\boxed{12}.",
        "12",
        1.0,
        0.0,
    ),
    # 7. <answer> tag present but no \boxed -> half format reward.
    (
        "answer_tag_no_boxed",
        "<answer>42</answer>",
        "42",
        1.0,
        0.15,
    ),
    # 8. Wrong, no format at all.
    (
        "wrong_no_format",
        "I think the answer is 99.",
        "42",
        0.0,
        0.0,
    ),
    # 9. Symbolic expression equivalence (1+1 == 2).
    (
        "symbolic_expr_eq",
        "<answer>\\boxed{1+1}</answer>",
        "2",
        1.0,
        0.30,
    ),
]


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print("=" * 92)
    print("mmk12_reward.compute_score  ->  line 154 (numeric/LaTeX) branch")
    print("=" * 92)
    header = f"{'case':<28}{'acc':>8}{'exp_acc':>10}{'fmt':>8}{'exp_fmt':>10}{'score':>9}{'ok':>5}"
    print(header)
    print("-" * 92)

    n_pass = 0
    for name, sol, gt, exp_acc, exp_fmt in CASES:
        res = compute_score(sol, gt, format_score=FORMAT_SCORE)
        acc = res["accuracy"]
        fmt = res["format_reward"]
        score = res["score"]

        acc_ok = _close(acc, exp_acc)
        fmt_ok = _close(fmt, exp_fmt)
        # score = (acc + fmt) / (1 + format_score)
        exp_score = (exp_acc + exp_fmt) / (1.0 + FORMAT_SCORE)
        score_ok = _close(score, exp_score)
        ok = acc_ok and fmt_ok and score_ok
        n_pass += int(ok)

        flag = "OK" if ok else "FAIL"
        print(f"{name:<28}{acc:>8.3f}{exp_acc:>10.3f}{fmt:>8.3f}{exp_fmt:>10.3f}{score:>9.3f}{flag:>5}")
        if not ok:
            print(f"    -> got score={score:.4f} (exp {exp_score:.4f}); "
                  f"acc_ok={acc_ok} fmt_ok={fmt_ok} score_ok={score_ok}")

    print("-" * 92)
    print(f"passed {n_pass}/{len(CASES)} cases")

    # Verify line 154 was actually reached: every non-choice case must have
    # triggered exactly one _math_verify_score call.
    print()
    print(f"_math_verify_score calls recorded: {len(_math_verify_calls)} "
          f"(expected {len(CASES)} = one per non-choice case -> line 154)")
    for i, (gt, content) in enumerate(_math_verify_calls):
        print(f"  [{i + 1}] gt={gt!r:<14} content={content!r}")

    line154_hit = len(_math_verify_calls) == len(CASES)
    print()
    print(f"line 154 reached: {'YES' if line154_hit else 'NO'}")
    print("=" * 92)

    if n_pass != len(CASES) or not line154_hit:
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
