"""OPRO: Optimization by PROmpting for zero-shot vulnerability detection.

Iteratively improves the user prompt template by using the same LLM as both
scorer (temperature=0) and optimizer (temperature=1.0). After each step, the
best prompt and the full optimization history are checkpointed as JSON under
the storage_structure() path.

Reference: Yang et al., "Large Language Models as Optimizers" (2023).
"""

from __future__ import annotations

import json
import random
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from src.models.api_based import ContextExceededError

if TYPE_CHECKING:
    from src.models.api_based import ApiBasedModel

# How many past (instruction, score) pairs to include in the meta-prompt.
_MAX_HISTORY_IN_META_PROMPT = 20
# Tokens budget for the optimizer call (instructions can be verbose).
_OPTIMIZER_MAX_TOKENS = 1024
# Maximum API calls per step before giving up and skipping.
_MAX_RETRIES_PER_STEP = 3
# Regex that accepts common output-format directive variations (e.g. "(1) YES", "YES/NO").
_OUTPUT_FORMAT_RE = re.compile(r"\(1\)|\(2\)|YES|NO", re.IGNORECASE)
# System prompt reinforcing output format for the optimizer (meta-prompt) call.
_OPTIMIZER_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. "
    "When asked to generate a new instruction, you MUST output ONLY a single "
    "<INS>...</INS> block and nothing else. "
    "The instruction inside the tags MUST contain the placeholder `{func}` "
    "exactly once. The instruction should be generated to maximize the "
    "F1 score of a vulnerability detection task across 300k samples "
)


class OPROOptimizer:
    """Prompt optimizer using the OPRO algorithm.

    Args:
        model: An initialised ``ApiBasedModel`` whose ``user_prompt_template``
            is used as the starting point.
        train_data: The validation split produced by ``GenericDataModule.setup()``.
            Each element is a dict with ``"text"``, ``"label"``, and ``"index"`` keys.
            Used for prompt evaluation (hyperparameter search).
        steps: Number of optimization steps (candidate prompts to try).
        train_size: Number of training samples to evaluate each candidate on.
            Sampled once at construction time and kept fixed across all steps.
        checkpoint_dir: If given, history.parquet and best_prompt.txt are
            written here after every evaluated step.
        hyperparams: Experiment hyperparameters written to metadata.json once
            at the start of optimize().
    """

    def __init__(
        self,
        model: "ApiBasedModel",
        train_data: list,
        steps: int = 10,
        train_size: int = 128,
        candidates_per_step: int = 8,
        checkpoint_dir: Path | None = None,
        hyperparams: dict | None = None,
        log_dir: Path | None = None,
        loggers: list | None = None,
    ) -> None:
        self.model = model
        self.steps = steps
        self.candidates_per_step = candidates_per_step
        self.checkpoint_dir = checkpoint_dir
        self.hyperparams = hyperparams or {}
        self.log_dir = log_dir
        self.loggers = loggers or []

        rng = random.Random(42)
        sample_size = min(train_size, len(train_data))
        eval_indices = set(rng.sample(range(len(train_data)), sample_size))
        self.eval_data = [train_data[i] for i in sorted(eval_indices)]

        # Pick a small balanced set of examples for the meta-prompt header.
        # Drawn from data NOT in eval_data to prevent the optimizer from embedding
        # eval samples verbatim in generated prompts (reward hacking).
        holdout = [train_data[i] for i in range(len(train_data)) if i not in eval_indices]
        pool = holdout if holdout else train_data
        vulnerable = [d for d in pool if d["label"] == 1]
        safe = [d for d in pool if d["label"] == 0]
        n_vuln = min(2, len(vulnerable))
        n_safe = min(1, len(safe))
        self.few_shot_examples = rng.sample(vulnerable, n_vuln) + rng.sample(safe, n_safe)

        self.history: list[tuple[str, float, dict]] = []

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_prompt(
        self, prompt_template: str
    ) -> tuple[float, list[int], list[int], list[str], list[int]]:
        """Evaluate *prompt_template* on the fixed eval set.

        Returns (f1, predictions, labels, answers, indices).

        Returns 0.0 F1 if the prompt produces only one predicted class
        across the entire eval set — this penalises degenerate prompts that
        game F1 by always predicting vulnerable or always predicting safe.
        """
        predictions: list[int] = []
        labels: list[int] = []
        answers: list[str] = []
        indices: list[int] = []
        skipped = 0
        api_errors = 0
        for sample in self.eval_data:
            try:
                answer = self.model._generate_answer_with_prompt(sample["text"], prompt_template)
            except ContextExceededError:
                skipped += 1
                if skipped == 1:
                    print("[OPRO] Warning: sample exceeded context window, skipping.")
                continue
            except RuntimeError as exc:
                api_errors += 1
                if api_errors == 1:
                    print(f"[OPRO] Warning: API error during evaluation, skipping sample: {exc}")
                continue
            answers.append(answer)
            predictions.append(self.model._to_binary_label(answer))
            labels.append(int(sample["label"]))
            indices.append(int(sample["index"]))
        if skipped:
            print(f"[OPRO] Skipped {skipped}/{len(self.eval_data)} samples (context exceeded).")
        if api_errors:
            print(f"[OPRO] Skipped {api_errors}/{len(self.eval_data)} samples (API errors).")
        if len(set(predictions)) < 2:
            return 0.0, predictions, labels, answers, indices
        return _macro_f1(predictions, labels), predictions, labels, answers, indices

    # ------------------------------------------------------------------
    # Meta-prompt construction
    # ------------------------------------------------------------------

    def build_meta_prompt(self) -> str:
        """Build the meta-prompt from the optimization history and few-shot examples."""
        lines: list[str] = []

        lines.append(textwrap.dedent("""\
            I have a code vulnerability detection task.
            Given a C/C++ function, a model must output "(1) YES" if the code contains
            a security vulnerability, or "(2) NO" if it is safe.
        """))

        # Few-shot examples to ground the optimizer.
        if self.few_shot_examples:
            lines.append("### Example code snippets and expected labels\n")
            for i, ex in enumerate(self.few_shot_examples, 1):
                label_str = "VULNERABLE" if ex["label"] == 1 else "SAFE"
                # Truncate very long snippets to avoid blowing the context.
                snippet = ex["text"][:1000].strip()
                lines.append(f"Example {i} (label: {label_str}):\n```c\n{snippet}\n```\n")

        # Optimization trajectory: take most recent window chronologically, then sort
        # ascending so best appears last. This preserves negative signal from recent bad
        # candidates rather than discarding them in favour of the global best-ever set.
        window_size = max(_MAX_HISTORY_IN_META_PROMPT, 2 * self.candidates_per_step + 4)
        recent_window = self.history[-window_size:]
        recent = sorted(recent_window, key=lambda t: t[1])
        lines.append("### Previous instructions with their evaluation scores (macro F1, higher is better)\n")
        for instruction, score, _metrics in recent:
            lines.append(f"score: {score:.4f}\ninstruction:\n{instruction}\n")

        lines.append(textwrap.dedent("""\
            ### Task
            Generate a new instruction that is different from all previous instructions
            and will achieve a higher F1 score on the vulnerability detection task.

            Requirements:
            - The instruction MUST contain the placeholder `{func}` exactly once,
              where the code snippet will be inserted at inference time.
            - The instruction must end with a clear output directive instructing the
              model to reply ONLY with "(1) YES" or "(2) NO".
            - The instruction should be self-contained and clear.

            ### Example of a valid response
            <INS>
            You are a security auditor. Examine the following C/C++ function for memory
            safety issues, integer overflows, use-after-free, injection flaws, and other
            common vulnerability classes:

            {func}

            Does this function contain a security vulnerability?
            Reply ONLY with "(1) YES" or "(2) NO". Do not include any explanation.
            </INS>

            Now generate your new instruction wrapped in <INS></INS> tags. Output nothing outside the tags.
        """))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_instruction(response: str) -> str | None:
        """Parse the <INS>…</INS> tagged instruction from the optimizer response."""
        # Strip reasoning-model thinking blocks (Qwen3, DeepSeek-R1, etc.) before extraction
        # so that {func} references inside <think> don't pollute the fallback path.
        stripped = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
        match = re.search(r"<INS>(.*?)</INS>", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _generate_candidate(self, step: int, c_idx: int) -> str | None:
        """Try up to _MAX_RETRIES_PER_STEP times to get a valid candidate prompt."""
        for attempt in range(1, _MAX_RETRIES_PER_STEP + 1):
            meta_prompt = self.build_meta_prompt()
            try:
                response = self.model.client.chat.completions.create(
                    model=self.model.model_name,
                    messages=[
                        {"role": "system", "content": _OPTIMIZER_SYSTEM_PROMPT},
                        {"role": "user", "content": meta_prompt},
                    ],
                    max_tokens=_OPTIMIZER_MAX_TOKENS,
                    temperature=1.0,
                    **({"extra_body": _eb} if (_eb := getattr(self.model, "extra_body_params", {})) else {}),
                )
                raw = response.choices[0].message.content.strip()
            except ContextExceededError as exc:
                print(f"[OPRO] Step {step} c{c_idx}: Meta-prompt exceeds context window, skipping. {exc}")
                return None
            except Exception as exc:  # noqa: BLE001
                print(f"[OPRO] Step {step} c{c_idx} attempt {attempt}: API error (url={self.model.api_url}): {exc}, retrying.")
                continue

            # Strip thinking blocks before fallback path so {func} inside <think> doesn't
            # trigger a false positive match.
            raw_stripped = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
            extracted = self.extract_instruction(raw)
            if extracted is None:
                # Fallback: try treating the thinking-stripped response as the candidate body.
                # Strip common preamble patterns (markdown fences, leading label lines).
                cleaned = re.sub(r"^```[^\n]*\n?", "", raw_stripped, flags=re.MULTILINE)
                cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
                if "{func}" in cleaned and _OUTPUT_FORMAT_RE.search(cleaned):
                    extracted = cleaned
                else:
                    print(f"[OPRO] Step {step} c{c_idx} attempt {attempt}: No <INS> tag found, retrying.")
                    continue
            if "{func}" not in extracted:
                print(f"[OPRO] Step {step} c{c_idx} attempt {attempt}: Candidate missing {{func}} placeholder, retrying.")
                continue
            if not _OUTPUT_FORMAT_RE.search(extracted):
                print(f"[OPRO] Step {step} c{c_idx} attempt {attempt}: Candidate missing output format directive, retrying.")
                continue

            # Escape any literal braces that aren't {func} so str.format(func=...) won't raise.
            return extracted.replace("{", "{{").replace("}", "}}").replace("{{func}}", "{func}")

        print(f"[OPRO] Step {step} c{c_idx}: All {_MAX_RETRIES_PER_STEP} retries exhausted, skipping.")
        return None

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self) -> str:
        """Run the OPRO loop and return the best prompt template found."""
        self._load_history()
        self._save_metadata()

        initial_prompt = self.model.user_prompt_template

        if not self.history:
            print(f"[OPRO] Evaluating initial prompt on {len(self.eval_data)} samples …")
            score, preds, lbls, answers, idxs = self.evaluate_prompt(initial_prompt)
            metrics = _compute_metrics(preds, lbls)
            self.history.append((initial_prompt, score, metrics))
            self._save_train_eval(step=0, predictions=preds, labels=lbls, answers=answers, indices=idxs)
            if self.loggers:
                log_metrics = {
                    "train_f1": score,
                    "train_accuracy": metrics["accuracy"],
                    "train_recall": metrics["recall"],
                    "train_precision": metrics["precision"],
                    "train_balance": metrics["balance"],
                }
                for logger in self.loggers:
                    logger.log_metrics(log_metrics, step=0)
            print(
                f"[OPRO] Initial score: F1={score:.4f}  acc={metrics['accuracy']:.4f}"
                f"  rec={metrics['recall']:.4f}  prec={metrics['precision']:.4f}"
            )

        best_prompt, best_score, _ = max(self.history, key=lambda t: t[1])

        # Steps already completed (excluding the initial evaluation).
        # With batching, history grows as: 1 + completed_steps * candidates_per_step.
        completed_steps = max(0, (len(self.history) - 1) // self.candidates_per_step)
        remaining = self.steps - completed_steps

        if remaining <= 0:
            print(f"[OPRO] All {self.steps} steps already completed. Returning best prompt.")
            return best_prompt

        for step in range(completed_steps + 1, self.steps + 1):
            print(f"[OPRO] Step {step}/{self.steps} — generating {self.candidates_per_step} candidates …")

            # Generation phase: produce the full batch before evaluating any of them.
            batch: list[str] = []
            for c_idx in range(self.candidates_per_step):
                candidate = self._generate_candidate(step, c_idx)
                if candidate is not None:
                    batch.append(candidate)

            if not batch:
                print(f"[OPRO] Step {step}: No valid candidates generated, skipping.")
                continue

            # Evaluation phase: evaluate, log, and append all candidates to history.
            # All are appended before the next call to build_meta_prompt() so the
            # optimizer sees the full batch's results in the next step's meta-prompt.
            for c_idx, candidate in enumerate(batch):
                global_step_id = (step - 1) * self.candidates_per_step + c_idx + 1
                try:
                    score, preds, lbls, answers, idxs = self.evaluate_prompt(candidate)
                    metrics = _compute_metrics(preds, lbls)
                    self.history.append((candidate, score, metrics))
                    self._save_train_eval(step=global_step_id, predictions=preds, labels=lbls, answers=answers, indices=idxs)
                    if self.loggers:
                        log_metrics = {
                            "train_f1": score,
                            "train_accuracy": metrics["accuracy"],
                            "train_recall": metrics["recall"],
                            "train_precision": metrics["precision"],
                            "train_balance": metrics["balance"],
                        }
                        for logger in self.loggers:
                            logger.log_metrics(log_metrics, step=global_step_id)
                    print(
                        f"[OPRO] Step {step} c{c_idx}: F1={score:.4f}  acc={metrics['accuracy']:.4f}"
                        f"  rec={metrics['recall']:.4f}  prec={metrics['precision']:.4f}  best={best_score:.4f}"
                    )
                    if score > best_score:
                        best_score = score
                        best_prompt = candidate
                        print(f"[OPRO] New best at step {step} c{c_idx}! score={best_score:.4f}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[OPRO] Step {step} c{c_idx}: Evaluation failed, skipping candidate: {exc}")

            self._checkpoint(best_prompt, best_score)

        print(f"[OPRO] Optimization complete. Best score: {best_score:.4f}")
        return best_prompt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        """Write experiment metadata once at the start of optimize()."""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "steps": self.steps,
            "eval_size": len(self.eval_data),
            "system_prompt": self.model.system_prompt,
            "initial_user_prompt": self.model.user_prompt_template,
            **self.hyperparams,
        }
        (self.checkpoint_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    def _checkpoint(self, best_prompt: str, best_score: float) -> None:
        """Called after every step. Writes history.json + best_prompt.txt."""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "step": i // self.candidates_per_step,
                "candidate_idx": i % self.candidates_per_step,
                "global_idx": i,
                "prompt": p,
                "score": s,
                "is_best": (p == best_prompt),
                "metrics": m,
            }
            for i, (p, s, m) in enumerate(self.history)
        ]
        (self.checkpoint_dir / "history.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

        (self.checkpoint_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")

    def _load_history(self) -> None:
        """Restore history from a previous run if checkpoint exists."""
        if self.checkpoint_dir is None:
            return
        path = self.checkpoint_dir / "history.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            self.history = [(r["prompt"], r["score"], r.get("metrics", {})) for r in rows]
            print(f"[OPRO] Resumed from {path}: {len(self.history)} steps already done.")

    def _save_train_eval(
        self,
        step: int,
        predictions: list[int],
        labels: list[int],
        answers: list[str],
        indices: list[int],
    ) -> None:
        """Append this step's per-sample outputs to train_predictions_labels.parquet."""
        if self.checkpoint_dir is None:
            return
        import polars as pl
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "train_predictions_labels.parquet"
        new_rows = pl.DataFrame({
            "step": [step] * len(indices),
            "index": indices,
            "prediction": predictions,
            "label": labels,
            "answer": answers,
        })
        if path.exists():
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_rows])
        else:
            combined = new_rows
        combined.write_parquet(path)

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            per_step_path = self.log_dir / f"train_predictions_labels_epoch000_step{step:08d}.parquet"
            new_rows.write_parquet(per_step_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _macro_f1(predictions: list[int], labels: list[int]) -> float:
    """Compute binary F1 for the positive (vulnerable) class."""
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _compute_metrics(predictions: list[int], labels: list[int]) -> dict:
    """Compute full metric suite matching on_test_epoch_end output."""
    n = len(predictions)
    if n == 0:
        return {"f1": 0.0, "accuracy": 0.0, "recall": 0.0, "precision": 0.0, "balance": 0.0, "size": 0}

    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))

    accuracy = (tp + tn) / n
    # Macro-averaged precision and recall (matching torchmetrics average="macro")
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall = (rec1 + rec0) / 2
    precision = (prec1 + prec0) / 2
    f1 = _macro_f1(predictions, labels)
    n_pos = sum(labels)
    balance = n_pos / n if n > 0 else 0.0

    return {
        "f1": f1,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "balance": balance,
        "size": n,
    }
