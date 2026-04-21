"""
StyleMind AI - Pipeline Orchestrator
Runs the full intent → plan → retrieve → assemble → lookbook pipeline
with maximum parallelism to keep latency low.

Performance design:
  1. Intent Parser (Haiku, ~1s)
  2. Style Planner (Sonnet, ~3-4s)  ← reduced max_tokens vs raw agents
  3. Retrieval: all blueprint slots fired in parallel via ThreadPoolExecutor
  4. Assembly: rule-based, instant
  5. Lookbook: both outfits generated in parallel via ThreadPoolExecutor
Total wall-clock: ~7-10s (vs ~18-25s sequential)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.intent_parser import IntentParser, ParsedIntent
from agents.lookbook_generator import LookbookEntry, LookbookGenerator
from agents.outfit_assembler import OutfitAssembler, ScoredOutfit
from agents.retriever import ProductRetriever
from agents.style_planner import PlannerOutput, StylePlanner
from agents.taste_profile import TasteProfile


# ── Result container ───────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    intent: ParsedIntent
    plan: PlannerOutput
    outfits: list[ScoredOutfit]
    lookbooks: list[LookbookEntry]     # parallel-indexed with outfits
    timing: dict[str, float] = field(default_factory=dict)

    def get_outfit_with_lookbook(self, idx: int) -> tuple[ScoredOutfit, Optional[LookbookEntry]]:
        outfit = self.outfits[idx]
        lb = self.lookbooks[idx] if idx < len(self.lookbooks) else None
        return outfit, lb


# ── Pipeline ───────────────────────────────────────────────────────────────

class StyleMindPipeline:
    """
    End-to-end pipeline with parallel retrieval and lookbook generation.

    progress_cb: optional callback(stage: str, detail: str) for UI updates.
    """

    def __init__(
        self,
        retriever: Optional[ProductRetriever] = None,
        progress_cb: Optional[Callable[[str, str], None]] = None,
    ):
        self._retriever = retriever
        self._progress = progress_cb or (lambda stage, detail: None)
        self._parser = IntentParser()
        self._planner = StylePlanner()
        self._assembler = OutfitAssembler()
        self._lookbook_gen = LookbookGenerator()

    @property
    def retriever(self) -> ProductRetriever:
        if self._retriever is None:
            self._retriever = ProductRetriever()
        return self._retriever

    def run(
        self,
        query: str,
        profile: Optional[TasteProfile] = None,
        top_k_per_slot: int = 8,
        num_outfits: int = 2,
    ) -> PipelineResult:
        timing: dict[str, float] = {}
        t0 = time.perf_counter()

        # ── Step 1: Parse intent ──────────────────────────────────────────
        self._progress("parse", "Parsing your request…")
        t = time.perf_counter()
        intent = self._parser.parse(query, profile)
        # Hard-enforce profile gender — LLM may not reliably inherit it
        if (intent.gender_expression in ("unspecified", "", None)
                and profile
                and profile.gender_expression not in ("unspecified", "", None)):
            intent.gender_expression = profile.gender_expression
        timing["parse"] = time.perf_counter() - t

        # ── Step 2: Plan blueprints ───────────────────────────────────────
        self._progress("plan", "Planning outfit blueprints…")
        t = time.perf_counter()
        plan = self._planner.plan(intent, profile, num_outfits=num_outfits)
        # Hard-enforce count: trim excess blueprints; LLMs sometimes overshoot
        if len(plan.blueprints) > num_outfits:
            plan.blueprints = plan.blueprints[:num_outfits]
        timing["plan"] = time.perf_counter() - t

        # ── Step 3: Parallel retrieval ────────────────────────────────────
        self._progress("retrieve", f"Finding products for {len(plan.blueprints)} outfits…")
        t = time.perf_counter()
        intent_dict = intent.model_dump()

        # Build task list: (blueprint_name, slot, search_query, target_color)
        tasks = [
            (bp.outfit_name, bp_item.slot, bp_item.search_query, bp_item.target_color)
            for bp in plan.blueprints
            for bp_item in bp.items
        ]

        all_slot_candidates: dict[str, dict[str, list[dict]]] = {
            bp.outfit_name: {} for bp in plan.blueprints
        }

        def _retrieve_slot(task):
            bp_name, slot, query_str, color = task
            candidates = self.retriever.retrieve_for_blueprint_slot(
                search_query=query_str,
                slot=slot,
                target_color=color,
                intent_data=intent_dict,
                top_k=top_k_per_slot,
            )
            return bp_name, slot, candidates

        with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            futures = {executor.submit(_retrieve_slot, t): t for t in tasks}
            for future in as_completed(futures):
                bp_name, slot, candidates = future.result()
                all_slot_candidates[bp_name][slot] = candidates

        timing["retrieve"] = time.perf_counter() - t

        # ── Step 4: Assemble + score outfits ─────────────────────────────
        self._progress("assemble", "Assembling and scoring outfits…")
        t = time.perf_counter()
        outfits = self._assembler.assemble_all(
            plan.blueprints,
            all_slot_candidates,
            budget_max=intent.budget_max,
        )
        timing["assemble"] = time.perf_counter() - t

        # ── Step 5: Parallel lookbook generation ──────────────────────────
        self._progress("lookbook", "Writing your lookbook…")
        t = time.perf_counter()
        lookbooks = self._generate_lookbooks_parallel(outfits, intent, profile)
        timing["lookbook"] = time.perf_counter() - t

        # Attach prose to outfits
        for outfit, lb in zip(outfits, lookbooks):
            outfit.lookbook_prose = lb.prose

        timing["total"] = time.perf_counter() - t0
        self._progress("done", f"Done in {timing['total']:.1f}s")

        return PipelineResult(
            intent=intent,
            plan=plan,
            outfits=outfits,
            lookbooks=lookbooks,
            timing=timing,
        )

    def _generate_lookbooks_parallel(
        self,
        outfits: list[ScoredOutfit],
        intent: ParsedIntent,
        profile: Optional[TasteProfile],
    ) -> list[LookbookEntry]:
        if not outfits:
            return []

        results = [None] * len(outfits)

        def _gen(idx_outfit):
            idx, outfit = idx_outfit
            return idx, self._lookbook_gen.generate(outfit, intent, profile)

        with ThreadPoolExecutor(max_workers=len(outfits)) as executor:
            futures = {executor.submit(_gen, (i, o)): i for i, o in enumerate(outfits)}
            for future in as_completed(futures):
                idx, entry = future.result()
                results[idx] = entry

        return results
