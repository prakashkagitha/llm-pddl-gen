# src/inference/pddlwithpythonenv_inference.py
import os
from typing import List, Dict
from .base_inference import BaseInference


class PDDLWithPythonEnvInference(BaseInference):
    """
    Two-step pipeline

    1. Natural-language → *Python planner environment* (domain/problem
       code wrapped in <domain_file>/<problem_file>).
    2. (Python env + NL description) → final PDDL.

    Tags remain identical to other pipelines, so BaseInference continues
    to extract df/pf without modification.
    """

    # ------------------------------------------------------------------ #
    # prompt helpers
    def _env_prefix(self) -> str:
        """Planner-synthesis prompt (no py2pddl mentions)."""
        return (
            "You are a Python engineer who translates natural-language planning "
            "descriptions into **executable symbolic-planner code**.\n"
            "Your task has TWO parts:\n"
            "  1. Fill in every section marked “# LLM: Please synthesise” in the "
            "     planner template below so that the code runs under Python 3.11.\n"
            "     • Define all object types, concrete objects, predicates, actions\n"
            "       (with pre-conditions & effects) needed to solve the problem.\n"
            "     • Provide helper functions:\n"
            "         – available_actions(state)\n"
            "         – execute(action, state)\n"
            "         – is_goal(state)\n"
            "  2. Keep the search loop unchanged.\n"
            "\n"
            "Wrap the DOMAIN part of the code in <domain_file>…</domain_file> and the\n"
            "PROBLEM-specific part in <problem_file>…</problem_file>.  No text outside\n"
            "those tags.  Begin private reasoning inside a <think> tag.\n\n"
            "## Planner template\n"
            "```python\n"
            "<domain_file>\n"
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "from typing import List, Tuple, Iterable\n\n"
            "# ---------- Domain objects & types ----------\n"
            "# LLM: Please synthesise classes/enums for object types\n\n"
            "# ---------- Predicates ----------\n"
            "# LLM: Please synthesise predicate representations\n\n"
            "# ---------- Action schema ----------\n"
            "@dataclass(frozen=True)\n"
            "class Action:\n"
            "    # LLM: Please synthesise action fields\n"
            "    name: str\n"
            "    params: Tuple\n"
            "    def precond(self, state: \"State\") -> bool: ...\n"
            "    def effect(self, state: \"State\") -> \"State\": ...\n\n"
            "# ---------- World state ----------\n"
            "@dataclass(frozen=True)\n"
            "class State:\n"
            "    # LLM: Please synthesise immutable state representation\n"
            "    ...\n\n"
            "# ---------- Planner helpers ----------\n"
            "def available_actions(s: State) -> Iterable[Action]:\n"
            "    \"\"\"Enumerate all ground actions valid in state *s*.\"\"\"\n"
            "    # LLM: Please synthesise\n"
            "    ...\n\n"
            "def execute(a: Action, s: State) -> State:\n"
            "    \"\"\"Return successor state after applying *a*.\"\"\"\n"
            "    return a.effect(s)\n"
            "</domain_file>\n\n"
            "<problem_file>\n"
            "# ---------- Concrete objects ----------\n"
            "# LLM: Please create concrete objects for this problem\n\n"
            "# ---------- Initial state ----------\n"
            "def initial_state() -> State:\n"
            "    # LLM: Please build and return initial State\n"
            "    ...\n\n"
            "# ---------- Goal test ----------\n"
            "def is_goal(s: State) -> bool:\n"
            "    # LLM: Please implement goal condition\n"
            "    ...\n\n"
            "# ---------- Search loop ----------\n"
            "def solve():\n"
            "    init = initial_state()\n"
            "    frontier: List[Tuple[State, List[Action]]] = [(init, [])]\n"
            "    while frontier:\n"
            "        state, acts = frontier.pop()\n"
            "        for act in available_actions(state):\n"
            "            nxt = execute(act, state)\n"
            "            if is_goal(nxt):\n"
            "                return acts + [act]\n"
            "            frontier.append((nxt, acts + [act]))\n"
            "</problem_file>\n"
            "```\n"
        )

    def _pddl_prefix(self) -> str:
        with open("prompts/only_pddl_instruction.txt") as f:
            return f.read().rstrip() + "\n\n"

    # ------------------------------------------------------------------ #
    def _descs(self, pid: str):
        base = os.path.join("data", f"textual_{self.domain}", self.data_type)
        with open(os.path.join(base, f"{pid}_domain.txt")) as f:
            d = f.read()
        with open(os.path.join(base, f"{pid}_problem.txt")) as f:
            p = f.read()
        return d, p

    # ------------------------------------------------------------------ #
    def batch_generate_candidates(self, pids: List[str]) -> List[List[Dict]]:
        # ---------- STEP 1  NL → Python environment ----------
        env_prompts, doms, probs = [], [], []
        for pid in pids:
            d_desc, p_desc = self._descs(pid)
            doms.append(d_desc), probs.append(p_desc)

            env_prompts.append(
                self._env_prefix()
                + "### Domain description\n" + d_desc
                + "\n\n### Problem description\n" + p_desc
                + "\n\n<think>"
            )

        env_outs = self.llm.generate(env_prompts, self.sampler)
        env_domains, env_problems = [], []
        for o in env_outs:
            txt = o.outputs[0].text.split("</think>", 1)[-1]
            env_domains.append(self._unwrap(txt, "domain_file"))
            env_problems.append(self._unwrap(txt, "problem_file"))

        # ---------- STEP 2  (env + NL) → PDDL ----------
        pddl_prompts = []
        for d_desc, p_desc, env_dom, env_prob in zip(
            doms, probs, env_domains, env_problems
        ):
            pddl_prompts.append(
                self._pddl_prefix()
                + "You are an expert PDDL engineer.\n"
                  "Using ONLY the resources below, write *syntactically correct* and "
                  "mutually consistent PDDL.\n\n"
                + "### Original domain description\n" + d_desc
                + "\n\n### Original problem description\n" + p_desc
                + "\n\n### Python planner environment – domain\n```python\n"
                + env_dom.strip()
                + "\n```\n\n### Python planner environment – problem\n```python\n"
                + env_prob.strip()
                + "\n```\n\n"
                  "Wrap the final domain PDDL in <domain_file>…</domain_file> and the "
                  "final problem PDDL in <problem_file>…</problem_file>. "
                  "Do not output anything else.\n<think>"
            )

        pddl_outs = self.llm.generate(pddl_prompts, self.sampler)

        # ---------- package results ----------
        candidates = []
        for out, env_dom, env_prob in zip(pddl_outs, env_domains, env_problems):
            cand = self._resp2dict(out.outputs[0].text)          # df, pf, raw, summary
            cand["python_domain_file"] = env_dom                 # keep intermediates
            cand["python_problem_file"] = env_prob
            candidates.append([cand])

        return candidates

    # single-example interface not used
    def get_prompt(self, pid: str):
        raise NotImplementedError