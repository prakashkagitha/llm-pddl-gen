"""
Light-weight wrapper around VAL’s `Validate` binary.

Changes vs. the previous version
--------------------------------
* **Success criterion** Now returns *True* when the string
  `"Plan valid"` appears in VAL’s stdout. Every other outcome is a failure.
* **Problem file lookup** Instead of the hard-coded `temp_problem.pddl`,
  the validator automatically locates the reference **problem** file that
  belongs to the same dataset folder as the domain.  Callers therefore pass
  `(domain_name, problem_id, plan_path)` — for example:

      validator.validate("blocksworld", "p02", "/tmp/my_plan.txt")

  and the class will invoke VAL on:

      …/BlocksWorld-111_PDDL/domain.pddl
      …/BlocksWorld-111_PDDL/p02.pddl
      /tmp/my_plan.txt
"""

from __future__ import annotations

import os
import subprocess
from typing import Tuple


class Validator:
    def __init__(self) -> None:
        # Path to the compiled VAL binary (edit if your layout differs)
        self.validate_executable = "../../VAL/build/linux64/Release/bin/Validate"

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def validate(self, domain: str, problem_id: str, plan_file: str) -> bool:
        """
        Return **True** iff VAL prints `"Plan valid"` for the given
        (domain, problem, plan) triple.
        """
        domain_pddl = self._get_domain_path(domain)
        problem_pddl = self._get_problem_path(domain_pddl, problem_id)

        try:
            result = subprocess.run(
                [self.validate_executable, "-v", domain_pddl, problem_pddl, plan_file],
                capture_output=True,
                text=True,
                check=True,
            )
            return "Plan valid" in result.stdout
        except subprocess.CalledProcessError:
            return False

    def validate_with_error(
        self, domain: str, problem_id: str, plan_file: str
    ) -> Tuple[bool, str]:
        """
        Same as *validate* but always returns a tuple *(is_valid, message)*.

        *is_valid*   – boolean  
        *message*    – empty string on success, otherwise VAL’s stdout/stderr
        """
        domain_pddl = self._get_domain_path(domain)
        problem_pddl = self._get_problem_path(domain_pddl, problem_id)

        try:
            result = subprocess.run(
                [self.validate_executable, "-v", domain_pddl, problem_pddl, plan_file],
                capture_output=True,
                text=True,
                check=True,
            )
            ok = "Plan valid" in result.stdout
            return ok, "" if ok else result.stdout
        except subprocess.CalledProcessError as exc:
            return False, exc.stdout or exc.stderr or str(exc)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_domain_path(domain: str) -> str:
        """
        Map logical domain names to the canonical *domain.pddl* inside each
        dataset folder.  Extend this mapping if you add more domains.
        """
        mapping = {
            "blocksworld": "data/textual_blocksworld/BlocksWorld-100_PDDL/domain.pddl",
            "logistics":   "data/textual_logistics/Logistics-100_PDDL/domain.pddl",
            "barman":      "data/textual_barman/Barman-100_PDDL/domain.pddl",
        }
        try:
            return mapping[domain.lower()]
        except KeyError as e:
            raise ValueError(f"Unknown domain: {domain}") from e

    @staticmethod
    def _get_problem_path(domain_pddl: str, problem_id: str) -> str:
        """
        *domain_pddl* is the absolute/relative path returned by *_get_domain_path*.
        We assume problem files live in the **same folder** and follow the naming
        pattern `pNN.pddl` (e.g. *p02.pddl*, *p105.pddl* …).

        *problem_id* should already include the leading 'p', e.g. `'p02'`.
        """
        folder = os.path.dirname(domain_pddl)
        filename = f"{problem_id}.pddl"
        problem_pddl = os.path.join(folder, filename)

        if not os.path.exists(problem_pddl):
            raise FileNotFoundError(
                f"Expected problem file not found: {problem_pddl}"
            )
        return problem_pddl
