"""
Utilities for converting JSON-PDDL specifications into PDDL domain and
problem files. The converter mirrors the schema enforced by the
json_instruction prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


class JSONPDDLConversionError(RuntimeError):
    """Raised when the JSON specification cannot be converted."""


@dataclass(frozen=True)
class Assignment:
    op: str
    function: str
    arguments: Sequence[Any]
    value: Any


def convert_json_to_pddl(spec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Convert a JSON-PDDL specification into domain and problem PDDL strings.

    Parameters
    ----------
    spec:
        Dictionary with ``"domain"`` and ``"problem"`` keys following the
        schema emitted by json_instruction.txt.

    Returns
    -------
    tuple[str, str]
        ``(domain_pddl, problem_pddl)``
    """
    if not isinstance(spec, dict):
        raise JSONPDDLConversionError("Specification must be a JSON object.")

    if "domain" not in spec or "problem" not in spec:
        raise JSONPDDLConversionError("Specification requires 'domain' and 'problem' keys.")

    domain_spec = spec["domain"]
    problem_spec = spec["problem"]

    if not isinstance(domain_spec, dict):
        raise JSONPDDLConversionError("Domain section must be a JSON object.")
    if not isinstance(problem_spec, dict):
        raise JSONPDDLConversionError("Problem section must be a JSON object.")

    domain_name = _require_string(domain_spec, "name", "domain.name")
    problem_domain = _require_string(problem_spec, "domain", "problem.domain")
    if problem_domain != domain_name:
        raise JSONPDDLConversionError(
            f"Problem references domain '{problem_domain}', expected '{domain_name}'."
        )

    domain_pddl = _build_domain(domain_spec)
    problem_pddl = _build_problem(problem_spec, domain_name)
    return domain_pddl, problem_pddl


# --------------------------------------------------------------------------- #
#  Domain helpers                                                             #
# --------------------------------------------------------------------------- #
def _build_domain(domain: Dict[str, Any]) -> str:
    name = _require_string(domain, "name", "domain.name")
    requirements = _require_list(domain, "requirements", "domain.requirements")
    types = _as_list(domain.get("types", []))
    if "constants" in domain:
        constants_value = domain.get("constants")
        if isinstance(constants_value, (list, tuple)) and constants_value:
            raise JSONPDDLConversionError("domain.constants are not supported; use problem.objects.")
        raise JSONPDDLConversionError("domain.constants key is not supported; remove it.")
    predicates = _require_list(domain, "predicates", "domain.predicates")
    actions = _require_list(domain, "actions", "domain.actions")
    functions = _as_list(domain.get("functions", []))

    lines: List[str] = [f"(define (domain {name})"]

    if requirements:
        req_tokens = " ".join(_format_requirement(req) for req in requirements)
        lines.append(f"  (:requirements {req_tokens})")

    if types:
        lines.extend(_format_types(types))

    if functions:
        lines.append("  (:functions")
        for fn in functions:
            lines.append(f"    {_format_function(fn)}")
        lines.append("  )")

    lines.append("  (:predicates")
    for pred in predicates:
        lines.append(f"    {_format_predicate(pred)}")
    lines.append("  )")

    for action in actions:
        lines.extend(_format_action(action))

    lines.append(")")
    return "\n".join(lines)


def _format_requirement(req: Any) -> str:
    token = str(req).strip()
    return token if token.startswith(":") else f":{token}"


def _format_types(types: Sequence[Dict[str, Any]]) -> List[str]:
    parent_map: Dict[str, List[str]] = {}
    for entry in types:
        if not isinstance(entry, dict):
            raise JSONPDDLConversionError("domain.types entries must be objects.")
        name = _require_string(entry, "name", "domain.types[].name")
        parent = str(entry.get("parent", "object")).strip() or "object"
        parent_map.setdefault(parent, []).append(name)

    lines = ["  (:types"]
    for parent, names in parent_map.items():
        type_line = " ".join(names)
        if parent == "object":
            lines.append(f"    {type_line}")
        else:
            lines.append(f"    {type_line} - {parent}")
    lines.append("  )")
    return lines



def _format_predicate(pred: Dict[str, Any]) -> str:
    name = _require_string(pred, "name", "predicate.name")
    parameters = _as_list(pred.get("parameters", []))
    if not parameters:
        return f"({name})"
    param_str = " ".join(_format_typed_variable(p, f"predicate '{name}'") for p in parameters)
    return f"({name} {param_str})"


def _format_function(fn: Dict[str, Any]) -> str:
    name = _require_string(fn, "name", "function.name")
    parameters = _as_list(fn.get("parameters", []))
    return_type = str(fn.get("return", "number")).strip() or "number"
    if parameters:
        param_str = " ".join(_format_typed_variable(p, f"function '{name}'") for p in parameters)
        return f"({name} {param_str}) - {return_type}"
    return f"({name}) - {return_type}"


def _format_action(action: Dict[str, Any]) -> List[str]:
    if not isinstance(action, dict):
        raise JSONPDDLConversionError("domain.actions entries must be objects.")

    name = _require_string(action, "name", "action.name")
    parameters = _as_list(action.get("parameters", []))
    param_vars: List[str] = []
    param_tokens: List[str] = []
    for param in parameters:
        var, token = _normalize_parameter(param, action_name=name)
        param_vars.append(var)
        param_tokens.append(token)

    param_block = " ".join(param_tokens)
    preconditions = action.get("preconditions", {})
    effects = action.get("effects", {})

    var_set: Set[str] = set(var.lstrip("?") for var in param_vars)
    precondition_str = _format_condition_block(preconditions, var_set)
    effect_str = _format_effects(effects, var_set)

    lines = [
        f"  (:action {name}",
        f"     :parameters ({param_block})" if param_block else "     :parameters ()",
        f"     :precondition {precondition_str}",
        f"     :effect {effect_str}",
        "  )",
    ]
    return lines


# --------------------------------------------------------------------------- #
#  Condition / effect helpers                                                 #
# --------------------------------------------------------------------------- #
def _format_condition_block(condition: Any, var_names: Set[str]) -> str:
    if not condition:
        return "()"
    if not isinstance(condition, dict):
        raise JSONPDDLConversionError("Conditions must be objects.")

    parts: List[str] = []
    positives = _as_list(condition.get("positive", []))
    negatives = _as_list(condition.get("negative", []))
    universals = _as_list(condition.get("universal", []))

    parts.extend(_format_literals(positives, var_names))
    parts.extend(f"(not {_format_literal(lit, var_names)})" for lit in negatives)

    for universal in universals:
        if not isinstance(universal, dict):
            raise JSONPDDLConversionError("universal entries must be objects.")
        variables = _as_list(universal.get("variables", []))
        universal_vars: List[str] = []
        extended_names = set(var_names)
        for var in variables:
            token = _format_typed_variable(var, "universal.variables")
            universal_vars.append(token)
            extended_names.add(token.split()[0].lstrip("?"))
        body = _format_condition_block(universal, extended_names)
        parts.append(f"(forall ({' '.join(universal_vars)}) {body})")

    return _combine_with_and(parts)


def _format_effects(effects: Any, var_names: Set[str]) -> str:
    if not effects:
        return "()"
    if not isinstance(effects, dict):
        raise JSONPDDLConversionError("Effects must be objects.")

    literal_parts: List[str] = []
    positives = _as_list(effects.get("add", []))
    negatives = _as_list(effects.get("delete", []))
    conditional = _as_list(effects.get("conditional", []))
    assignments = _as_list(effects.get("assign", []))

    literal_parts.extend(_format_literals(positives, var_names))
    literal_parts.extend(f"(not {_format_literal(lit, var_names)})" for lit in negatives)

    for cond in conditional:
        if not isinstance(cond, dict):
            raise JSONPDDLConversionError("conditional effects must be objects.")
        cond_block = _format_condition_block(cond.get("condition", {}), var_names)
        cond_parts: List[str] = []
        cond_parts.extend(_format_literals(_as_list(cond.get("add", [])), var_names))
        cond_parts.extend(
            f"(not {_format_literal(lit, var_names)})" for lit in _as_list(cond.get("delete", []))
        )
        cond_parts.extend(_format_assignment(assign, var_names) for assign in _as_list(cond.get("assign", [])))
        effect_body = _combine_with_and(cond_parts)
        literal_parts.append(f"(when {cond_block} {effect_body})")

    literal_parts.extend(_format_assignment(assign, var_names) for assign in assignments)
    return _combine_with_and(literal_parts)


def _format_assignment(assign: Any, var_names: Set[str]) -> str:
    if not isinstance(assign, dict):
        raise JSONPDDLConversionError("assign entries must be objects.")
    op = str(assign.get("op", "assign")).strip() or "assign"
    function = _require_string(assign, "function", "assign.function")
    args = _as_list(assign.get("arguments", []))
    target_args = " ".join(_format_term(arg, var_names) for arg in args)
    target = f"({function}{(' ' + target_args) if target_args else ''})"
    value = _format_term(assign.get("value"), var_names)
    return f"({op} {target} {value})"


def _format_literals(literals: Sequence[Any], var_names: Set[str]) -> List[str]:
    return [_format_literal(lit, var_names) for lit in literals]


def _format_literal(literal: Any, var_names: Set[str]) -> str:
    if not isinstance(literal, dict):
        raise JSONPDDLConversionError("Literals must be objects.")
    predicate = _require_string(literal, "predicate", "literal.predicate")
    args = _as_list(literal.get("arguments", []))
    arg_tokens = [_format_term(arg, var_names) for arg in args]
    if arg_tokens:
        return f"({predicate} {' '.join(arg_tokens)})"
    return f"({predicate})"


def _format_term(term: Any, var_names: Set[str]) -> str:
    if isinstance(term, (int, float)):
        return str(term)
    if isinstance(term, str):
        stripped = term.strip()
        if not stripped:
            raise JSONPDDLConversionError("Empty symbol encountered in arguments.")
        if stripped.startswith("?"):
            return stripped
        if stripped in var_names:
            return f"?{stripped}"
        return stripped
    if isinstance(term, dict):
        if "function" in term:
            name = _require_string(term, "function", "term.function")
            args = _as_list(term.get("arguments", []))
            arg_tokens = " ".join(_format_term(arg, var_names) for arg in args)
            return f"({name}{(' ' + arg_tokens) if arg_tokens else ''})"
        if "op" in term and "arguments" in term:
            op = str(term["op"]).strip()
            args = _as_list(term.get("arguments", []))
            arg_tokens = " ".join(_format_term(arg, var_names) for arg in args)
            return f"({op} {arg_tokens})"
    raise JSONPDDLConversionError(f"Unsupported term expression: {term!r}")


def _combine_with_and(parts: Iterable[str]) -> str:
    items = [p for p in parts if p]
    if not items:
        return "()"
    if len(items) == 1:
        return items[0]
    return f"(and {' '.join(items)})"


def _normalize_parameter(param: Any, *, action_name: str) -> Tuple[str, str]:
    if not isinstance(param, dict):
        raise JSONPDDLConversionError(f"Parameters for action '{action_name}' must be objects.")
    name = _require_string(param, "name", f"action '{action_name}' parameter.name")
    typ = str(param.get("type", "object")).strip() or "object"
    if name.startswith("?"):
        clean_name = name[1:]
    else:
        clean_name = name
    return f"?{clean_name}", f"?{clean_name} - {typ}"


def _format_typed_variable(var: Any, context: str) -> str:
    if not isinstance(var, dict):
        raise JSONPDDLConversionError(f"{context} entries must be objects.")
    name = _require_string(var, "name", f"{context}.name")
    typ = str(var.get("type", "object")).strip() or "object"
    clean = name[1:] if name.startswith("?") else name
    return f"?{clean} - {typ}"


# --------------------------------------------------------------------------- #
#  Problem helpers                                                            #
# --------------------------------------------------------------------------- #
def _build_problem(problem: Dict[str, Any], domain_name: str) -> str:
    name = _require_string(problem, "name", "problem.name")
    objects = problem.get("objects", {})
    init = _require_list(problem, "init", "problem.init")
    goal = problem.get("goal", {})
    constraints = _as_list(problem.get("constraints", []))
    metric = problem.get("metric")

    lines: List[str] = [
        f"(define (problem {name})",
        f"  (:domain {domain_name})",
    ]

    if objects:
        if not isinstance(objects, dict):
            raise JSONPDDLConversionError("problem.objects must be a dictionary.")
        lines.extend(_format_objects(objects))

    lines.append("  (:init")
    for item in init:
        entry = _format_init_entry(item)
        if entry:
            lines.append(f"    {entry}")
    lines.append("  )")

    goal_str = _format_goal(goal)
    lines.append(f"  (:goal {goal_str})")

    if constraints:
        constraint_lines = " ".join(str(constraint).strip() for constraint in constraints if str(constraint).strip())
        if constraint_lines:
            lines.append(f"  (:constraints {constraint_lines})")

    if metric:
        lines.append(_format_metric(metric))

    lines.append(")")
    return "\n".join(lines)


def _format_objects(objects: Dict[str, Any]) -> List[str]:
    lines = ["  (:objects"]
    for typ, names in objects.items():
        if isinstance(names, dict):
            # Allow {"type": {"token": "alias"}} – flatten keys.
            tokens = list(names.keys())
        else:
            tokens = _as_list(names)
        if not tokens:
            continue
        joined = " ".join(str(token).strip() for token in tokens)
        lines.append(f"    {joined} - {typ}")
    lines.append("  )")
    return lines


def _format_init_entry(item: Any) -> str:
    if not isinstance(item, dict):
        raise JSONPDDLConversionError("problem.init entries must be objects.")

    if "predicate" in item:
        value = item.get("value", True)
        if value is False:
            raise JSONPDDLConversionError("problem.init does not support negative facts.")
        return _format_literal(item, set())

    if "function" in item:
        function = _require_string(item, "function", "init.function")
        arguments = _as_list(item.get("arguments", []))
        args = " ".join(str(arg).strip() for arg in arguments)
        value = item.get("value")
        if value is None:
            raise JSONPDDLConversionError("Numeric init entries require 'value'.")
        value_str = str(value)
        return f"(= ({function}{(' ' + args) if args else ''}) {value_str})"

    raise JSONPDDLConversionError("problem.init entry must contain 'predicate' or 'function'.")


def _format_goal(goal: Any) -> str:
    if not goal:
        return "()"
    if not isinstance(goal, dict):
        raise JSONPDDLConversionError("problem.goal must be an object.")

    structure = str(goal.get("structure", "conjunctive")).strip().lower()
    conditions = _as_list(goal.get("conditions", []))
    numeric_conditions = _as_list(goal.get("numeric", []))

    logical_parts: List[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise JSONPDDLConversionError("goal.conditions entries must be objects.")
        if condition.get("value", True) is False:
            logical_parts.append(f"(not {_format_literal(condition, set())})")
        else:
            logical_parts.append(_format_literal(condition, set()))

    logical_parts.extend(_format_numeric_condition(entry) for entry in numeric_conditions)

    if not logical_parts:
        return "()"
    if len(logical_parts) == 1:
        base = logical_parts[0]
    else:
        connector = {"conjunctive": "and", "disjunctive": "or"}.get(structure, "and")
        base = f"({connector} {' '.join(logical_parts)})"

    if structure == "negated":
        return f"(not {base})"
    return base


def _format_numeric_condition(entry: Any) -> str:
    if not isinstance(entry, dict):
        raise JSONPDDLConversionError("goal.numeric entries must be objects.")
    op = str(entry.get("op", "=")).strip() or "="
    function = _require_string(entry, "function", "goal.numeric.function")
    arguments = _as_list(entry.get("arguments", []))
    value = entry.get("value")
    if value is None:
        raise JSONPDDLConversionError("goal.numeric entries require 'value'.")
    args = " ".join(str(arg).strip() for arg in arguments)
    return f"({op} ({function}{(' ' + args) if args else ''}) {value})"


def _format_metric(metric: Any) -> str:
    if not isinstance(metric, dict):
        raise JSONPDDLConversionError("problem.metric must be an object.")
    metric_type = str(metric.get("type", "minimize")).strip() or "minimize"
    expression = metric.get("expression")
    if expression is None:
        raise JSONPDDLConversionError("problem.metric requires 'expression'.")
    expr_str = _format_metric_expression(expression)
    return f"  (:metric {metric_type} {expr_str})"


def _format_metric_expression(expr: Any) -> str:
    if isinstance(expr, str):
        return expr
    if isinstance(expr, dict):
        if "function" in expr:
            name = _require_string(expr, "function", "metric.expression.function")
            arguments = _as_list(expr.get("arguments", []))
            args = " ".join(str(arg).strip() for arg in arguments)
            return f"({name}{(' ' + args) if args else ''})"
        if "op" in expr and "arguments" in expr:
            op = str(expr["op"]).strip()
            args = _as_list(expr["arguments"])
            formatted = " ".join(_format_metric_expression(arg) for arg in args)
            return f"({op} {formatted})"
    if isinstance(expr, (int, float)):
        return str(expr)
    raise JSONPDDLConversionError(f"Unsupported metric expression: {expr!r}")


# --------------------------------------------------------------------------- #
#  General utilities                                                          #
# --------------------------------------------------------------------------- #
def _require_string(obj: Dict[str, Any], key: str, context: str) -> str:
    if key not in obj:
        raise JSONPDDLConversionError(f"Missing key '{key}' in {context}.")
    value = obj[key]
    if not isinstance(value, str):
        raise JSONPDDLConversionError(f"{context} must be a string.")
    stripped = value.strip()
    if not stripped:
        raise JSONPDDLConversionError(f"{context} cannot be empty.")
    return stripped


def _require_list(obj: Dict[str, Any], key: str, context: str) -> List[Any]:
    if key not in obj:
        raise JSONPDDLConversionError(f"Missing key '{key}' in {context}.")
    value = obj[key]
    return _as_list(value, context)


def _as_list(value: Any, context: str = "") -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise JSONPDDLConversionError(f"{context or 'Value'} must be a list.")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Convert JSON-PDDL into PDDL files.")
    parser.add_argument("spec", type=pathlib.Path, help="Path to the JSON-PDDL specification.")
    parser.add_argument("--domain-out", type=pathlib.Path, help="Destination for domain PDDL.")
    parser.add_argument("--problem-out", type=pathlib.Path, help="Destination for problem PDDL.")
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the converted domain and problem to stdout when outputs are not specified.",
    )

    args = parser.parse_args(argv)

    with args.spec.open() as f:
        spec = json.load(f)

    domain_pddl, problem_pddl = convert_json_to_pddl(spec)

    if args.domain_out:
        args.domain_out.write_text(domain_pddl + "\n")
    if args.problem_out:
        args.problem_out.write_text(problem_pddl + "\n")

    should_print = args.print or (not args.domain_out and not args.problem_out)
    if should_print:
        print("; --- domain ---")
        print(domain_pddl)
        print("; --- problem ---")
        print(problem_pddl)


if __name__ == "__main__":
    main()
