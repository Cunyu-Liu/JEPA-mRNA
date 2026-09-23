"""Manuscript linter for the honesty / boundary rules of spec §2.3, §2.4, §9.2, §9.3.

Four checks (each independently selectable with ``--checks``):

(a) **forbidden claims** -- none of the six statements listed in spec §2.4 /
    ``spec/constraints.md`` §2 may appear as an assertion.  The six are: claiming to
    have first proposed the Gibbs / partition-function framework; claiming the
    cotranscriptional decision order as an innovation; claiming "illegal-structure
    rate = 0" as a contribution; claiming ``MLP_T = 0`` is equivalent to
    ViennaRNA; claiming to reproduce Jev's RLCD; claiming better algorithmic
    complexity than LinearFold.
(b) **non-claimed devices in a contribution sentence** -- the five implementation
    devices of spec §2.3 (Gibbs framework / exact likelihood, Turner residual,
    constructive symmetry, implicitly-differentiated differentiable DP,
    multi-channel SHAPE/DMS evidence) must not appear inside a sentence that also
    makes a contribution claim.
(c) **missing Q1-Q12 landing points** -- every anticipated objection of spec §9.3
    must have an explicit landing point in the manuscript.  Spec: "Q1-Q12 每一条都
    必须在稿件中有明确落点。任何一条无落点，即视为未准备好投稿."
(d) **forbidden generalization phrasing** -- the claim must be "smaller OOD
    degradation (more robust)", never "higher OOD accuracy" (spec §0.9.3, see
    ``paper/generalization_claim.md``).

Negation handling
-----------------
(a) and (d) are skipped when the match is negated ("we do **not** claim ...",
"不主张 ..."), so an honest disclaimer is not flagged.  (b) is skipped when the
sentence disclaims the contribution ("... are not our contribution").

Boundary note: the guard documents themselves (``generalization_claim.md``,
``contributions.md``, ``limitations.md``, ``reviewer_objections.md``) quote the
forbidden phrasings as counter-examples.  Do not lint them as manuscripts.

Usage::

    python paper/check_manuscript.py --manuscript paper/manuscript_stub.md
    python paper/check_manuscript.py --manuscript draft.md --checks a,b,d
    python paper/check_manuscript.py --objections paper/reviewer_objections.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ALLOWED_GENERALIZATION_PATTERNS",
    "CONTRIBUTION_MARKERS",
    "FORBIDDEN_CLAIMS",
    "FORBIDDEN_OOD_PATTERNS",
    "NON_CLAIMED_DEVICES",
    "OBJECTION_IDS",
    "UNFILLED_MARKERS",
    "check_generalization_wording",
    "check_objections_landing_points",
    "find_forbidden_claims",
    "find_forbidden_ood_phrasing",
    "find_non_claimed_devices_in_contribution_sentences",
    "iter_sentences",
    "lint_manuscript",
    "main",
    "parse_objections_table",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER_DIR = _HERE
DEFAULT_MANUSCRIPT = os.path.join(_PAPER_DIR, "manuscript_stub.md")
DEFAULT_OBJECTIONS = os.path.join(_PAPER_DIR, "reviewer_objections.md")

OBJECTION_IDS: Tuple[str, ...] = tuple(f"Q{i}" for i in range(1, 13))


# ---------------------------------------------------------------------------
# rule tables
# ---------------------------------------------------------------------------
#: The six forbidden claims (spec §2.4 / constraints.md §2).
FORBIDDEN_CLAIMS: Tuple[Dict[str, object], ...] = (
    {
        "id": "no_gibbs_framework_first",
        "rule": "不主张首创 Gibbs / 配分函数框架 (spec §2.4 #1)",
        "patterns": (
            r"(first|earliest|original|the\s+first)\s+(to\s+)?(propose|introduce|present|"
            r"formulate|develop|use)\b[^.]{0,70}\b(gibbs|partition[- ]function|"
            r"log[- ]linear\s+(crf|model))",
            r"\bnovel\b[^.]{0,40}\b(gibbs\s+framework|partition[- ]function\s+framework)",
            r"首创[^。]{0,14}(Gibbs|配分函数)",
            r"首次(提出|引入|将|把)[^。]{0,20}(Gibbs|配分函数)",
            r"(Gibbs|配分函数)[^。]{0,10}(框架|模型)[^。]{0,10}(首创|首次|原创)",
        ),
    },
    {
        "id": "no_cotranscriptional_innovation",
        "rule": "不主张共转录决策顺序是创新 (spec §2.4 #2)",
        "patterns": (
            r"(co[- ]?transcriptional|5'\s*(->|to|→)\s*3')[^.]{0,70}"
            r"\b(contribution|novel|innovat|first|key\s+insight)",
            r"\b(contribution|novel|innovat)[^.]{0,70}"
            r"(co[- ]?transcriptional|5'\s*(->|to|→)\s*3')",
            r"共转录[^。]{0,18}(创新|贡献|首创|亮点|新颖)",
            r"(创新|贡献|首创|亮点)[^。]{0,18}共转录",
        ),
    },
    {
        "id": "no_illegal_rate_contribution",
        "rule": '不主张"非法结构率 = 0"是贡献 (spec §2.4 #3)',
        "patterns": (
            r"(illegal[- ]structure\s+rate|legality\s+rate|非法结构率)[^.]{0,60}"
            r"\b(contribution|novel|innovat|contribute|key\s+result|main\s+result)",
            r"\b(contribution|novel|innovat)[^.]{0,60}"
            r"(illegal[- ]structure\s+rate|非法结构率)",
            r"非法结构率[^。]{0,20}(是|为|构成|作为)[^。]{0,10}(贡献|创新|亮点)",
        ),
    },
    {
        "id": "no_mlp_t_equals_vienna",
        "rule": "不主张 MLP_T=0 等价于 ViennaRNA (spec §2.4 #4)",
        "patterns": (
            r"(MLP_?T|turner\s+residual|zero[- ]initiali[sz]ed\s+residual|"
            r"zeroing\s+the\s+(turner\s+)?residual)[^.]{0,70}"
            r"(equivalent\s+to|equals?|is\s+exactly|recovers?|reproduces?|reduces?\s+to)"
            r"[^.]{0,25}vienna",
            r"(equivalent|equal)\s+to\s+vienna",
            r"等价于[^。]{0,12}ViennaRNA",
            r"MLP_?T\s*=\s*0[^。]{0,24}(等价|就是|即)",
        ),
    },
    {
        "id": "no_reproduce_jev_rlcd",
        "rule": "不主张复现了 Jev 的 RLCD (spec §2.4 #5)",
        "patterns": (
            r"(reproduce[sd]?|reproducing|replication\s+of|match(es|ed)?|equivalent\s+to|"
            r"outperform(s|ed)?|improve[sd]?\s+on|surpass(es|ed)?)[^.]{0,40}"
            r"jev[^.]{0,25}rlcd",
            r"jev[^.]{0,25}rlcd[^.]{0,50}"
            r"(reproduc|match|equivalen|outperform|surpass|improve)",
            r"(复现|等价于|超越|优于|达到)[^。]{0,12}(Jev|RLCD)",
            r"RLCD[^。]{0,18}(复现|等价)",
        ),
    },
    {
        "id": "no_better_complexity_than_linearfold",
        "rule": "不主张在算法复杂度上优于 LinearFold (spec §2.4 #6)",
        "patterns": (
            r"(better|lower|less|improved|superior|smaller|faster)[^.]{0,40}"
            r"complexity[^.]{0,40}linearfold",
            r"complexity[^.]{0,50}(better|lower|superior)[^.]{0,40}linearfold",
            r"(优于|低于|好于|快于)[^。]{0,12}LinearFold",
            r"复杂度[^。]{0,24}(优于|低于|好于)[^。]{0,14}LinearFold",
        ),
    },
)

#: The five implementation devices that must not appear in a contribution sentence
#: (spec §2.3).  Patterns are deliberately narrow: "partition function" alone is
#: unavoidable when describing the exact marginals, so the device is keyed on the
#: *framework* wording instead.
NON_CLAIMED_DEVICES: Tuple[Dict[str, object], ...] = (
    {"id": "gibbs_framework",
     "label": "Gibbs framework / exact likelihood",
     "patterns": (r"\bgibbs\b", r"log[- ]linear\s+(crf|model)",
                  r"exact\s+likelihood", r"精确似然", r"配分函数框架")},
    {"id": "turner_residual",
     "label": "Turner residual prior (MLP_T)",
     "patterns": (r"turner\s+residual", r"MLP_?T", r"turner\s+残差")},
    {"id": "constructive_symmetry",
     "label": "constructive symmetry",
     "patterns": (r"constructive\s+symmetr", r"构造性对称")},
    {"id": "implicit_diff_dp",
     "label": "implicitly-differentiated differentiable DP",
     "patterns": (r"implicit(ly)?\s+differentiat", r"隐式微分",
                  r"differentiable\s+nussinov", r"soft[- ]nussinov")},
    {"id": "multi_channel_evidence",
     "label": "multi-channel evidence (SHAPE/DMS)",
     "patterns": (r"\bSHAPE\b", r"\bDMS\b", r"multi[- ]channel",
                  r"probing\s+channel", r"reactivity\s+channel", r"多通道证据")},
)

#: Markers that make a sentence a *contribution* sentence.
CONTRIBUTION_MARKERS: Tuple[str, ...] = (
    r"\bcontribution(s)?\b", r"\bcontribute(s|d)?\b", r"\bwe\s+propose\b",
    r"\bwe\s+present\b", r"\bwe\s+introduce\b", r"\bwe\s+show\b",
    r"\bnovel(ty)?\b", r"\bkey\s+insight\b", r"\bour\s+work\b",
    r"贡献", r"创新", r"首创", r"本文提出",
)

#: Phrases that make a sentence an explicit *disclaimer* rather than a claim.
_DISCLAIMER_PATTERNS: Tuple[str, ...] = (
    r"not\s+(a|our|the)\s+contribution", r"are\s+not\s+our\s+contribution",
    r"is\s+not\s+a\s+contribution", r"not\s+claimed", r"we\s+do\s+not\s+claim",
    r"does\s+not\s+claim", r"no\s+contribution\s+claim",
    r"不(是|构成|主张|声称)[^。]{0,8}贡献", r"并非[^。]{0,8}贡献",
    r"不作[^。]{0,8}贡献", r"不主张",
)

#: Forbidden generalization phrasings (spec §0.9.3).
FORBIDDEN_OOD_PATTERNS: Tuple[str, ...] = (
    r"higher\s+(out[- ]of[- ]distribution|ood)\s+accuracy",
    r"better\s+(out[- ]of[- ]distribution|ood)\s+accuracy",
    r"improv(e|es|ed|ing)\s+(the\s+)?(out[- ]of[- ]distribution|ood)\s+accuracy",
    r"higher\s+cross[- ]family\s+(accuracy|performance)",
    r"superior\s+cross[- ]family\s+(accuracy|performance)",
    r"cross[- ]family\s+(accuracy|performance)\s+(surpass|exceed|outperform)",
    r"(ood|out[- ]of[- ]distribution)[^.]{0,24}精度(更高|提升)",
    r"跨家族(精度|表现)[^。]{0,8}(超越|更高|高于|超过)",
    r"OOD\s*精度更高",
)

#: The wording the claim must actually use.
ALLOWED_GENERALIZATION_PATTERNS: Tuple[str, ...] = (
    r"smaller\s+(out[- ]of[- ]distribution|ood)\s+degradation",
    r"more\s+robust",
    r"degrades?\s+less",
    r"less\s+(out[- ]of[- ]distribution|ood)\s+degradation",
    r"衰减更小", r"更鲁棒", r"退化更小",
)

#: Cells that mean "the landing point has not been written yet".
UNFILLED_MARKERS: Tuple[str, ...] = (
    "", "-", "—", "–", "n/a", "N/A", "tbd", "TBD", "待填", "待补", "待核验",
    "todo", "TODO", "?", "??",
)

_NEGATIONS: Tuple[str, ...] = (
    "not ", "n't", "never", "no ", "without", "rather than", "instead of",
    "不得", "不主张", "不是", "并非", "不能", "不构成", "未声称",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def iter_sentences(text: str) -> Iterable[Tuple[int, str]]:
    """Yield ``(line_number, sentence)`` pairs (1-based line numbers)."""
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        for piece in re.split(r"(?<=[.!?。！？])\s*", stripped):
            piece = piece.strip()
            if piece:
                yield lineno, piece


def _is_negated(sentence: str, start: int, end: Optional[int] = None,
                window: int = 45) -> bool:
    """True when the match is negated.

    Two-sided on purpose: a disclaimer can precede the match ("we do **not** claim
    to be the first ...") or sit *inside* it ("our complexity is **not** necessarily
    better than LinearFold"), and both must suppress the finding -- otherwise the
    linter would punish exactly the honest sentences it exists to encourage.
    """
    prefix = sentence[max(0, start - window):start].lower()
    if any(word in prefix for word in _NEGATIONS):
        return True
    if end is not None:
        span = sentence[start:end].lower()
        if any(word in span for word in _NEGATIONS):
            return True
    return False


def _sentence_disclaims(sentence: str) -> bool:
    low = sentence.lower()
    return any(re.search(pattern, low) for pattern in _DISCLAIMER_PATTERNS)


def _find(text: str, patterns: Sequence[str], *, skip_negated: bool) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    for lineno, sentence in iter_sentences(text):
        for pattern in patterns:
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                if skip_negated and _is_negated(sentence, match.start(), match.end()):
                    continue
                findings.append({
                    "line": lineno,
                    "pattern": pattern,
                    "match": match.group(0),
                    "sentence": sentence,
                })
                break
    return findings


# ---------------------------------------------------------------------------
# check (a): forbidden claims
# ---------------------------------------------------------------------------
def find_forbidden_claims(text: str) -> List[Dict[str, object]]:
    """Return every forbidden-claim assertion found in ``text``."""
    findings: List[Dict[str, object]] = []
    for claim in FORBIDDEN_CLAIMS:
        for hit in _find(text, claim["patterns"], skip_negated=True):
            findings.append({"id": claim["id"], "rule": claim["rule"], **hit})
    return findings


# ---------------------------------------------------------------------------
# check (b): non-claimed devices inside contribution sentences
# ---------------------------------------------------------------------------
def find_non_claimed_devices_in_contribution_sentences(
        text: str) -> List[Dict[str, object]]:
    """Flag a contribution sentence that also names a non-claimed device."""
    findings: List[Dict[str, object]] = []
    marker_re = re.compile("|".join(CONTRIBUTION_MARKERS), flags=re.IGNORECASE)
    for lineno, sentence in iter_sentences(text):
        marker = marker_re.search(sentence)
        if marker is None:
            continue
        if _sentence_disclaims(sentence):
            continue
        for device in NON_CLAIMED_DEVICES:
            device_re = re.compile("|".join(device["patterns"]), flags=re.IGNORECASE)
            hit = device_re.search(sentence)
            if hit is None:
                continue
            findings.append({
                "id": device["id"],
                "label": device["label"],
                "line": lineno,
                "marker": marker.group(0),
                "device_match": hit.group(0),
                "sentence": sentence,
            })
    return findings


# ---------------------------------------------------------------------------
# check (c): Q1-Q12 landing points
# ---------------------------------------------------------------------------
def parse_objections_table(path: str) -> Dict[str, object]:
    """Parse the Q1-Q12 table and report which landing points are still unfilled.

    The landing column is located by header text (``landing`` / ``落点``); when no
    such header exists the last column is used.  A row is unfilled when its landing
    cell is empty or one of :data:`UNFILLED_MARKERS`.
    """
    if not os.path.isfile(path):
        return {"found": False, "path": path, "reason": "file does not exist",
                "rows": {}, "missing": list(OBJECTION_IDS), "all_filled": False}
    lines = _read(path).splitlines()

    header_index: Optional[int] = None
    landing_col: Optional[int] = None
    for index, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if any("landing" in c.lower() or "落点" in c for c in cells):
            header_index = index
            landing_col = next(i for i, c in enumerate(cells)
                               if "landing" in c.lower() or "落点" in c)
            break
    if header_index is None:
        return {"found": False, "path": path,
                "reason": "no table with a 'landing point' / '落点' column",
                "rows": {}, "missing": list(OBJECTION_IDS), "all_filled": False}

    rows: Dict[str, Dict[str, object]] = {}
    for line in lines[header_index + 2:]:
        if not line.strip().startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if landing_col >= len(cells):
            continue
        id_match = re.search(r"\bQ(\d{1,2})\b", " ".join(cells[:2]))
        if id_match is None:
            continue
        qid = f"Q{id_match.group(1)}"
        landing = cells[landing_col]
        rows[qid] = {
            "landing": landing,
            "filled": landing not in UNFILLED_MARKERS and "待填" not in landing,
        }

    missing = [qid for qid in OBJECTION_IDS
               if qid not in rows or not rows[qid]["filled"]]
    return {"found": True, "path": path, "rows": rows, "missing": missing,
            "n_rows": len(rows), "all_filled": not missing,
            "landing_column": landing_col}


def check_objections_landing_points(path: str) -> Dict[str, object]:
    """The spec §9.3 pre-submission self-check: all twelve must have a landing point."""
    report = parse_objections_table(path)
    report["ok"] = bool(report.get("all_filled"))
    return report


# ---------------------------------------------------------------------------
# check (d): generalization wording
# ---------------------------------------------------------------------------
def find_forbidden_ood_phrasing(text: str) -> List[Dict[str, object]]:
    """Return every forbidden "higher OOD accuracy" style phrase."""
    return _find(text, FORBIDDEN_OOD_PATTERNS, skip_negated=True)


def check_generalization_wording(text: str) -> Dict[str, object]:
    """Guard for the spec §0.9.3 wording rule.

    ``ok`` is ``True`` only when no forbidden phrasing is present, whatever else
    the text says.  ``allowed_hits`` lists the occurrences of the sanctioned
    "smaller degradation / more robust" wording, so a caller can confirm the claim
    is actually expressed in the defensible form.
    """
    findings = find_forbidden_ood_phrasing(text)
    allowed_hits = _find(text, ALLOWED_GENERALIZATION_PATTERNS, skip_negated=False)
    return {
        "ok": not findings,
        "findings": findings,
        "allowed_hits": allowed_hits,
        "rule": ('the OOD claim must be "smaller OOD degradation (more robust)", '
                 'never "higher OOD accuracy" (spec §0.9.3)'),
    }


# ---------------------------------------------------------------------------
# top-level lint
# ---------------------------------------------------------------------------
def _parse_checks(spec: str) -> set:
    """``"abcd"`` / ``"a,b,d"`` / ``"a b d"`` -> ``{"a", "b", "d"}``."""
    selected = set()
    for token in re.split(r"[,\s]+", spec.strip()):
        if not token:
            continue
        if token.lower() in {"a", "b", "c", "d"}:
            selected.add(token.lower())
            continue
        # A run like "abcd" means "all four".
        chars = {ch for ch in token.lower() if ch in "abcd"}
        if not chars:
            raise ValueError(f"unknown check {token!r}; expected a subset of 'abcd'")
        selected |= chars
    return selected


def lint_manuscript(manuscript_path: str, objections_path: Optional[str] = None,
                    checks: str = "abcd") -> Dict[str, object]:
    """Run the selected checks and return a structured, JSON-serialisable report."""
    text = _read(manuscript_path)
    selected = _parse_checks(checks)
    report: Dict[str, object] = {
        "manuscript": manuscript_path,
        "checks_run": sorted(selected),
        "ok": True,
    }

    if "a" in selected:
        findings = find_forbidden_claims(text)
        report["forbidden_claims"] = findings
        report["a_ok"] = not findings

    if "b" in selected:
        findings = find_non_claimed_devices_in_contribution_sentences(text)
        report["non_claimed_devices_in_contribution_sentence"] = findings
        report["b_ok"] = not findings

    if "c" in selected:
        target = objections_path
        if not target:
            # Prefer a Q1-Q12 table inside the manuscript itself; otherwise fall
            # back to the shipped objections file.
            inline = parse_objections_table(manuscript_path)
            target = manuscript_path if inline.get("found") else DEFAULT_OBJECTIONS
        objections = check_objections_landing_points(target)
        report["objections"] = objections
        report["c_ok"] = bool(objections["ok"])

    if "d" in selected:
        guard = check_generalization_wording(text)
        report["generalization_wording"] = guard
        report["d_ok"] = bool(guard["ok"])

    report["ok"] = all(bool(report.get(f"{c}_ok", True)) for c in selected)
    report["failed_checks"] = sorted(c for c in selected if not report.get(f"{c}_ok", True))
    return report


def format_report(report: Dict[str, object]) -> str:
    lines = [f"manuscript lint: {report['manuscript']}",
             f"checks run      : {','.join(report['checks_run'])}", ""]

    def verdict(key: str) -> str:
        return "PASS" if report.get(key, True) else "FAIL"

    if "a" in report["checks_run"]:
        lines.append(f"  (a) forbidden claims                     : {verdict('a_ok')}")
        for hit in report.get("forbidden_claims", []):
            lines.append(f"        line {hit['line']}: [{hit['id']}] {hit['match']!r}")
    if "b" in report["checks_run"]:
        lines.append(f"  (b) non-claimed device in contribution   : {verdict('b_ok')}")
        for hit in report.get("non_claimed_devices_in_contribution_sentence", []):
            lines.append(f"        line {hit['line']}: [{hit['id']}] "
                         f"marker={hit['marker']!r} device={hit['device_match']!r}")
    if "c" in report["checks_run"]:
        obj = report["objections"]
        lines.append(f"  (c) Q1-Q12 landing points                : {verdict('c_ok')}")
        if not obj.get("found"):
            lines.append(f"        {obj.get('reason', 'table not found')} ({obj['path']})")
        elif obj["missing"]:
            lines.append(f"        unfilled: {', '.join(obj['missing'])} "
                         f"({len(obj['missing'])}/{len(OBJECTION_IDS)})")
        else:
            lines.append(f"        all {len(OBJECTION_IDS)} landing points filled")
    if "d" in report["checks_run"]:
        guard = report["generalization_wording"]
        lines.append(f"  (d) generalization wording               : {verdict('d_ok')}")
        for hit in guard["findings"]:
            lines.append(f"        line {hit['line']}: {hit['match']!r}")
        if guard["allowed_hits"]:
            lines.append(f"        sanctioned wording present: "
                         f"{sorted({h['match'] for h in guard['allowed_hits']})}")
    lines += ["", f"  OVERALL: {'PASS' if report['ok'] else 'FAIL'}"
                  + (f" (failed: {','.join(report['failed_checks'])})"
                     if report["failed_checks"] else "")]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manuscript honesty / boundary linter")
    p.add_argument("--manuscript", default=DEFAULT_MANUSCRIPT)
    p.add_argument("--objections", default="",
                   help="file holding the Q1-Q12 table (default: inside the manuscript, "
                        "else paper/reviewer_objections.md)")
    p.add_argument("--checks", default="abcd",
                   help="subset of a,b,c,d (comma or space separated)")
    p.add_argument("--json", action="store_true", help="emit the raw JSON report")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.manuscript):
        print(f"FATAL: manuscript not found: {args.manuscript}", file=sys.stderr)
        return 2
    report = lint_manuscript(args.manuscript,
                             objections_path=args.objections or None,
                             checks=args.checks)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
