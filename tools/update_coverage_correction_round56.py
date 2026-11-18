import hashlib
import json
from pathlib import Path
from pptx import Presentation

from add_evidence_audit_round56 import DECK, WORK, set_notes


def main():
    artifact = WORK / "historical_bidirectional_coverage.json"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == "8688a3fa4b6bdaaf4634d3a757173674bd2670afa144764feec3d0d3ff2362e5"
    report = json.loads(artifact.read_text())
    assert report["raw_detected_by_any_source"] == report["strict_detected_by_any_source"] == 162489
    assert report["queries_lost_from_raw_union"] == 0
    prs = Presentation(DECK)
    assert len(prs.slides) == 29
    slide = prs.slides[27]
    assert slide.shapes[1].text == "三源同源联合：未检出也不等于未见"
    notes = slide.notes_slide.notes_text_frame.text
    assert "每条查询最多保留300个target" in notes
    notes = notes.replace("且每条查询最多保留300个target，", "--max-seqs 300不是最终输出行数上限，已保留全部原始行并独立检查query和target双向80%覆盖。")
    notes += "独立后过滤只排除RNAcentral的9行target覆盖不足命中，三源联合检出仍为162489，未丢失任何查询。更正证据historical_bidirectional_coverage.json，SHA256=8688a3fa…62e5。"
    set_notes(prs, slide, notes)
    prs.save(DECK)
    print(json.dumps({"status": "PASS_COVERAGE_NOTE_CORRECTION", "slides": len(prs.slides),
                      "deck_sha256": hashlib.sha256(DECK.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
