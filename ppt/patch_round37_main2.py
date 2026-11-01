#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path.cwd() / "ppt"))
from patch_round36 import replace_text, set_notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prs = Presentation(args.deck)
    assert len(prs.slides) == 23
    replace_text(prs.slides[0], "2026-10-31 · Round 36", "2026-11-01 · Round 37")
    replace_text(prs.slides[15], "结论口径：Round 36 统一复核", "结论口径：Round 37 统一复核")
    replace_text(prs.slides[-1], "RNA-JEV / ROUND 36", "RNA-JEV / ROUND 37")

    slide = prs.slides[4]
    replace_text(slide, "主图 2｜124.78 是 head-only 峰值",
                 "主图 2｜RTX 4090 端到端推理吞吐：最快合格配置")
    replace_text(slide,
                 "RTX 4090、TS0；JEV 梯图为 cached-embedding/head-only，812,375 head；非 raw-sequence 端到端。",
                 "六系统、TS0 n=1,288、raw-sequence 全链路；每模型取预测等价门内的最快配置。")
    replace_text(slide, "台账 §15.79–15.82；speed_v3/v4 JSON",
                 "round37 源 JSON；全 TS0 三次重复；raw-sequence 端到端")
    replace_text(slide,
                 "B=1 两者接近（21.45 vs 22.28）；4090 复现 head-only=117.735 seq/s，历史计数一致，证明非硬件失败。raw-seq fp32 最优 50.68/46.13/17.62；共同协议三 JEV 均在前沿，UFold 未测，不称绝对 SOTA。",
                 "r2d B=16=50.68、plana B=16=46.13、xens2 B=8=17.62 seq/s；共同协议三 JEV 均在精度-吞吐前沿。UFold 未测，不作无限制绝对 SOTA。")
    pictures = [shape for shape in slide.shapes if shape.shape_type == 13]
    assert len(pictures) == 1
    picture = pictures[0]
    assert abs(picture.left / 914400 - 1.3665004374453193) < 1e-9
    assert abs(picture.top / 914400 - 1.48) < 1e-9
    assert abs(picture.width / 914400 - 10.6) < 1e-9
    assert abs(picture.height / 914400 - 5.3) < 1e-9
    _, relationship_id = slide.part.get_or_add_image_part(str(args.image))
    picture._element.blipFill.blip.rEmbed = relationship_id
    set_notes(slide,
              "主图 2 只展示 raw-sequence 端到端速度：原始序列经过编码器、任务头与必要解码，不使用缓存表征。每模型选择预测等价门内最快配置，r2d B=16=50.679、plana B=16=46.127、xens2 B=8=17.622 seq/s；RiNALMo-ft B=4=23.804、MXfold2 B=2=14.760、RNAformer B=4=2.733。全 TS0 三次重复，误差线为范围。共同协议三 JEV 均处精度-吞吐前沿，UFold 未测，不宣称无限制绝对速度 SOTA。")
    prs.save(args.out)
    print(f"ROUND37_MAIN2_DECK_WRITTEN {args.out}")


if __name__ == "__main__":
    main()
