#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path.cwd() / "ppt"))
from patch_round36 import replace_text, set_notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    best = summary["best_eligible_by_model"]
    expected = {"r2d": (16, 50.678537569059166), "plana": (16, 46.12748484457258), "xens2": (8, 17.62171209060639)}
    for model, (batch, throughput) in expected.items():
        assert best[model]["batch_size"] == batch
        assert best[model]["eligible"] is True
        assert abs(best[model]["throughput_median_seq_per_s"] - throughput) < 1e-12
    rnaformer = [row["throughput_median_seq_per_s"] for row in summary["reference_models"] if row["model"] == "rnaformer"]
    assert len(rnaformer) == 3

    prs = Presentation(args.deck)
    assert len(prs.slides) == 23

    slide = prs.slides[4]
    replace_text(slide,
                 "B=1 两者接近（21.45 vs 22.28）；4090 复现 head-only=117.735 seq/s，历史计数一致，证明非硬件失败。raw-sequence fp32 B=16/8 达 50.68/46.13/17.62，最终 SOTA 待同机基线闭环。",
                 "B=1 两者接近（21.45 vs 22.28）；4090 复现 head-only=117.735 seq/s，历史计数一致，证明非硬件失败。raw-seq fp32 最优 50.68/46.13/17.62；共同协议三 JEV 均在前沿，UFold 未测，不称绝对 SOTA。")
    set_notes(slide,
              "同 B=1 时，head-only=21.45、raw-sequence=22.28 seq/s，二者接近。4090 复现 head-only=117.735 seq/s，TP/FP/FN 与历史一致，证明不是硬件复刻失败。raw-sequence fp32 扩展 batch 达 r2d B=16=50.679、plana B=16=46.127、xens2 B=8=17.622 seq/s；三者均通过数值等价门并处于共同协议精度-吞吐前沿。UFold 未测，因此不宣称无限制绝对速度 SOTA。")

    slide = prs.slides[8]
    replace_text(slide,
                 "同一 4090、TS0 n=1,288、strict micro-F1、raw-sequence 端到端；a 为 B=1 Pareto，b 为 B=1/2/4 吞吐。",
                 "同一 4090、TS0 n=1,288、strict micro-F1、raw-sequence 端到端；a 为 B=1 Pareto，b 为扩展 batch 网格。")
    replace_text(slide,
                 "5 模型 × B=1/2/4 共 15 格；三次全量重复；跨 batch 数值等价（对 B=1 ≥99% 配对一致，|ΔF1|≤1e-3）",
                 "6 系统 18/18 格 + JEV B=16/16/8；三次全量重复；跨 batch ≥99% 配对一致、|ΔF1|≤1e-3")
    replace_text(slide,
                 "当前仅为已测五系统内部 Pareto，不是全模型速度 SOTA；RNAformer/UFold 同机完整点补齐后再判定。旧 124.78 seq/s 为 head-only 历史值，禁止混入 raw-sequence 前沿。",
                 "共同 4090 协议：三 JEV 均处精度-吞吐前沿，最优 50.68/46.13/17.62 seq/s；RNAformer B=1/2/4=2.56/2.72/2.73。UFold 未测，不作绝对 SOTA；旧 124.78 为 head-only。")
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
              "RTX 4090、TS0 n=1288、raw-sequence 端到端。六系统 B=1/2/4 共 18 格已闭环；fp32 扩展 batch 最优为 r2d B=16=50.679、plana B=16=46.127、xens2 B=8=17.622 seq/s，配对一致率分别为 0.995375/0.998024/0.998954，绝对 F1 差分别为 0.000493/0.000119/0.00000029，全部通过等价门。RNAformer B=1/2/4=2.556/2.719/2.733 seq/s。共同协议下三 JEV 均处于精度-吞吐前沿，r2d 最快；UFold 未测，不能宣称无限制绝对速度 SOTA。旧 124.78 为 head-only 历史参照。")

    slide = prs.slides[15]
    replace_text(slide,
                 "4090 raw-seq：r2d 22.28→37.42、xens2 7.47→12.38 seq/s（B=1→4）；速度优化中，当前未证明 SOTA",
                 "4090 raw-seq fp32：r2d B=16=50.68、plana B=16=46.13、xens2 B=8=17.62 seq/s；三 JEV 均处精度-吞吐前沿")
    replace_text(slide,
                 "RNAformer/UFold 同机完整点未闭环；历史 124.78 为 head-only；跨 batch 仅数值等价",
                 "RNAformer B=1/2/4=2.56/2.72/2.73；UFold 未测，不作绝对 SOTA；跨 batch 通过数值等价门")

    slide = prs.slides[18]
    replace_text(slide,
                 "已测五系统内部 Pareto ≠ 全模型速度 SOTA；同机基线未闭环，后续集中做无损推理优化。",
                 "三 JEV 在共同 4090 协议下均处前沿，r2d 最快 50.68 seq/s；UFold 未测，不宣称绝对 SOTA。")
    set_notes(slide,
              "head-only：cached embedding→决策头→必要 DP，不含编码器；forward-only：raw sequence→编码器+任务头，不含 DP；raw-sequence：包含编码器、任务头和必要 DP。4090 复现 head-only=117.735 seq/s，历史计数一致。raw-sequence 下三 JEV 均处于共同协议精度-吞吐前沿，r2d B=16 最快 50.679 seq/s；UFold 未测，不宣称无限制绝对速度 SOTA。")

    slide = prs.slides[21]
    replace_text(slide, "raw-seq B=1：r2d 22.28 seq/s",
                 "raw-seq 最优 50.68/46.13/17.62 seq/s")
    replace_text(slide, "批大小不改变预测", "共同协议三 JEV 均在前沿")
    replace_text(slide,
                 "15/18 格只支持已测五系统内部比较；旧 head-only 124.78 仅历史值，当前未证明速度 SOTA",
                 "RNAformer 已补 B=1/2/4；UFold 未测，不作绝对 SOTA；旧 124.78 为 head-only")

    prs.save(args.out)
    print(f"ROUND36_FINAL_SPEED_DECK {args.out}")


if __name__ == "__main__":
    main()
