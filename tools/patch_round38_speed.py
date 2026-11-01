#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from pptx import Presentation

from patch_round36 import normalize_font_and_spacing, replace_text, set_notes


def replace_picture(slide, png):
    old = next(shape for shape in slide.shapes if shape.shape_type == 13)
    new = slide.shapes.add_picture(str(png), old.left, old.top, old.width, old.height)
    old.element.addprevious(new.element)
    old.element.getparent().remove(old.element)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    validation = json.loads((root / "fig05_speed/round38_speed_validation.json").read_text())
    assert validation["status"] == "PASS" and validation["legacy_012_counts_match"]
    source = json.loads((root / "fig02_speed/fig02_end_to_end_speed_4090_round38_source.json").read_text())
    efold = next(row for row in source["models"] if row["model"] == "efold")
    eterna = next(row for row in source["models"] if row["model"] == "eternafold")
    prs = Presentation(args.deck)
    assert len(prs.slides) == 23

    slide = prs.slides[4]
    replace_picture(slide, root / "fig02_speed/fig02_end_to_end_speed_4090_round38_3840x1920.png")
    replace_text(slide, "六系统、TS0 n=1,288、raw-sequence 全链路；每模型取预测等价门内的最快配置。",
                 "八系统（7 GPU + 1 CPU）、TS0 n=1,288、raw-sequence 全链路；GPU 取预测等价门内最快配置。")
    replace_text(slide, "round37 源 JSON；全 TS0 三次重复；raw-sequence 端到端",
                 "round38 源 JSON；全 TS0 三次重复；raw-sequence 端到端；三项权重 blocked")
    replace_text(slide,
                 "r2d/plana/xens2 B=32 分别达 53.91/47.92/19.89 seq/s；共同协议三 JEV 均在精度-吞吐前沿。eFold/UFold/BPfold/E2Efold/EternaFold 扩展测速中，不作无限制绝对 SOTA。",
                 f"r2d 最快 53.91 seq/s；eFold 0.1.3 B=4={efold['throughput']:.2f}，EternaFold CPU={eterna['throughput']:.2f}。BPfold/UFold/E2Efold 权重 blocked，不作绝对 SOTA。")
    set_notes(slide,
              f"eFold 0.1.3 B4 median={efold['throughput']:.6f} seq/s，F1={efold['f1']:.6f}；EternaFold 官方 serial CPU median={eterna['throughput']:.6f} seq/s，F1={eterna['f1']:.6f}。eFold 0.1.2 旧板 TP/FP/FN=23498/25358/11393 已独立复现；0.1.3 与 0.1.2 权重和预测不同，不能混称同一版本。BPfold/UFold/E2Efold 缺官方权重，保持 blocked。")

    slide = prs.slides[8]
    replace_picture(slide, root / "fig05_speed/fig05_accuracy_speed_4090_round38_3840x1920.png")
    replace_text(slide, "6 系统 18/18 格 + JEV B=32 饱和点；三次全量重复；跨 batch ≥99% 配对一致、|ΔF1|≤1e-3",
                 "原 18 格 + eFold 3 格 + EternaFold CPU 1 格；三次全量重复；跨 batch ≥99% 配对一致、|ΔF1|≤1e-3")
    replace_text(slide,
                 "共同 4090 协议：三 JEV 均处精度-吞吐前沿，B=32 最优 53.91/47.92/19.89 seq/s；RNAformer B=1/2/4=2.56/2.72/2.73。eFold/UFold/BPfold/E2Efold/EternaFold 扩展测速中，不作绝对 SOTA；TF32 不替代更快 B=32。",
                 f"三 JEV 仍处前沿；eFold B=1/2/4=16.20/16.29/{efold['throughput']:.2f} seq/s，配对一致≥99.97%；EternaFold CPU={eterna['throughput']:.2f}。三项 blocked，不作绝对 SOTA。")
    set_notes(slide,
              "eFold B2/B4 相对 B1 配对一致率=0.999770/0.999732，F1 极差=0.0000356，符合数值等价门。EternaFold 是单线程 CPU，B2/B4 不适用于官方 serial CLI，只在 a 面板展示。")

    replace_text(prs.slides[18], "五模型扩展测速中，不宣称绝对 SOTA",
                 "eFold/EternaFold 已测；三项权重 blocked，不宣称绝对 SOTA")
    replace_text(prs.slides[21], "五模型扩展测速中，不作绝对 SOTA",
                 "eFold/EternaFold 已测；三项 blocked，不作绝对 SOTA")
    normalize_font_and_spacing(prs)
    prs.save(args.out)
    print(f"ROUND38_SPEED_DECK_WRITTEN {args.out}")


if __name__ == "__main__":
    main()
