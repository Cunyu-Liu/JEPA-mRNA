#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path.cwd() / "ppt"))
from patch_round36 import replace_text, set_notes


def replace_image(slide, image):
    pictures = [shape for shape in slide.shapes if shape.shape_type == 13]
    assert len(pictures) == 1
    picture = pictures[0]
    assert abs(picture.left / 914400 - 1.3665004374453193) < 1e-9
    assert abs(picture.top / 914400 - 1.48) < 1e-9
    assert abs(picture.width / 914400 - 10.6) < 1e-9
    assert abs(picture.height / 914400 - 5.3) < 1e-9
    _, relationship_id = slide.part.get_or_add_image_part(str(image))
    picture._element.blipFill.blip.rEmbed = relationship_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--fig02", type=Path, required=True)
    parser.add_argument("--fig05", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prs = Presentation(args.deck)
    assert len(prs.slides) == 23

    slide = prs.slides[4]
    replace_text(slide,
                 "r2d B=16=50.68、plana B=16=46.13、xens2 B=8=17.62 seq/s；共同协议三 JEV 均在精度-吞吐前沿。UFold 未测，不作无限制绝对 SOTA。",
                 "r2d/plana/xens2 B=32 分别达 53.91/47.92/19.89 seq/s；共同协议三 JEV 均在精度-吞吐前沿。UFold 未测，不作无限制绝对 SOTA。")
    replace_image(slide, args.fig02)
    set_notes(slide,
              "主图 2 只展示 raw-sequence 端到端速度。新一轮全 TS0 三次重复后，最优合格配置更新为 r2d B=32=53.907、plana B=32=47.924、xens2 B=32=19.892 seq/s；RiNALMo-ft B=4=23.804、MXfold2 B=2=14.760、RNAformer B=4=2.733。共同协议三 JEV 均处精度-吞吐前沿。r2d TF32 B=16=52.426 seq/s，虽通过等价门但慢于 fp32 B=32，不选作最优。UFold 未测，不宣称无限制绝对速度 SOTA。")

    slide = prs.slides[8]
    replace_text(slide,
                 "6 系统 18/18 格 + JEV B=16/16/8；三次全量重复；跨 batch ≥99% 配对一致、|ΔF1|≤1e-3",
                 "6 系统 18/18 格 + JEV B=32 饱和点；三次全量重复；跨 batch ≥99% 配对一致、|ΔF1|≤1e-3")
    replace_text(slide,
                 "共同 4090 协议：三 JEV 均处精度-吞吐前沿，最优 50.68/46.13/17.62 seq/s；RNAformer B=1/2/4=2.56/2.72/2.73。UFold 未测，不作绝对 SOTA；旧 124.78 为 head-only。",
                 "共同 4090 协议：三 JEV 均处精度-吞吐前沿，B=32 最优 53.91/47.92/19.89 seq/s；RNAformer B=1/2/4=2.56/2.72/2.73。UFold 未测，不作绝对 SOTA；TF32 不替代更快 B=32。")
    replace_image(slide, args.fig05)
    set_notes(slide,
              "RTX 4090、TS0 n=1288、raw-sequence 端到端。新增 xens2 B=16=19.452 seq/s，三模型 B=32 分别为 r2d=53.907、plana=47.924、xens2=19.892 seq/s，均通过全 TS0 三次重复和预测等价门。r2d TF32 B=16=52.426 seq/s，配对一致率 0.999712、绝对 F1 差 0.0000365，通过但慢于 fp32 B=32，不选作最优。共同协议三 JEV 均处精度-吞吐前沿，r2d 最快；UFold 未测，不宣称无限制绝对速度 SOTA。")

    slide = prs.slides[15]
    replace_text(slide,
                 "4090 raw-seq fp32：r2d B=16=50.68、plana B=16=46.13、xens2 B=8=17.62 seq/s；三 JEV 均处精度-吞吐前沿",
                 "4090 raw-seq fp32：三模型 B=32 分别达 53.91/47.92/19.89 seq/s；三 JEV 均处精度-吞吐前沿")

    slide = prs.slides[18]
    replace_text(slide,
                 "三 JEV 在共同 4090 协议下均处前沿，r2d 最快 50.68 seq/s；UFold 未测",
                 "三 JEV 在共同 4090 协议下均处前沿，r2d B=32 最快 53.91 seq/s；UFold 未测，不宣称绝对 SOTA。")
    set_notes(slide,
              "head-only：cached embedding→决策头→必要 DP，不含编码器；forward-only：raw sequence→编码器+任务头，不含 DP；raw-sequence：包含编码器、任务头和必要 DP。主图 2 已只使用 raw-sequence，历史口径仅留本附录。新一轮 r2d B=32=53.907 seq/s 为当前最快合格配置；TF32 B=16=52.426，未替代更快 B=32。UFold 未测，不宣称无限制绝对速度 SOTA。")

    slide = prs.slides[21]
    replace_text(slide, "raw-seq 最优 50.68/46.13/17.62 seq/s",
                 "raw-seq 最优 53.91/47.92/19.89 seq/s")
    prs.save(args.out)
    print(f"ROUND37_SPEED_DECK_WRITTEN {args.out}")


if __name__ == "__main__":
    main()
