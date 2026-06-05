import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Luke Jansen
# 6/3/2026
# Test SAM2 on an image
# Run from the contact_reasoning directory.

checkpoint = "sam2_repo/checkpoints/sam2.1_hiera_large.pt" # use normal file path 
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # DO NOT USE normal file path -
# Hydra then looks for a primary config with that exact name in the package tree —
# - it does not treat it as a filesystem path under contact_reasoning.

sam2_model = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2_model)


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_point_coords(s: str) -> np.ndarray:
    """'500,375' -> (1,2); '500,375;600,400' -> (2,2)"""
    parts = [p.strip() for p in s.split(";") if p.strip()]
    rows = [_parse_floats(p) for p in parts]
    for r in rows:
        if len(r) != 2:
            raise argparse.ArgumentTypeError(
                f"Each point must be x,y; got {s!r}"
            )
    return np.array(rows, dtype=np.float64)


def parse_point_labels(s: str) -> np.ndarray:
    return np.array(_parse_floats(s), dtype=np.int64)


def parse_box(s: str) -> np.ndarray:
    vals = _parse_floats(s)
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2 (four numbers)")
    return np.array([vals], dtype=np.float64)


def parse_mask_input(s: str) -> np.ndarray:
    """Path to .npy (Bx1xHxW logits) or comma-separated floats (advanced)."""
    if os.path.isfile(s):
        return np.load(s)
    vals = _parse_floats(s)
    return np.array(vals, dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SAM2 image predictor on one image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image_path", help="Path to input image")
    p.add_argument(
        "--point-coords",
        type=parse_point_coords,
        default=parse_point_coords("500,375"),
        help="Points as x,y or x1,y1;x2,y2;...",
    )
    p.add_argument(
        "--point-labels",
        type=parse_point_labels,
        default=parse_point_labels("1"),
        help="One label per point: 1=foreground, 0=background (comma-separated)",
    )
    p.add_argument(
        "--box",
        type=parse_box,
        default=None,
        help="Box prompt x1,y1,x2,y2 (XYXY)",
    )
    p.add_argument(
        "--mask-input",
        type=parse_mask_input,
        default=None,
        metavar="PATH_OR_VALUES",
        help="Optional: path to .npy low-res mask logits (Bx1xHxW). "
        "SAM2 expects ~256x256 logits, not a pixel box.",
    )
    p.add_argument(
        "--multimask-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return three masks when ambiguous",
    )
    p.add_argument(
        "--return-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return unthresholded mask logits",
    )
    p.add_argument(
        "--normalize-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize point/box coords to [0,1] vs image size",
    )
    return p


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
def main():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    args = build_arg_parser().parse_args()

    point_coords = args.point_coords
    point_labels = args.point_labels
    if point_labels.shape[0] != point_coords.shape[0]:
        sys.exit(
            f"point_labels length {point_labels.shape[0]} != "
            f"number of points {point_coords.shape[0]}"
        )

    box = args.box
    mask_input = args.mask_input
    # If user passed a 1-d array from comma list, _prep_prompts may break;
    # mask_input from .npy is the intended path.

    image_path = args.image_path
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_points(point_coords, point_labels, plt.gca())
    # plt.axis("on")
    # plt.show()

    print("image shape:", image.shape)
    print("point_coords shape:", point_coords.shape)
    print("pointq_labels shape:", point_labels.shape)
    print("box shape:", "None" if box is None else box.shape)
    print("mask_input shape:", "None" if mask_input is None else mask_input.shape)
    print("multimask_output:", args.multimask_output)
    print("return_logits:", args.return_logits)
    print("normalize_coords:", args.normalize_coords)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=args.multimask_output,
            return_logits=args.return_logits,
            normalize_coords=args.normalize_coords,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
    print("masks shape:", masks.shape)
    print("scores shape:", scores.shape)
    print("logits shape:", logits.shape)
    # with multimask_output = True, there are 3 masks, where scores gives tyhe model's own esitmation of the quality of these masks
    show_masks(image, masks, scores, point_coords=point_coords, input_labels=point_labels, borders=True)

if __name__ == "__main__":
    main()
