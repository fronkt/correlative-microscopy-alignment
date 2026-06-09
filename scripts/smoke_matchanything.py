"""MatchAnything smoke test via the transformers Hub model."""

import time

import numpy as np
from PIL import Image


def main() -> None:
    print("importing transformers ...")
    import transformers
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    print(f"transformers version: {transformers.__version__}")

    print("\nloading zju-community/matchanything_eloftr ...")
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr")
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr")
    print(f"loaded processor + model in {time.perf_counter() - t0:.1f}s")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"model on {device}")

    # Build a tiny synthetic pair (avoid network)
    rng = np.random.default_rng(0)
    a = (rng.random((480, 640, 3)) * 255).astype("uint8")
    b = (rng.random((480, 640, 3)) * 255).astype("uint8")
    img_a = Image.fromarray(a)
    img_b = Image.fromarray(b)

    print("\nrunning processor + forward on a noise pair ...")
    inputs = proc([img_a, img_b], return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(**inputs)
    print(f"forward in {time.perf_counter() - t0:.2f}s")
    print(f"output keys: {list(out.keys())}")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")

    if hasattr(proc, "post_process_keypoint_matching"):
        # processor expects (batch, 2, 2) — one pair of (H, W)s per batch entry
        sizes = torch.tensor(
            [[[img_a.height, img_a.width], [img_b.height, img_b.width]]],
            device=device,
        )
        print("\npost_process_keypoint_matching ...")
        post = proc.post_process_keypoint_matching(out, sizes, threshold=0.2)
        print(f"post len: {len(post)}")
        if len(post):
            for k, v in post[0].items():
                if hasattr(v, "shape"):
                    print(f"  [0] {k}: shape={tuple(v.shape)}")
                else:
                    print(f"  [0] {k}: {v}")


if __name__ == "__main__":
    main()
