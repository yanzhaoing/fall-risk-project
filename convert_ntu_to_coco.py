#!/usr/bin/env python3
# convert_ntu_to_coco.py
# NTU RGB+D skeleton -> COCO 17 keypoints
# Usage: python convert_ntu_to_coco.py --input_dir PATH --output_dir PATH --normalize

import os
import json
import argparse
import numpy as np
from pathlib import Path

NTU_TO_COCO = {
    0: 3, 1: 3, 2: 3, 3: 3, 4: 3,
    5: 4, 6: 8, 7: 5, 8: 9,
    9: 6, 10: 10, 11: 12, 12: 16,
    13: 13, 14: 17, 15: 14, 16: 18,
}

FALL_ACTIONS = {42, 43}
ADL_ACTIONS = {1,2,3,4,5,6,7,8,9,10,28,29,30,35,36,37}


def parse_skeleton(filepath):
    """Parse .skeleton file. Format per frame:
    num_bodies
    bodyID clippedEdges handLConf handLState handRConf handRState isResting leanX leanY trackingState
    numJoints
    x y z depthX depthY colorX colorY oriW oriX oriY oriZ trackingState  (x numJoints)
    """
    frames = []
    with open(filepath, 'r') as f:
        num_frames = int(f.readline().strip())
        for _ in range(num_frames):
            num_bodies = int(f.readline().strip())
            bodies = []
            for _ in range(num_bodies):
                # bodyID + properties on ONE line
                body_line = f.readline().strip().split()
                body_id = int(body_line[0])
                num_joints = int(f.readline().strip())
                joints = []
                for _ in range(num_joints):
                    vals = list(map(float, f.readline().strip().split()))
                    joints.append({
                        'x': vals[0], 'y': vals[1], 'z': vals[2],
                        'tracking': int(vals[11])
                    })
                bodies.append({'id': body_id, 'joints': joints})
            frames.append(bodies)
    return frames


def ntu_to_coco(joints):
    coco = []
    for coco_idx in range(17):
        ntu_idx = NTU_TO_COCO[coco_idx]
        j = joints[ntu_idx]
        conf = {2: 1.0, 1: 0.5, 0: 0.0}.get(j['tracking'], 0.0)
        coco.append([j['x'], j['y'], conf])
    return coco


def normalize_sequence(seq):
    arr = np.array(seq)
    xy = arr[:, :, :2]
    mn, mx = xy.min(), xy.max()
    if mx - mn > 0:
        arr[:, :, :2] = (xy - mn) / (mx - mn)
    return arr.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--actions", default="all")
    parser.add_argument("--max_per_class", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.actions == "all":
        targets = FALL_ACTIONS | ADL_ACTIONS
    elif args.actions == "fall_only":
        targets = FALL_ACTIONS
    else:
        targets = set(int(x) for x in args.actions.split(","))

    counts = {0: 0, 1: 0}
    files = sorted(input_dir.glob("*.skeleton"))
    print(f"Found {len(files)} .skeleton files")

    processed = 0
    skipped = 0
    for skel in files:
        name = skel.stem
        try:
            a_pos = name.index('A')
            action_id = int(name[a_pos+1:])
            p_pos = name.index('P')
            performer_id = int(name[p_pos+1:name.index('R')])
        except (ValueError, IndexError):
            skipped += 1
            continue

        if action_id not in targets:
            continue

        label = 1 if action_id in FALL_ACTIONS else 0
        if args.max_per_class > 0 and counts[label] >= args.max_per_class:
            continue

        try:
            frames = parse_skeleton(str(skel))
        except Exception as e:
            skipped += 1
            continue

        if not frames:
            skipped += 1
            continue

        kpts_seq = []
        for frame in frames:
            if not frame:
                continue
            best = max(frame, key=lambda b: sum(1 for j in b['joints'] if j['tracking'] == 2))
            kpts_seq.append(ntu_to_coco(best['joints']))

        if not kpts_seq:
            skipped += 1
            continue

        if args.normalize:
            kpts_seq = normalize_sequence(kpts_seq)

        counts[label] += 1
        activity = f"{'Fall' if label else 'ADL'}_{counts[label]:03d}"
        out_dir = output_dir / f"Subject_{performer_id:02d}" / activity
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "keypoints.json", 'w') as f:
            json.dump({
                "keypoints": kpts_seq,
                "label": label,
                "source": "NTU_RGBD",
                "action_id": action_id,
                "frames": len(kpts_seq),
            }, f)

        processed += 1
        if processed % 200 == 0:
            print(f"  processed {processed} ...")

    print(f"\nDone! {processed} converted, {skipped} skipped")
    print(f"  Fall: {counts[1]}")
    print(f"  ADL:  {counts[0]}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
