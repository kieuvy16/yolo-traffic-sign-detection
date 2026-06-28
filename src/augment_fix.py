"""
Quick augmentation for remaining classes below 300 bbox.
"""
import os, sys, traceback
sys.stdout.reconfigure(line_buffering=True)

try:
    import cv2
    import numpy as np
    from collections import defaultdict

    IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database_goc", "images")
    LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database_goc", "labels")
    TARGET = 300

    print(f"IMAGES_DIR: {IMAGES_DIR}", flush=True)
    print(f"LABELS_DIR: {LABELS_DIR}", flush=True)

    def hflip(img, bb):
        return cv2.flip(img, 1), [(c, 1.0-cx, cy, w, h) for c,cx,cy,w,h in bb]

    def bright(img, bb, f):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*f, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), bb

    def rot(img, bb, a):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), a, 1.0)
        ca, sa = abs(M[0,0]), abs(M[0,1])
        nw, nh = int(h*sa+w*ca), int(h*ca+w*sa)
        M[0,2] += (nw-w)/2
        M[1,2] += (nh-h)/2
        ri = cv2.warpAffine(img, M, (nw,nh), borderValue=(114,114,114))
        nb = []
        for c,cx,cy,bw,bh in bb:
            pts = np.array([
                [(cx-bw/2)*w, (cy-bh/2)*h],
                [(cx+bw/2)*w, (cy-bh/2)*h],
                [(cx+bw/2)*w, (cy+bh/2)*h],
                [(cx-bw/2)*w, (cy+bh/2)*h]
            ], dtype=np.float32)
            rc = M.dot(np.hstack([pts, np.ones((4,1), dtype=np.float32)]).T).T
            x1 = max(0, rc[:,0].min())
            y1 = max(0, rc[:,1].min())
            x2 = min(nw, rc[:,0].max())
            y2 = min(nh, rc[:,1].max())
            ncx = ((x1+x2)/2)/nw
            ncy = ((y1+y2)/2)/nh
            nbw = (x2-x1)/nw
            nbh = (y2-y1)/nh
            if nbw > 0.01 and nbh > 0.01 and nbw < 1 and nbh < 1:
                nb.append((c, ncx, ncy, nbw, nbh))
        return ri, nb

    AUGS = [
        ('fl', lambda i,b: hflip(i,b)),
        ('bu', lambda i,b: bright(i,b,1.4)),
        ('bd', lambda i,b: bright(i,b,0.6)),
        ('rp', lambda i,b: rot(i,b,10)),
        ('rn', lambda i,b: rot(i,b,-10)),
        ('r2', lambda i,b: rot(i,b,15)),
        ('r3', lambda i,b: rot(i,b,-15)),
        ('fb', lambda i,b: bright(*hflip(i,b), 1.3)),
    ]

    # Count bboxes per class
    print("Counting labels...", flush=True)
    lfs = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    print(f"Total label files: {len(lfs)}", flush=True)

    bc = defaultdict(int)
    ipc = defaultdict(list)
    for lf in lfs:
        seen = set()
        with open(os.path.join(LABELS_DIR, lf)) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    c = int(p[0])
                    bc[c] += 1
                    seen.add(c)
        for c in seen:
            ipc[c].append(lf)

    print("Done counting.", flush=True)

    total = 0
    for cid in range(52):
        cnt = bc.get(cid, 0)
        if cnt >= TARGET:
            continue
        need = TARGET - cnt
        srcs = ipc.get(cid, [])
        if not srcs:
            print(f"SKIP class {cid}: no sources", flush=True)
            continue

        print(f"Augmenting class {cid}: {cnt} -> {TARGET} (need {need}, {len(srcs)} sources)", flush=True)
        created = 0
        ai = 0
        while created < need:
            for lf in srcs:
                if created >= need:
                    break
                base = os.path.splitext(lf)[0]
                ip = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    p = os.path.join(IMAGES_DIR, base + ext)
                    if os.path.exists(p):
                        ip = p
                        break
                if not ip:
                    continue
                img = cv2.imread(ip)
                if img is None:
                    continue
                bb = []
                with open(os.path.join(LABELS_DIR, lf)) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) >= 5:
                            bb.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                an, af = AUGS[ai % len(AUGS)]
                ai += 1
                try:
                    aug_img, aug_bb = af(img, bb)
                except Exception as e:
                    continue
                if not aug_bb:
                    continue
                ie = os.path.splitext(ip)[1]
                nn = f"a{cid}_{an}_{created}"
                cv2.imwrite(os.path.join(IMAGES_DIR, nn + ie), aug_img)
                with open(os.path.join(LABELS_DIR, nn + '.txt'), 'w') as f:
                    for b in aug_bb:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                created += 1
            if ai > need * 3:
                break
        print(f"  -> Created {created} augmented images", flush=True)
        total += created

    print(f"\nTotal new augmented: {total}", flush=True)

    # Final count
    imgs = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
    lbls = len([f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')])
    print(f"Final: {imgs} images, {lbls} labels", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
