"""Test oracle only (not runtime): run Pranava's sahi_engine.py on the 4 test frames
and save a MATLAB-friendly reference (short field names) for the Block 1 parity test.
Usage: python make_py_reference.py <C3_detector_v1 folder> <out.mat>"""
import os, sys, numpy as np, scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sahi_engine as se
pkg, out = sys.argv[1], sys.argv[2]
det = se.C3Detector(os.path.join(pkg, 'best_matlab.onnx'))
ref = {}
for mode, bands in (('dual', se.BANDS), ('fast', se.BANDS_FAST)):
    lst = []
    for name in se.TEST_IMAGES:
        r = se.process_image(os.path.join(pkg, 'test_images', name), det, bands)
        lst.append({'name': name,
                    'ff_xywh': np.atleast_2d(r['ff_boxes']).reshape(-1, 4), 'ff_scores': r['ff_scores'].reshape(-1, 1),
                    'ff_cls': r['ff_cls'].reshape(-1, 1).astype(float),
                    'sahi_xywh': np.atleast_2d(r['sahi_boxes']).reshape(-1, 4), 'sahi_scores': r['sahi_scores'].reshape(-1, 1),
                    'sahi_cls': np.asarray(r['sahi_cls']).reshape(-1, 1).astype(float),
                    'is_new': r['is_new_sahi'].reshape(-1, 1).astype(float), 'n_tiles': r['n_tiles'],
                    't_ff_ms': r['t_fullframe_ms'], 't_sahi_ms': r['t_sahi_ms']})
        print(mode, name[:30], len(r['ff_scores']), len(r['sahi_scores']), int(r['is_new_sahi'].sum()))
    ref[mode] = np.array(lst, dtype=object)
sio.savemat(out, ref)
