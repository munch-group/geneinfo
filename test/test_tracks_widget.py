"""Tests for the Tracks widget's data transport.

The Python side (revisions, per-chromosome fetch replies, binary packing,
embedded mode) is tested directly. The browser side is tested by running the
widget's own ES module in Node (``test/js/tracks_harness.mjs``) against a stub
DOM and WebGL, fed with payloads produced here; skipped when ``node`` is not
on the PATH.

Run with ``python -m unittest test.test_tracks_widget``.
"""
import base64
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from geneinfo.widget import Tracks
from geneinfo.widget.segment_viewer_gl import _chrom_slice, _split_buffers

CHROMS = {'chr1': 1_000_000, 'chr2': 500_000}
HARNESS = Path(__file__).parent / 'js' / 'tracks_harness.mjs'


def _frames():
    rng = np.random.default_rng(2)
    n = 300
    seg = pd.DataFrame({'chrom': rng.choice(['chr1', 'chr2'], n),
                        'start': rng.integers(0, 400_000, n)})
    seg['end'] = seg.start + 5000
    seg['ind'] = rng.choice(list('abcdefgh'), n)
    seg['v'] = rng.random(n)
    pts = pd.DataFrame({'chrom': rng.choice(['chr1', 'chr2'], 200),
                        'pos': rng.integers(0, 450_000, 200),
                        'y': rng.random(200),
                        'g': rng.choice(['x', 'y'], 200)}).sort_values('pos')
    arcs = pd.DataFrame({'chrom': ['chr1', 'chr1', 'chr2'],
                         'pos1': [1000, 5000, 2000],
                         'pos2': [90000, 400000, 300000]})
    return seg, pts, arcs


def _add_first_two(t, seg, pts):
    return (t.add_point_track(pts, 'P', x='pos', y='y', group_by='g')
             .add_heatmap_track(seg, 'H', individual_col='ind', value_col='v',
                                palette='viridis'))


def _add_rest(t, seg, pts, arcs):
    return (t.add_segment_track(seg, 'S', individual_col='ind')
             .add_fill_track(pts, 'F', x='pos', y='y', step='post')
             .add_histogram_track(pts, 'Hi', x='pos', y='y')
             .add_arc_track(arcs, 'A')
             .add_line_track(pts, 'L', x='pos', y='y'))


_SYNCED = ['chrom_sizes', 'track_configs', 'track_revs', 'viewport', 'vlines',
           'spans', 'zoom_speed', 'pan_speed', 'min_span', 'theme']


def _state(t):
    return json.loads(json.dumps({k: getattr(t, k) for k in _SYNCED}))


def _replies(t):
    out = {}
    for tid in t.track_revs:
        out[tid] = {}
        for ch in CHROMS:
            resp, bufs = t._cmd_fetch_track({'tid': tid, 'chrom': ch}, [])
            out[tid][ch] = {
                'response': resp,
                'buffers': [base64.b64encode(b).decode() for b in bufs],
            }
    return out


class TestTransportPython(unittest.TestCase):

    def setUp(self):
        self.seg, self.pts, self.arcs = _frames()
        self.t = _add_rest(_add_first_two(Tracks(CHROMS), self.seg, self.pts),
                           self.seg, self.pts, self.arcs)

    def test_track_data_not_synced(self):
        self.assertNotIn('track_data', self.t.keys)
        self.assertEqual(self.t.embedded_data, {})
        self.assertIn('track_revs', self.t.keys)

    def test_fetch_returns_one_chromosome_as_binary(self):
        for tid, cfg in ((c['id'], c) for c in self.t.track_configs):
            resp, bufs = self.t._cmd_fetch_track({'tid': tid, 'chrom': 'chr2'}, [])
            self.assertNotIn('error', resp)
            self.assertEqual(resp['rev'], self.t.track_revs[tid])
            json.dumps(resp)  # the reply must be plain JSON
            self.assertEqual(len(resp['paths']), len(bufs))
            self.assertTrue(all(isinstance(b, bytes) and b for b in bufs))
            payload = resp['payload']
            chroms = (set(payload['segs']) | set(payload['dens'])
                      if cfg['type'] == 'segment' else set(payload))
            self.assertLessEqual(chroms, {'chr2'}, cfg['type'])

    def test_fetch_unknown_chrom_is_empty(self):
        resp, bufs = self.t._cmd_fetch_track({'tid': 't0', 'chrom': 'chrZ'}, [])
        self.assertEqual((resp['payload'], bufs), ({}, []))

    def test_fetch_unknown_track_reports_error(self):
        resp, bufs = self.t._cmd_fetch_track({'tid': 't99', 'chrom': 'chr1'}, [])
        self.assertIn('error', resp)
        self.assertEqual(bufs, [])

    def test_fetch_failure_is_reported_not_raised(self):
        self.t.track_data['t0'] = object()  # not a payload
        resp, _ = self.t._cmd_fetch_track({'tid': 't0', 'chrom': 'chr1'}, [])
        self.assertIn('error', resp)

    def test_revisions(self):
        t = self.t
        revs = dict(t.track_revs)
        self.assertEqual(len(set(revs.values())), len(revs))
        hid = next(c['id'] for c in t.track_configs if c['type'] == 'heatmap')
        t._rebin_heatmap_for_view(hid, 'chr1', 100_000, 200_000)
        self.assertGreater(t.track_revs[hid], revs[hid])
        self.assertEqual({k: v for k, v in t.track_revs.items() if k != hid},
                         {k: v for k, v in revs.items() if k != hid})
        before = dict(t.track_revs)
        t.hide_track('P')
        self.assertEqual(t.track_revs, before)  # config-only change
        t.remove_track('P')
        self.assertNotIn('t0', t.track_revs)
        t.clear_tracks()
        self.assertEqual((t.track_revs, t.track_data), ({}, {}))

    def test_split_buffers(self):
        payload = {'a': {'x': b'\x01\x02', 'e': b''}, 'l': [b'\x03', 'txt', 4]}
        out, paths, bufs = _split_buffers(payload)
        self.assertEqual(out, {'a': {'x': None, 'e': ''}, 'l': [None, 'txt', 4]})
        self.assertEqual(paths, [['a', 'x'], ['l', 0]])
        self.assertEqual(bufs, [b'\x01\x02', b'\x03'])
        self.assertEqual(payload['a']['x'], b'\x01\x02')  # input untouched

    def test_chrom_slice(self):
        seg = {'segs': {'chr1': 1, 'chr2': 2}, 'dens': {'chr1': 3}}
        self.assertEqual(_chrom_slice('segment', seg, 'chr2'),
                         {'segs': {'chr2': 2}, 'dens': {}})
        self.assertEqual(_chrom_slice('heatmap', {'chr1': 1}, 'chr1'), {'chr1': 1})
        self.assertEqual(_chrom_slice('heatmap', {'chr1': 1}, 'chr2'), {})

    def test_heatmap_lut_stays_text(self):
        cfg = next(c for c in self.t.track_configs if c['type'] == 'heatmap')
        self.assertIsInstance(cfg['lut'], str)

    def test_embed_data(self):
        from ipywidgets.widgets.widget import _remove_buffers
        e = _add_first_two(Tracks(CHROMS, embed_data=True), self.seg, self.pts)
        self.assertEqual(e.embedded_data, e.track_data)
        state, _, bufs = _remove_buffers(e.get_state())
        self.assertTrue(bufs)  # packed arrays travel as binary buffers
        json.dumps(state)


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class TestTransportBrowser(unittest.TestCase):

    def test_harness(self):
        from ipywidgets.widgets.widget import _remove_buffers
        seg, pts, arcs = _frames()
        t = _add_first_two(Tracks(CHROMS), seg, pts)
        two = _state(t)
        _add_rest(t, seg, pts, arcs)
        e = _add_first_two(Tracks(CHROMS, embed_data=True), seg, pts)
        emb_state, emb_paths, emb_bufs = _remove_buffers(
            {'embedded_data': e.embedded_data})
        cases = {
            'empty': {'state': _state(Tracks(CHROMS)), 'replies': {}},
            'two': {'state': two, 'replies': _replies(t)},
            'seven': {'state': _state(t), 'replies': _replies(t)},
            'embedded': {
                'state': {**_state(e), **emb_state},
                'replies': {},
                'embedded_paths': [p[1:] for p in emb_paths],
                'embedded_buffers': [base64.b64encode(bytes(b)).decode()
                                     for b in emb_bufs],
            },
        }
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'widget.mjs').write_text(Tracks._esm)
            Path(d, 'cases.json').write_text(json.dumps(cases))
            run = subprocess.run(['node', str(HARNESS), d],
                                 capture_output=True, text=True, timeout=120)
        failed = [l for l in run.stdout.splitlines() if l.startswith('FAIL')]
        self.assertEqual(run.returncode, 0,
                         '\n'.join(failed) or run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
