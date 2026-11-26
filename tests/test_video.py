import numpy as np
import pandas as pd
import pytest

from saque.video.ball import BallParams, track_falling_ball
from saque.video.pose import effective_fps, to_uniform_grid
from saque.video.stickers import StickerParams, find_sticker
from tests import synth


def _image_with_disc(cx, cy, r, rgb=(190, 230, 30), size=(400, 300)):
    import cv2
    img = np.full((size[0], size[1], 3), (40, 90, 160), np.uint8)       # fondo azul
    cv2.circle(img, (cx, cy), r, rgb, -1)
    return img


def test_find_sticker_returns_centroid_near_landmark():
    img = _image_with_disc(150, 200, 8)
    x, y, area = find_sticker(img, 160, 190, StickerParams(search_radius_px=40, min_area_px=20))
    assert (x, y) == pytest.approx((150, 200), abs=1.0)
    assert area > 100


def test_find_sticker_ignores_far_and_unsaturated_blobs():
    p = StickerParams(search_radius_px=30, min_area_px=20)
    far = _image_with_disc(150, 200, 8)
    assert np.isnan(find_sticker(far, 50, 50, p)[0])
    pale = _image_with_disc(150, 200, 8, rgb=(200, 210, 170))             # poco saturado (playera)
    assert np.isnan(find_sticker(pale, 150, 200, p)[0])


def test_find_sticker_rejects_large_blob():
    big = _image_with_disc(150, 200, 40)                                 # pelota/playera: demasiado grande
    assert np.isnan(find_sticker(big, 150, 200, StickerParams(search_radius_px=60))[0])


def test_uniform_grid_inserts_lost_frames():
    df = synth.pose_frame(duration_s=1.0, lost_frames=range(10, 33))   # 23 fotogramas perdidos
    g = to_uniform_grid(df, 120.0)
    assert len(g) == 120
    assert g["frame"].isna().sum() == 23
    assert not g.loc[g["frame"].isna(), "detected"].any()
    assert np.allclose(np.diff(g["time_s"]), 1 / 120)


def test_effective_fps_ignores_gaps():
    pts = np.r_[0.0, 0.2 + np.arange(200) * 749 / 90000]                 # salto inicial + 120.16 fps
    assert effective_fps(pts) == pytest.approx(90000 / 749, rel=1e-6)


def test_ball_tracker_ends_at_contact_and_ignores_rising_toss():
    fs, p = 120.0, BallParams()
    dets = []
    for i in range(80):
        t = i / fs
        cands = []
        if i < 20:                                   # lanzamiento subiendo (se ignora)
            cands.append([300, 400 - 15 * i])
        if 30 <= i <= 60:                            # caída desde arriba hasta el contacto (i=60)
            k = i - 30
            cands.append([320 + 0.5 * k, 5 + 2 * k + 0.4 * k ** 2])
        if i > 60:                                   # tras el golpe sale disparada hacia un lado
            cands.append([320 + 60 * (i - 60), 300])
        cands.append([800, 900])                     # mancha fija del fondo
        dets.append((t, np.array(cands, float)))
    tr = track_falling_ball(dets, p)
    assert tr["t"].iloc[-1] == pytest.approx(60 / fs)
    assert tr["y"].iloc[0] < 50
