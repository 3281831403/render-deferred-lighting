"""Behavioural tests for the deferred G-Buffer kernels.

Every test drives the public API in ``deferred.gbuffer``.  The initial stubs
raise ``NotImplementedError``, so the whole file is red until the eight
functions are implemented.  All tolerances are absolute ``1e-9`` unless a
validation tolerance is stated in the function docstring.
"""

import copy
import math

from deferred.gbuffer import (
    decode_normal,
    encode_depth,
    encode_normal,
    light_batch,
    reconstruct_pos,
    resolve_msaa,
    shade_gbuffer,
    validate_gbuffer,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# small deterministic helpers (no third-party maths library)
# ---------------------------------------------------------------------------

def _close(a, b, tol=TOL):
    return abs(a - b) <= tol


def _close_vec(a, b, tol=TOL):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _unit(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / length, v[1] / length, v[2] / length)


def _mat_vec(m, v):
    return tuple(
        m[r * 4 + 0] * v[0]
        + m[r * 4 + 1] * v[1]
        + m[r * 4 + 2] * v[2]
        + m[r * 4 + 3] * v[3]
        for r in range(4)
    )


def _invert4(m):
    """Gauss-Jordan inverse of a row-major 4x4 matrix, partial pivoting."""
    a = [
        list(m[r * 4:(r + 1) * 4]) + [1.0 if c == r else 0.0 for c in range(4)]
        for r in range(4)
    ]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-15:
            raise ValueError("singular matrix")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for r in range(4):
            if r == col:
                continue
            factor = a[r][col]
            if factor != 0.0:
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return tuple(a[r][4 + c] for r in range(4) for c in range(4))


def _projection_matrix(near, far):
    """DirectX-style perspective: NDC z is 0 at near and 1 at far.

    The camera looks along +z, so a view point is ``(x, y, d)`` with
    ``d`` the positive view depth.  Row-major, column-vector convention.
    """
    focal = 1.0  # 90 degree vertical field of view, square viewport
    return (
        focal, 0.0, 0.0, 0.0,
        0.0, focal, 0.0, 0.0,
        0.0, 0.0, far / (far - near), -far * near / (far - near),
        0.0, 0.0, 1.0, 0.0,
    )


def _depth01(d, near, far):
    """Closed-form reciprocal depth used to cross-check the kernels."""
    return far * (d - near) / ((far - near) * d)


def _gbuffer(normal=(0.0, 0.0, 1.0), position=(0.0, 0.0, 0.0),
             albedo=(0.5, 0.5, 0.5), ambient=(0.1, 0.1, 0.1)):
    """A valid one-pixel G-Buffer."""
    return {
        "width": 1,
        "height": 1,
        "position": [tuple(position)],
        "normal": [tuple(normal)],
        "albedo": [tuple(albedo)],
        "depth": [0.5],
        "opaque": [True],
        "ambient": tuple(ambient),
    }


def _light(position=(0.0, 0.0, 2.0), color=(1.0, 1.0, 1.0), intensity=1.0):
    return {
        "position": tuple(position),
        "color": tuple(color),
        "intensity": float(intensity),
    }


# ---------------------------------------------------------------------------
# one test per seeded defect
# ---------------------------------------------------------------------------

def test_encode_normal_lower_hemisphere_roundtrip():
    """Defect 1: folding only the upper hemisphere corrupts z < 0 normals."""
    for n in [
        (0.0, -0.6, -0.8),
        (0.6, 0.0, -0.8),
        (-0.48, 0.6, -0.64),
        (0.0, 0.0, -1.0),
    ]:
        encoded = encode_normal(n)
        assert 1 <= len(encoded) == 2
        decoded = decode_normal(encoded)
        assert _close_vec(decoded, n, TOL), (n, encoded, decoded)


def test_decode_normal_returns_unit_vector():
    """Defect 2: skipping the final normalisation loses unit length."""
    # (1, 4/7) is the octahedral image of (0.6, 0, -0.8); the unfolded raw
    # triple (3/7, 0, -4/7) has length 5/7 ~= 0.714, clearly not unit.
    n = decode_normal((1.0, 4.0 / 7.0))
    assert _close_vec(n, (0.6, 0.0, -0.8), TOL)
    assert _close(n[0] * n[0] + n[1] * n[1] + n[2] * n[2], 1.0, TOL)

    flat = decode_normal((0.5, 0.5))
    assert _close(flat[0] * flat[0] + flat[1] * flat[1] + flat[2] * flat[2], 1.0, TOL)


def test_encode_depth_is_reciprocal_not_linear():
    """Defect 3: linear depth storage destroys the 1/z distribution."""
    near, far = 0.1, 1.0
    d = 0.55
    encoded = encode_depth(d, near, far)
    assert _close(encoded, _depth01(d, near, far), TOL)
    # A linear encoding would be 0.5 here; reciprocal encoding is 10/11.
    assert encoded > 0.85


def test_reconstruct_pos_applies_perspective_w_division():
    """Defect 4: omitting clip.w division makes the error grow with depth."""
    near, far = 0.1, 1.0
    inv_proj = _invert4(_projection_matrix(near, far))
    for x, y, d in [(0.3, -0.2, 0.4), (0.15, 0.1, 0.9)]:
        ndc_x = x / d
        ndc_y = y / d
        uv = (0.5 + 0.5 * ndc_x, 0.5 - 0.5 * ndc_y)
        depth = _depth01(d, near, far)
        position = reconstruct_pos(uv, depth, inv_proj)
        assert _close_vec(position, (x, y, d), TOL), (x, y, d, position)


def test_ambient_is_added_once_for_multiple_lights():
    """Defect 5: ambient must not be multiplied by the light count."""
    gbuffer = _gbuffer(albedo=(0.5, 0.5, 0.5), ambient=(0.2, 0.2, 0.2))
    lights = [
        _light((0.0, 0.0, 2.0), intensity=1.0),
        _light((0.0, 0.0, 4.0), intensity=2.0),
    ]
    shaded = shade_gbuffer(gbuffer, lights)
    expected = 0.5 * (0.2 + 1.0 / 4.0 + 2.0 / 16.0)
    assert _close_vec(shaded[0], (expected, expected, expected), TOL)


def test_light_attenuation_uses_inverse_square():
    """Defect 6: 1/d falloff is far too bright at a distance."""
    gbuffer = _gbuffer(albedo=(1.0, 1.0, 1.0), ambient=(0.0, 0.0, 0.0))
    near_value = shade_gbuffer(gbuffer, [_light((0.0, 0.0, 2.0))])[0][0]
    far_value = shade_gbuffer(gbuffer, [_light((0.0, 0.0, 8.0))])[0][0]
    assert _close(near_value, 0.25, TOL)
    assert _close(far_value, 1.0 / 64.0, TOL)
    assert _close(near_value / far_value, 16.0, TOL)


def test_resolve_msaa_averages_color_and_takes_nearest_depth():
    """Defect 7: averaged depth produces false silhouettes at edges."""
    pixels = [
        [
            {"color": (1.0, 0.0, 0.0), "depth": 0.8},
            {"color": (0.0, 0.0, 1.0), "depth": 0.2},
        ],
        [
            {"color": (0.2, 0.4, 0.6), "depth": 0.9},
            {"color": (0.4, 0.4, 0.4), "depth": 0.5},
            {"color": (0.0, 0.2, 0.2), "depth": 0.7},
        ],
    ]
    resolved = resolve_msaa(pixels)
    assert _close_vec(resolved["color"][0], (0.5, 0.0, 0.5), TOL)
    assert _close(resolved["depth"][0], 0.2, TOL)
    assert _close_vec(resolved["color"][1], (0.2, 1.0 / 3.0, 0.4), TOL)
    assert _close(resolved["depth"][1], 0.5, TOL)


def test_validate_gbuffer_rejects_non_unit_normal():
    """Defect 8: a non-unit normal must be reported, not silently accepted."""
    gbuffer = _gbuffer()
    assert validate_gbuffer(gbuffer) == []

    gbuffer["normal"] = [(2.0, 0.0, 0.0)]
    errors = validate_gbuffer(gbuffer)
    assert errors
    assert any("normal" in message.lower() for message in errors)


def test_light_batch_uses_each_frames_own_normals():
    """Defect 9: batch shading must not reuse the first frame's G-Buffer."""
    lights = [_light((0.0, 0.0, 2.0))]
    frames = [
        _gbuffer(normal=(0.0, 0.0, 1.0)),
        _gbuffer(normal=(0.0, 0.0, -1.0)),
    ]
    batch = light_batch(frames, {"lights": lights})
    first = shade_gbuffer(frames[0], lights)
    second = shade_gbuffer(frames[1], lights)
    assert _close_vec(batch["frames"][1][0], second[0], TOL)
    assert not _close_vec(batch["frames"][1][0], first[0], TOL)


def test_shade_gbuffer_does_not_write_depth_on_translucent_pixels():
    """Defect 10: no depth write-through for translucent (opaque=False) pixels."""
    gbuffer = {
        "width": 2,
        "height": 1,
        "position": [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)],
        "normal": [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        "albedo": [(0.5, 0.5, 0.5), (0.2, 0.2, 0.2)],
        "depth": [0.4, 0.8],
        "opaque": [True, False],
        "ambient": (0.1, 0.1, 0.1),
    }
    snapshot = copy.deepcopy(gbuffer)
    shaded = shade_gbuffer(gbuffer, [_light((0.0, 0.0, 2.0))])
    assert len(shaded) == 2
    assert gbuffer == snapshot


# ---------------------------------------------------------------------------
# supporting contract tests
# ---------------------------------------------------------------------------

def test_normal_roundtrip_arbitrary_unit_normals():
    """Octahedral encode/decode is an exact inverse for every hemisphere."""
    normals = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        _unit((1.0, 2.0, 3.0)),
        _unit((-0.3, -0.5, -0.8)),
        _unit((0.2, -0.9, 0.1)),
    ]
    for n in normals:
        decoded = decode_normal(encode_normal(n))
        assert _close_vec(decoded, n, TOL), (n, decoded)


def test_encode_depth_endpoints_are_exact():
    """The non-linear mapping pins the near and far planes exactly."""
    near, far = 0.1, 1.0
    assert _close(encode_depth(near, near, far), 0.0, TOL)
    assert _close(encode_depth(far, near, far), 1.0, TOL)


def test_single_light_matches_analytic_solution():
    """One light, one surface: compare against the closed-form energy."""
    albedo = (0.2, 0.4, 0.6)
    ambient = (0.05, 0.1, 0.15)
    gbuffer = _gbuffer(
        normal=(0.0, 0.0, 1.0),
        position=(0.0, 0.0, 0.0),
        albedo=albedo,
        ambient=ambient,
    )
    light = _light((0.0, 0.0, 2.0), color=(0.9, 0.8, 0.7), intensity=1.5)
    shaded = shade_gbuffer(gbuffer, [light])
    attenuation = 1.5 / 4.0
    expected = tuple(
        albedo[c] * (ambient[c] + attenuation * light["color"][c])
        for c in range(3)
    )
    assert _close_vec(shaded[0], expected, TOL)


def test_ambient_independent_of_light_count():
    """Ambient is a scene constant: 0, 1 or 3 lights give the same term."""
    gbuffer = _gbuffer(albedo=(0.25, 0.5, 0.75), ambient=(0.2, 0.2, 0.2))
    zero_light = _light((0.0, 0.0, 2.0), intensity=0.0)
    results = [
        shade_gbuffer(gbuffer, [dict(zero_light) for _ in range(count)])
        for count in (0, 1, 3)
    ]
    expected = tuple(0.2 * channel for channel in (0.25, 0.5, 0.75))
    for shaded in results:
        assert _close_vec(shaded[0], expected, TOL)


def test_light_batch_frames_match_individual_shading():
    """Batched frames equal separate shading and leave the inputs untouched."""
    lights = [_light((0.0, 0.0, 2.0), intensity=2.0)]
    frames = [
        _gbuffer(normal=(0.0, 0.0, 1.0), albedo=(0.6, 0.3, 0.2)),
        _gbuffer(normal=_unit((0.0, 0.6, 0.8)), albedo=(0.1, 0.9, 0.4)),
    ]
    snapshot = copy.deepcopy(frames)
    batch = light_batch(frames, {"lights": lights})
    assert batch["frame_count"] == 2
    for index in range(2):
        expected = shade_gbuffer(frames[index], lights)
        assert _close_vec(batch["frames"][index][0], expected[0], TOL)
    assert frames == snapshot
