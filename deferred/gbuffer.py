"""G-Buffer encoding / reconstruction kernels for the deferred lighting exercise.

Conventions used throughout this module
---------------------------------------
* Depth is normalised to ``[0, 1]``: ``0`` is the near plane and ``1`` is the
  far plane.  Perspective depth is non-linear in view distance (it goes with
  ``1/z``).
* Normals are unit vectors: ``x*x + y*y + z*z == 1``.
* Numeric comparisons in the test-suite use an absolute tolerance of ``1e-9``.
* Matrices are sequences of 16 floats in row-major order and operate on
  column vectors: ``clip = M @ (x, y, z, 1)``.
"""

import math


def _sign(value):
    """Return ``+1`` for non-negative values (including zero) and ``-1`` otherwise."""
    return 1.0 if value >= 0.0 else -1.0


def encode_normal(n):
    """Encode a unit normal into octahedral coordinates in ``[-1, 1]^2``.

    Parameters
    ----------
    n : tuple[float, float, float]
        Unit normal ``(x, y, z)``.  Callers guarantee ``|n| == 1``.

    Returns
    -------
    tuple[float, float]
        ``(ex, ey)`` with both components inside ``[-1, 1]``.

    Contract
    --------
    * The direct projection ``(x / l1, y / l1)`` (with
      ``l1 = |x| + |y| + |z|``) is only valid for the ``z >= 0`` hemisphere.
    * For ``z < 0`` the point must be folded onto the outer diamond using
      ``(1 - |py|) * sign(px)`` and ``(1 - |px|) * sign(py)``, where ``sign``
      returns ``+1`` for zero so the pole stays well defined.
    * ``decode_normal(encode_normal(n))`` must return ``n`` within ``1e-9``
      for every unit normal, upper *and* lower hemisphere.
    """
    x, y, z = n
    l1 = abs(x) + abs(y) + abs(z)
    if l1 == 0.0:
        # Degenerate input; the +Z pole is the only sensible encoding.
        return (0.0, 0.0)

    px = x / l1
    py = y / l1

    if z < 0.0:
        # Fold the lower hemisphere onto the outer diamond.  Both components
        # must be computed from the *unfolded* px/py, hence the temporaries.
        folded_x = (1.0 - abs(py)) * _sign(px)
        folded_y = (1.0 - abs(px)) * _sign(py)
        px, py = folded_x, folded_y

    return (px, py)


def decode_normal(e):
    """Decode an octahedral coordinate back into a normalised 3D normal.

    Parameters
    ----------
    e : tuple[float, float]
        ``(ex, ey)`` inside ``[-1, 1]^2``, as produced by :func:`encode_normal`.

    Returns
    -------
    tuple[float, float, float]
        A unit normal.  Its length must be ``1`` within ``1e-9``; returning the
        raw unfolded triple is not acceptable.

    Contract
    --------
    * Start from ``(ex, ey, 1 - |ex| - |ey|)``.  When that ``z`` is negative,
      unwrap ``x`` and ``y`` with ``(1 - |y|) * sign(x)`` and
      ``(1 - |x|) * sign(y)`` (keeping the negative ``z``) and then normalise.
    * The mapping is continuous across the diamond border
      ``|ex| + |ey| == 1``, where ``z`` reaches zero.
    * It is the exact inverse of :func:`encode_normal` within ``1e-9``.
    """
    px, py = e
    z = 1.0 - abs(px) - abs(py)

    if z < 0.0:
        unwrapped_x = (1.0 - abs(py)) * _sign(px)
        unwrapped_y = (1.0 - abs(px)) * _sign(py)
        px, py = unwrapped_x, unwrapped_y

    length = math.sqrt(px * px + py * py + z * z)
    if length == 0.0:
        # Only reachable for a degenerate coordinate; fall back to the +Z pole.
        return (0.0, 0.0, 1.0)

    return (px / length, py / length, z / length)


def encode_depth(d, near, far):
    """Encode a normalised linear depth with the non-linear ``1/z`` curve.

    Parameters
    ----------
    d : float
        Linear view depth on the same normalised ``[0, 1]`` scale as ``near``
        and ``far``; ``near <= d <= far``.
    near : float
        Normalised near plane, ``0 < near < far``.
    far : float
        Normalised far plane, ``near < far <= 1``.

    Returns
    -------
    float
        Stored depth in ``[0, 1]``::

            far * (d - near) / ((far - near) * d)

        which is ``0`` exactly at ``near`` and ``1`` exactly at ``far``.

    Contract
    --------
    This is a reciprocal (``1/z``) mapping, not ``(d - near) / (far - near)``.
    The linear form loses the near-field distribution and makes reconstructed
    positions jitter at distance.
    """
    return far * (d - near) / ((far - near) * d)


def reconstruct_pos(uv, depth, inv_view_proj):
    """Reconstruct a 3D position from screen UV, stored depth and an inverse matrix.

    Parameters
    ----------
    uv : tuple[float, float]
        Screen UV in ``[0, 1]^2``: ``u`` grows to the right and ``v`` grows
        downwards.
    depth : float
        Stored depth in ``[0, 1]`` as produced by :func:`encode_depth`.
        The projection matrix follows the DirectX convention, so this value is
        already the clip-space NDC z (``0`` at the near plane, ``1`` at far).
    inv_view_proj : sequence[float]
        16 floats, row-major inverse of ``projection * view`` for column
        vectors (``clip = M @ (x, y, z, 1)``).

    Returns
    -------
    tuple[float, float, float]
        ``(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)`` where::

            ndc  = (2 * u - 1, 1 - 2 * v, depth, 1)
            clip = inv_view_proj @ ndc

    Contract
    --------
    The perspective divide by ``w`` is mandatory.  Returning ``clip.xyz``
    directly makes the positional error grow linearly with depth.
    """
    u, v = uv
    ndc = (2.0 * u - 1.0, 1.0 - 2.0 * v, depth, 1.0)

    m = inv_view_proj
    clip = tuple(
        m[r * 4 + 0] * ndc[0]
        + m[r * 4 + 1] * ndc[1]
        + m[r * 4 + 2] * ndc[2]
        + m[r * 4 + 3] * ndc[3]
        for r in range(4)
    )

    w = clip[3]
    return (clip[0] / w, clip[1] / w, clip[2] / w)


def shade_gbuffer(gbuffer, lights):
    """Shade every G-Buffer pixel with a list of point lights.

    Parameters
    ----------
    gbuffer : dict
        Parallel per-pixel lists plus one scene constant:

        ``position``
            ``(x, y, z)`` surface position per pixel.
        ``normal``
            Unit ``(x, y, z)`` per pixel.
        ``albedo``
            ``(r, g, b)`` in ``[0, 1]`` per pixel.
        ``depth``
            ``0..1`` stored depth per pixel.
        ``opaque``
            ``bool`` per pixel; ``False`` marks a translucent pixel.
        ``ambient``
            Scene ambient ``(r, g, b)`` radiance (a single tuple).
        ``width`` / ``height``
            Optional pixel-grid dimensions.

    lights : list[dict]
        Each light has ``position`` ``(x, y, z)``, ``color`` ``(r, g, b)`` and
        ``intensity`` (float).  A light must not sit exactly on a shaded
        surface (``|L| > 0``).

    Returns
    -------
    list[tuple[float, float, float]]
        One shaded ``(r, g, b)`` tuple per pixel, in pixel order.

    Contract
    --------
    For pixel ``i`` and ``L = light.position - position``::

        color[c] = ambient[c] * albedo[c]
                 + sum_lights albedo[c] * light.color[c] * light.intensity
                   * max(0, dot(normal, L / |L|)) / |L|^2

    * Ambient must be added exactly once, independent of the light count.
    * Falloff is inverse-square (``1 / d^2``); ``1 / d`` is too bright far away.
    * There is no specular term, so no highlight factor is applied twice.
    * The function is pure: it must not mutate ``gbuffer``.  In particular it
      must never write ``depth`` for translucent (``opaque is False``) pixels;
      translucency is composited later and must not overwrite opaque depth.
    """
    positions = gbuffer["position"]
    normals = gbuffer["normal"]
    albedos = gbuffer["albedo"]
    ambient = gbuffer["ambient"]

    shaded = []
    for index in range(len(positions)):
        position = positions[index]
        normal = normals[index]
        albedo = albedos[index]

        # Ambient is a scene constant: add it once, outside the light loop.
        color = [ambient[c] * albedo[c] for c in range(3)]

        for light in lights:
            light_position = light["position"]
            lx = light_position[0] - position[0]
            ly = light_position[1] - position[1]
            lz = light_position[2] - position[2]

            dist2 = lx * lx + ly * ly + lz * lz
            if dist2 == 0.0:
                continue
            inv_dist = 1.0 / math.sqrt(dist2)

            n_dot_l = (
                normal[0] * lx + normal[1] * ly + normal[2] * lz
            ) * inv_dist
            if n_dot_l <= 0.0:
                continue

            # Inverse-square falloff: intensity * N.L / |L|^2.
            attenuation = light["intensity"] * n_dot_l / dist2
            light_color = light["color"]
            for c in range(3):
                color[c] += albedo[c] * light_color[c] * attenuation

        shaded.append((color[0], color[1], color[2]))

    return shaded


def resolve_msaa(samples):
    """Resolve per-pixel multi-sample antialiasing coverage.

    Parameters
    ----------
    samples : list[list[dict]]
        One entry per pixel; a pixel is a non-empty list of sample dicts::

            {'color': (r, g, b), 'depth': float}

        with ``depth`` in ``[0, 1]``.

    Returns
    -------
    dict
        ``{'color': [(r, g, b), ...], 'depth': [float, ...]}`` with one entry
        per input pixel.

    Contract
    --------
    * Colour is the arithmetic mean of all sample colours (a simple box
      filter; samples are equally weighted).
    * Depth is the *nearest* sample, i.e. the minimum depth because ``0`` is
      the near plane.  Averaging depth blurs the silhouette and produces
      false contours at geometric edges.
    """
    colors = []
    depths = []

    for pixel_samples in samples:
        count = len(pixel_samples)
        if count == 0:
            colors.append((0.0, 0.0, 0.0))
            depths.append(1.0)
            continue

        r = g = b = 0.0
        nearest = float("inf")
        for sample in pixel_samples:
            sample_color = sample["color"]
            r += sample_color[0]
            g += sample_color[1]
            b += sample_color[2]

            # Depth 0 is the near plane, so the nearest sample is the minimum.
            if sample["depth"] < nearest:
                nearest = sample["depth"]

        colors.append((r / count, g / count, b / count))
        depths.append(nearest)

    return {"color": colors, "depth": depths}


def validate_gbuffer(gbuffer):
    """Validate a G-Buffer dictionary and return human-readable errors.

    Parameters
    ----------
    gbuffer : dict
        The structure described in :func:`shade_gbuffer`.  Required keys are
        ``position``, ``normal``, ``albedo``, ``depth``, ``opaque`` and
        ``ambient``.

    Returns
    -------
    list[str]
        Error messages; an empty list means the G-Buffer is valid.

    Contract
    --------
    The following must be reported:

    * any missing required channel;
    * mismatched lengths among ``position``, ``normal``, ``albedo``, ``depth``
      and ``opaque``;
    * a normal whose length differs from ``1`` by more than ``1e-6`` (a
      non-unit normal silently changes ``N.L`` and breaks octahedral round
      trips, so it must never be accepted);
    * a depth outside ``[0, 1]``.
    """
    errors = []
    required = ("position", "normal", "albedo", "depth", "opaque", "ambient")

    for channel in required:
        if channel not in gbuffer:
            errors.append("missing required channel: {}".format(channel))

    if errors:
        return errors

    channels = ("position", "normal", "albedo", "depth", "opaque")
    lengths = {channel: len(gbuffer[channel]) for channel in channels}
    expected = lengths["position"]
    for channel in channels:
        if lengths[channel] != expected:
            errors.append(
                "channel length mismatch: '{}' has {} entries, expected {}".format(
                    channel, lengths[channel], expected
                )
            )

    for index, normal in enumerate(gbuffer["normal"]):
        length = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        if abs(length - 1.0) > 1e-6:
            errors.append(
                "normal at index {} is not unit length: |n| = {}".format(index, length)
            )

    for index, depth in enumerate(gbuffer["depth"]):
        if depth < 0.0 or depth > 1.0:
            errors.append(
                "depth at index {} is outside [0, 1]: {}".format(index, depth)
            )

    return errors


def light_batch(frames, config):
    """Shade a batch of independent frames.

    Parameters
    ----------
    frames : list[dict]
        G-Buffer dictionaries, one per frame.
    config : dict
        Must contain ``'lights'``: the light list accepted by
        :func:`shade_gbuffer`.

    Returns
    -------
    dict
        ``{'frames': [colors_frame_0, colors_frame_1, ...],
        'frame_count': len(frames)}`` where every entry is exactly
        ``shade_gbuffer(frame, lights)``.

    Contract
    --------
    Frames are independent.  No normal, position, depth or colour may leak from
    one frame into another, so a cached or reused G-Buffer reference is a bug:
    frame ``i`` must be shaded with frame ``i``'s own normals.  Input frames
    must not be mutated.
    """
    lights = config["lights"]

    # Shade each frame from its own G-Buffer; nothing is cached across frames.
    shaded_frames = [shade_gbuffer(frame, lights) for frame in frames]

    return {"frames": shaded_frames, "frame_count": len(frames)}
