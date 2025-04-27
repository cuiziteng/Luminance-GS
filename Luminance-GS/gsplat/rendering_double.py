import math
from typing import Dict, Optional, Tuple
import sys
import torch
from torch import Tensor
from typing_extensions import Literal

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
gsplat_path = os.path.abspath(os.path.join(current_dir, '..', 'gsplat'))

sys.path.append(gsplat_path)

from cuda import _wrapper

from cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)


def rasterization_dual(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    colors_low: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
) -> Tuple[Tensor, Tensor, Dict]:
    
    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    
    if sh_degree is None:   # None
        # treat colors as post-activation values
        # colors should be in shape [N, D] or (C, N, D) (silently support)
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients. Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[1], colors.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
    )

    if packed:  # True
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    if not (
        colors.dim() == 3 and sh_degree is None
    ):  # silently support [C, N, D] color.
        colors = (
            colors[gaussian_ids] if packed else colors.expand(C, *([-1] * colors.dim()))
        )  # [nnz, D] or [C, N, 3]
        colors_low = (
            colors_low[gaussian_ids] if packed else colors_low.expand(C, *([-1] * colors_low.dim()))
        )  # [nnz, D] or [C, N, 3]

    else:
        if packed:
            colors = colors[camera_ids, gaussian_ids, :]
            colors_low = colors_low[camera_ids, gaussian_ids, :]

    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
        colors = spherical_harmonics(
            sh_degree, dirs, colors, masks=radii > 0
        )  # [nnz, D] or [C, N, 3]
        colors_low = spherical_harmonics(
            sh_degree, dirs, colors_low, masks=radii > 0
        )  # [nnz, D] or [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)
        colors_low = torch.clamp_min(colors_low + 0.5, 0.0)


    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:  # Here, RGB only
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        colors_low = torch.cat((colors_low, depths[..., None]), dim=-1)
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        colors_low = depths[..., None]
    else:  # RGB
        pass
    
    if colors.shape[-1] > 32:   # False
        # slice into 32-channel chunks
        n_chunks = (colors.shape[-1] + 31) // 32
        render_colors = []
        render_enh_colors = []
        render_alphas = []

        for i in range(n_chunks):
            colors_chunk = colors[..., i * 32 : (i + 1) * 32]
            colors_low_chunk = colors_low[..., i * 32 : (i + 1) * 32]
            backgrounds_chunk = (
                backgrounds[:, i * 32 : (i + 1) * 32]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_low_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_low_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_enh_colors.append(render_low_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_enh_colors = torch.cat(render_enh_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
        render_low_colors, render_low_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors_low,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    
    if render_mode in ["ED", "RGB+ED"]: # False
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
        render_low_colors = torch.cat(
            [
                render_low_colors[..., :-1],
                render_low_colors[..., -1:] / render_low_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
    }
    return render_colors, render_low_colors, render_alphas, render_low_alphas, meta

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # define Gaussians
    means = torch.randn((100, 3), device=device)
    quats = torch.randn((100, 4), device=device)
    scales = torch.rand((100, 3), device=device) * 0.1
    colors = torch.rand((100, 3), device=device)
    colors_low = torch.rand((100, 3), device=device)
    opacities = torch.rand((100,), device=device)
    # define cameras
    viewmats = torch.eye(4, device=device)[None, :, :]
    Ks = torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
    width, height = 300, 200
    # render
    colors, colors_low, alphas, alphas_low, meta = rasterization_dual(means, quats, scales, opacities, colors, colors_low, viewmats, Ks, width, height)
    print(colors.shape)
    print(colors_low.shape)