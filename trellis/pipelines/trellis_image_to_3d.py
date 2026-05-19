from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..utils.style_utils import split_and_combine_image, shuffle_image_patches, tile_and_resize_image
from ..utils.morphing_utils import *
from ..modules.spatial import patchify
from ..utils.ot_coherence import build_ot_motion_field, build_ss_anchors
import cv2
import os
try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    chamfer_distance = None
try:
    from scipy import ndimage as _ndi
except ImportError:
    _ndi = None
from glob import glob
os.environ["U2NET_PATH"] = "/root/autodl-tmp/MorphAny3D/u2net.onnx"

MORPHING_ATTENTION_DEFAULTS = {
    "modify": False,
    "sparse_modify": False,
    "gate_attn": False,
    "sparse_gate_attn": False,
    "gate_mode": "logits",
    "modify_mode": "legacy",
    "modify_precheck": False,
    "modify_precheck_threshold": 2.5,
    "modify_precheck_min_query_tokens": 64,
    "modify_precheck_min_key_tokens": 64,
    "modify_lambda_scale": 0.3,
    "modify_max_passes": 4,
    "modify_stop_conflict": 0.5,
    "modify_temperature": 1.0,
    "modify_sink_threshold": 0.15,
    "modify_sink_top_count_weight": 0.5,
    "modify_sink_mass_weight": 0.5,
    "modify_sink_penalty_type": "linear",
    "modify_sink_cap_iters": 4,
    "gate_entropy_threshold": 6.0,
    "gate_max_logit_threshold": 1.0,
    "gate_qk_confidence_threshold": 1.0,
    "ss_ca_oc_flag": False,
    "ss_ca_oc_grid_size": 16,
    "ss_ca_oc_desc_dim": 32,
    "delete_loaded_ca_oc_cache": False,
    "topo_repair_enable": False,
    "topo_repair_mu": 0.3,
    "topo_repair_uncert_weight": 1.0,
    "topo_repair_thin_weight": 0.5,
    "topo_repair_air_weight": 1.0,
    "topo_repair_target_weight": 1.0,
    "topo_repair_fg_weight": 0.5,
}


def _ss_token_coords_16(device: torch.device) -> torch.Tensor:
    coords = torch.meshgrid(
        *[torch.arange(16, device=device) for _ in range(3)],
        indexing="ij",
    )
    return torch.stack(coords, dim=-1).reshape(-1, 3)


def _endpoint_coords_to_occ16(
    coords_raw: torch.Tensor,
    ss_token_coords: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    coords = coords_raw.detach()
    if coords.ndim != 2 or coords.shape[-1] not in (3, 4):
        raise ValueError(f"expected endpoint coords [N, 3] or [N, 4], got {tuple(coords.shape)}")
    if coords.shape[-1] == 4:
        coords = coords[:, 1:]

    coords = coords.long()
    max_coord = int(coords.max().item()) if coords.numel() > 0 else 0
    if max_coord >= 16:
        coords_16 = torch.div(coords, 4, rounding_mode="floor").clamp_(0, 15)
        note = "endpoint coords treated as 64^3 voxel coords; token_coord = floor(voxel_coord / 4), clamped to [0, 15]"
    else:
        coords_16 = coords.clamp_(0, 15)
        note = "endpoint coords treated as already aligned to 16^3 token coords; clamped to [0, 15]"

    token_coords = ss_token_coords.detach().long().cpu()
    coord_to_index = {tuple(coord.tolist()): idx for idx, coord in enumerate(token_coords)}
    occ = torch.zeros(token_coords.shape[0], dtype=torch.bool)
    for coord in coords_16.cpu():
        idx = coord_to_index.get(tuple(coord.tolist()))
        if idx is not None:
            occ[idx] = True
    return occ, note


def _with_morphing_attention_defaults(morphing_params: dict) -> dict:
    for key, value in MORPHING_ATTENTION_DEFAULTS.items():
        morphing_params.setdefault(key, value)
    return morphing_params


def _topo_repair_structure():
    if _ndi is None:
        return None
    return _ndi.generate_binary_structure(3, 1)


def _topo_repair_reachable_air(local_occ: np.ndarray) -> np.ndarray:
    bg = ~local_occ.astype(bool)
    if _ndi is None or not bg.any():
        return np.zeros_like(bg, dtype=bool)

    labels, _ = _ndi.label(bg, structure=_topo_repair_structure())
    boundary = np.zeros_like(bg, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    boundary_labels = np.unique(labels[boundary & bg])
    boundary_labels = boundary_labels[boundary_labels != 0]
    if boundary_labels.size == 0:
        return np.zeros_like(bg, dtype=bool)
    return np.isin(labels, boundary_labels)


def _topo_repair_air_opening_gain(occ: np.ndarray, cluster: np.ndarray, pad: int = 3) -> float:
    if _ndi is None or not cluster.any():
        return 0.0

    pts = np.argwhere(cluster)
    lo = np.maximum(pts.min(axis=0) - pad, 0)
    hi = np.minimum(pts.max(axis=0) + pad + 1, np.array(occ.shape))
    slc = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    local_occ = occ[slc].astype(bool).copy()
    local_cluster = cluster[slc].astype(bool)

    reachable_before = _topo_repair_reachable_air(local_occ)
    local_occ_after = local_occ & ~local_cluster
    reachable_after = _topo_repair_reachable_air(local_occ_after)
    opened = reachable_after & ~reachable_before
    gain = float(opened.sum()) / float(max(int(local_cluster.sum()), 1))
    return float(np.clip(np.log1p(gain) / np.log1p(16.0), 0.0, 1.0))


def _morphing_cache_name(cache_path: str) -> str:
    path = os.path.normpath(str(cache_path))
    base = os.path.basename(path)
    if base == "cache":
        return os.path.basename(os.path.dirname(path))
    return base


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        # dinov2_model = torch.hub.load('/data1/GuoJunWen/.cache/torch/hub/facebookresearch_dinov2_main', name, source='local', pretrained=True)
        dinov2_model = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', name,source='local', pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])

        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, List[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    @torch.no_grad()
    def encode_ss(self, z_s: torch.Tensor, add_pos_emb: bool = True, do_layer_norm: bool = True) -> torch.Tensor:
        """
        Encode sparse-structure latent z_s into structural tokens.

        Args:
            z_s: torch.Tensor, shape [B, C, R, R, R]
            add_pos_emb: whether to add the pretrained absolute position embedding
            do_layer_norm: whether to apply layer norm on output tokens

        Returns:
            torch.Tensor: structural tokens, shape [B, N, D]
        """
        flow_model = self.models['sparse_structure_flow_model']

        assert isinstance(z_s, torch.Tensor), f"z_s must be torch.Tensor, got {type(z_s)}"
        assert z_s.ndim == 5, f"z_s must be [B, C, R, R, R], got {tuple(z_s.shape)}"

        z_s = z_s.to(self.device)

        # basic shape checks against pretrained SS flow model
        if hasattr(flow_model, "in_channels"):
            assert z_s.shape[1] == flow_model.in_channels, \
                f"z_s channels {z_s.shape[1]} != flow_model.in_channels {flow_model.in_channels}"
        if hasattr(flow_model, "resolution"):
            assert z_s.shape[2] == flow_model.resolution and z_s.shape[3] == flow_model.resolution and z_s.shape[4] == flow_model.resolution, \
                f"z_s spatial size {tuple(z_s.shape[2:])} != flow_model.resolution {(flow_model.resolution,) * 3}"

        # mimic the front half of SparseStructureFlowModel.forward(...)
        x = patchify(z_s, flow_model.patch_size)
        x = x.reshape(x.shape[0], -1, x.shape[-1])   # [B, N, patch_dim]
        x = flow_model.input_layer(x)                # [B, N, hidden_size]

        if add_pos_emb and hasattr(flow_model, "pos_emb") and flow_model.pos_emb is not None:
            x = x + flow_model.pos_emb[None]

        if do_layer_norm:
            x = F.layer_norm(x, x.shape[-1:])

        return x


    def get_ss_cond(self, z_s: torch.Tensor) -> dict:
        """
        Get structural conditioning tokens from SS latent z_s.
        Interface mirrors get_cond().

        Args:
            z_s: torch.Tensor, shape [B, C, R, R, R]

        Returns:
            dict with keys:
                'cond': structural tokens [B, N, D]
                'neg_cond': zero tokens with same shape
        """
        cond = self.encode_ss(z_s)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {}
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def _topo_repair_target_support(
        self,
        morphing_params: dict,
        spatial_shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        support = torch.zeros(spatial_shape, device=device, dtype=dtype)
        target_cache = morphing_params.get("tar_load_cache_path", None)
        if target_cache is None:
            return support

        zs_path = os.path.join(target_cache, "coords_zs.pt")
        coords_path = os.path.join(target_cache, "coords.pt")
        try:
            if os.path.exists(zs_path):
                z_t = torch.load(zs_path, map_location=device).to(device)
                target_logits = self.models['sparse_structure_decoder'](z_t)
                while target_logits.ndim > 3:
                    target_logits = target_logits[0]
                target_prob = torch.sigmoid(target_logits.to(dtype=dtype))
                target_occ = (target_logits > 0).to(dtype=dtype)
                density = F.avg_pool3d(target_occ[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
                return (0.5 * target_prob + 0.5 * density).clamp_(0.0, 1.0)

            if os.path.exists(coords_path):
                coords = torch.load(coords_path, map_location="cpu")
                if coords.ndim != 2 or coords.shape[-1] not in (3, 4):
                    return support
                coords = coords[:, 1:] if coords.shape[-1] == 4 else coords
                coords = coords.long()
                shape = torch.tensor(spatial_shape, dtype=torch.long)
                if coords.numel() > 0:
                    coords = coords.clamp_min(0)
                    coords = torch.minimum(coords, shape[None] - 1)
                    occ = torch.zeros(spatial_shape, device=device, dtype=dtype)
                    occ[coords[:, 0].to(device), coords[:, 1].to(device), coords[:, 2].to(device)] = 1.0
                    density = F.avg_pool3d(occ[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
                    return (0.5 * occ + 0.5 * density).clamp_(0.0, 1.0)
        except Exception as exc:
            print(f"[Topo repair] warning: failed to build target support: {exc}")
        return support

    def _soft_topology_aware_voxel_repair(
        self,
        voxel_logits: torch.Tensor,
        morphing_params: dict,
    ) -> torch.Tensor:
        if _ndi is None:
            print("[Topo repair] warning: scipy is unavailable; skip topology-aware repair")
            return voxel_logits

        original_shape = voxel_logits.shape
        if voxel_logits.ndim == 5:
            logits_bc = voxel_logits
        elif voxel_logits.ndim == 4:
            logits_bc = voxel_logits[:, None]
        else:
            raise ValueError(f"expected voxel logits [B, C, D, H, W] or [B, D, H, W], got {tuple(original_shape)}")

        repaired = logits_bc.clone()

        mu = float(morphing_params.get("topo_repair_mu", 0.3))
        uncert_weight = max(float(morphing_params.get("topo_repair_uncert_weight", 1.0)), 1e-6)
        thin_weight = float(morphing_params.get("topo_repair_thin_weight", 0.5))
        air_weight = float(morphing_params.get("topo_repair_air_weight", 1.0))
        target_weight = float(morphing_params.get("topo_repair_target_weight", 1.0))
        fg_weight = float(morphing_params.get("topo_repair_fg_weight", 0.5))

        for b in range(logits_bc.shape[0]):
            for c in range(logits_bc.shape[1]):
                logits_item = logits_bc[b, c]
                occ_before_t = logits_item > 0
                occ = occ_before_t.detach().cpu().numpy().astype(bool)
                logits_np = logits_item.detach().float().cpu().numpy()

                if not occ.any():
                    continue

                target_support = self._topo_repair_target_support(
                    morphing_params,
                    tuple(logits_item.shape),
                    logits_item.device,
                    logits_item.dtype,
                )
                target_support_np = target_support.detach().float().cpu().numpy()

                thickness = _ndi.distance_transform_edt(occ).astype(np.float32)
                thinness = np.zeros_like(thickness, dtype=np.float32)
                thinness[occ] = 1.0 / (thickness[occ] + 1e-6)
                uncertainty = np.exp(-np.abs(logits_np) * uncert_weight).astype(np.float32)
                fg_density = _ndi.uniform_filter(occ.astype(np.float32), size=3, mode="constant", cval=0.0)

                candidate = occ & (uncertainty >= np.exp(-1.0)) & (thickness <= 3.0)

                air_gain = np.zeros_like(logits_np, dtype=np.float32)
                if candidate.any():
                    labels, comp_count = _ndi.label(candidate, structure=_topo_repair_structure())
                    for comp_idx in range(1, int(comp_count) + 1):
                        cluster = labels == comp_idx
                        gain = _topo_repair_air_opening_gain(occ, cluster)
                        if gain > 0:
                            air_gain[cluster] = gain

                target_protect = np.clip(target_support_np * target_weight, 0.0, 1.0)
                fg_protect = np.clip(fg_density * fg_weight, 0.0, 1.0)
                score = (
                    candidate.astype(np.float32)
                    * uncertainty
                    * np.power(np.clip(thinness, 0.0, 1.0), max(thin_weight, 0.0))
                    * np.power(np.clip(air_gain, 0.0, 1.0), max(air_weight, 0.0))
                    * (1.0 - target_protect)
                    * (1.0 - fg_protect)
                ).astype(np.float32)

                score_t = torch.from_numpy(score).to(device=logits_item.device, dtype=logits_item.dtype)
                repaired[b, c] = logits_item - mu * score_t

        if voxel_logits.ndim == 4:
            repaired = repaired[:, 0]
        return repaired.reshape(original_shape)

    def sample_sparse_structure_morphing(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        morphing_params: dict = {}
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        morphing_params = _with_morphing_attention_defaults(morphing_params)
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution

        if not morphing_params["init_morphing_flag"]:
            noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        else:
            src_noise = torch.load(os.path.join(morphing_params["src_load_cache_path"], "coords_zs_init.pt")).to(self.device)
            tar_noise = torch.load(os.path.join(morphing_params["tar_load_cache_path"], "coords_zs_init.pt")).to(self.device)
            noise = feature_interp(src_noise, tar_noise, morphing_params["alpha"])
        explicit_sampler_params = dict(sampler_params)
        sampler_params = {**self.sparse_structure_sampler_params, **explicit_sampler_params}
        if "ss_steps" in morphing_params and "steps" not in explicit_sampler_params:
            sampler_params["steps"] = int(morphing_params["ss_steps"])
        sampler_kwargs = dict(morphing_params)
        sampler_kwargs["ss_num_steps"] = sampler_params.get("steps", sampler_kwargs.get("ss_num_steps", None))
        sampler_kwargs["ss_token_grid_size"] = getattr(flow_model, "resolution", reso) // getattr(flow_model, "patch_size", 1)

        if sampler_kwargs.get("ot_coherence_enabled", False) and sampler_kwargs.get("ot_coherence_stage", "ss") == "ss":
            cache = morphing_params.get("_ot_motion_field_cache", None)
            if cache is None:
                try:
                    src_coords = torch.load(os.path.join(morphing_params["src_load_cache_path"], "coords.pt"), map_location=self.device)
                    tgt_coords = torch.load(os.path.join(morphing_params["tar_load_cache_path"], "coords.pt"), map_location=self.device)
                    if src_coords.shape[-1] == 4:
                        src_coords = src_coords[:, 1:]
                    if tgt_coords.shape[-1] == 4:
                        tgt_coords = tgt_coords[:, 1:]
                    with torch.no_grad():
                        src_anchors = build_ss_anchors(
                            src_coords,
                            grid_size=reso,
                            patch_size=sampler_kwargs.get("ot_anchor_patch_size", 4),
                            max_anchors=sampler_kwargs.get("ot_max_anchors", 512),
                        )
                        tgt_anchors = build_ss_anchors(
                            tgt_coords,
                            grid_size=reso,
                            patch_size=sampler_kwargs.get("ot_anchor_patch_size", 4),
                            max_anchors=sampler_kwargs.get("ot_max_anchors", 512),
                        )
                        cache = build_ot_motion_field(src_anchors, tgt_anchors, sampler_kwargs)
                    morphing_params["_ot_motion_field_cache"] = cache
                except Exception as exc:
                    print(f"[MCRF] disabled for this run: {exc}")
                    cache = None
            if cache is not None:
                sampler_kwargs["ot_motion_field"] = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in cache.items()}
        sampler_kwargs.pop("_ot_motion_field_cache", None)

        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            **sampler_kwargs,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        voxel_logits = decoder(z_s)
        if morphing_params.get("topo_repair_enable", False):
            voxel_logits = self._soft_topology_aware_voxel_repair(voxel_logits, morphing_params)
        voxels = voxel_logits > 0
        coords = torch.argwhere(voxels)[:, [0, 2, 3, 4]].int()
        if "save_cache_path" in morphing_params and "morphing_idx" in morphing_params:
            coords_path = f"{morphing_params['save_cache_path']}/coords_morphing{morphing_params['morphing_idx']}.pt"
            if not os.path.exists(coords_path):
                torch.save(coords.detach().cpu(), coords_path)

        if "save_cache_path" in morphing_params and "morphing_idx" not in morphing_params:
            torch.save(noise.detach().cpu(), f"{morphing_params['save_cache_path']}/coords_zs_init.pt")
            torch.save(z_s.detach().cpu(), f"{morphing_params['save_cache_path']}/coords_zs.pt")
        return coords, voxels, z_s

    def decode_ss_latent(self, z_s: torch.tensor)->dict:
        decoder = self.models['sparse_structure_decoder']
        voxels = decoder(z_s)>0
        coords = torch.argwhere(voxels)[:, [0, 2, 3, 4]].int()
        return voxels, coords

    def _ensure_ss_endpoint_occ16_for_mavf(self, morphing_params: dict) -> None:
        if not morphing_params.get("mavf_enable", False):
            return
        src_cache = morphing_params.get("src_load_cache_path", None)
        tar_cache = morphing_params.get("tar_load_cache_path", None)
        if src_cache is None or tar_cache is None:
            return
        tar_name = _morphing_cache_name(tar_cache)
        endpoint_path = os.path.join(src_cache, f"ss_endpoint_occ16_to_{tar_name}.pt")
        morphing_params["ss_endpoint_occ16_path"] = endpoint_path
        if os.path.exists(endpoint_path) and not morphing_params.get("overwrite_ss_endpoint_occ16_cache", False):
            return

        try:
            src_coords_raw = torch.load(os.path.join(src_cache, "coords.pt"), map_location="cpu")
            tar_coords_raw = torch.load(os.path.join(tar_cache, "coords.pt"), map_location="cpu")
            ss_token_coords = _ss_token_coords_16(device=torch.device("cpu"))
            occ_s_16, note_s = _endpoint_coords_to_occ16(src_coords_raw, ss_token_coords)
            occ_t_16, note_t = _endpoint_coords_to_occ16(tar_coords_raw, ss_token_coords)
            torch.save(
                {
                    "ss_token_coords": ss_token_coords.cpu(),
                    "occ_s_16": occ_s_16.cpu(),
                    "occ_t_16": occ_t_16.cpu(),
                    "src_coords_raw": src_coords_raw.detach().cpu() if torch.is_tensor(src_coords_raw) else src_coords_raw,
                    "tar_coords_raw": tar_coords_raw.detach().cpu() if torch.is_tensor(tar_coords_raw) else tar_coords_raw,
                    "note": f"source: {note_s}; target: {note_t}",
                    "source_occupied_token_count": int(occ_s_16.sum().item()),
                    "target_occupied_token_count": int(occ_t_16.sum().item()),
                    "src_cache_path": str(src_cache),
                    "tar_cache_path": str(tar_cache),
                },
                endpoint_path,
            )
        except Exception as exc:
            print(f"[MAVF V0] warning: failed to build endpoint occupancy cache {endpoint_path}: {exc}")
        

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {}
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def sample_slat_morphing(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        morphing_params: dict = {}
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        morphing_params = _with_morphing_attention_defaults(morphing_params)
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        if not morphing_params["init_morphing_flag"]:
            noise = sp.SparseTensor(
                feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
                coords=coords,
            )
        else:
            src_noise = torch.load(os.path.join(morphing_params["src_load_cache_path"], "slat_init.pt")).to(self.device)
            src_coords = torch.load(os.path.join(morphing_params["src_load_cache_path"], "coords.pt")).to(self.device)
            tar_noise = torch.load(os.path.join(morphing_params["tar_load_cache_path"], "slat_init.pt")).to(self.device)
            tar_coords = torch.load(os.path.join(morphing_params["tar_load_cache_path"], "coords.pt")).to(self.device)
            src_eucdist_matrix = cal_eucdist_matrix(coords[:, 1:].detach().float(), src_coords[:, 1:].float())
            src_indices = torch.argmin(src_eucdist_matrix, dim=1)
            tar_eucdist_matrix = cal_eucdist_matrix(coords[:, 1:].detach().float(), tar_coords[:, 1:].float())
            tar_indices = torch.argmin(tar_eucdist_matrix, dim=1)
            feat_noise = feature_interp(src_noise[src_indices], tar_noise[tar_indices], morphing_params["alpha"])  
            noise = sp.SparseTensor(
                feats=feat_noise,
                coords=coords,
            )

        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            **morphing_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if "save_cache_path" in morphing_params and "morphing_idx" not in morphing_params:
            torch.save(noise.feats.detach().cpu(), f"{morphing_params['save_cache_path']}/slat_init.pt")
            torch.save(coords.detach().cpu(), f"{morphing_params['save_cache_path']}/coords.pt")

        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_morphing(
        self,
        src_img: Image.Image,
        tar_img: Image.Image,
        morphing_params: dict = {},
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field']
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            preprocess_image (bool): Whether to preprocess the image.
        """
        torch.manual_seed(seed)
        morphing_params = _with_morphing_attention_defaults(morphing_params)

        src_img = self.preprocess_image(src_img)
        src_cond = self.get_cond([src_img])
        
        tar_img = self.preprocess_image(tar_img)
        tar_cond = self.get_cond([tar_img])
        morphing_params["tar_cond"] = tar_cond["cond"]
        self._ensure_ss_endpoint_occ16_for_mavf(morphing_params)

        coords, voxels, z_s = self.sample_sparse_structure_morphing(src_cond, num_samples, sparse_structure_sampler_params, morphing_params)

        if morphing_params["oc_flag"]:  
            if chamfer_distance is None:
                raise ImportError("pytorch3d is required when morphing_params['oc_flag'] is True")
            if os.path.exists(f"{morphing_params['save_cache_path']}/coords_morphing{morphing_params['tfsa_cache_idx']}.pt"):
                coords_cache = torch.load(f"{morphing_params['save_cache_path']}/coords_morphing{morphing_params['tfsa_cache_idx']}.pt").to(self.device)

                preprocess_coords = coords[:, 1:].detach() / 63 - 0.5
                preprocess_coords_cache = coords_cache[:, 1:] / 63 - 0.5
                
                chamfer_distance_loss = []
                chamfer_distance_loss.append(chamfer_distance(preprocess_coords.float().unsqueeze(0), preprocess_coords_cache.float().unsqueeze(0), single_directional=False)[0])
                for angle in [90, 180, 270]: 
                    coords_rot = rotate_pc(preprocess_coords, angle)
                    chamfer_distance_loss.append(chamfer_distance(coords_rot.float().unsqueeze(0), preprocess_coords_cache.float().unsqueeze(0), single_directional=False)[0])
                chamfer_distance_loss = torch.stack(chamfer_distance_loss)
                best_idx = torch.argmin(chamfer_distance_loss).item()
                if best_idx > 0:
                    print(f"Orientation correction: rotate {best_idx * 90} degrees")
                    with torch.no_grad():
                        tmp = torch.rot90(self.models['sparse_structure_decoder'](z_s)[0].permute(1, 2, 3, 0), k=best_idx, dims=[0, 1]).permute(3, 0, 1, 2)[None]
                        voxels = (tmp > 0)
                        coords = torch.argwhere(voxels)[:, [0, 2, 3, 4]].int()
                    files = glob(f"{morphing_params['save_cache_path']}/ss_sa_morphing{morphing_params['morphing_idx']}_*")
                    for f in files:
                        tmp_cache = torch.load(f)
                        cache_shape = tmp_cache["k"].shape
                        tmp_cache["k"] = torch.rot90(tmp_cache["k"].reshape((16,16,16,-1)), k=best_idx, dims=[0, 1]).reshape(cache_shape)
                        tmp_cache["v"] = torch.rot90(tmp_cache["v"].reshape((16,16,16,-1)), k=best_idx, dims=[0, 1]).reshape(cache_shape)
                        torch.save(tmp_cache, f)

            if not os.path.exists(f"{morphing_params['save_cache_path']}/coords_morphing{morphing_params['morphing_idx']}.pt"):
                torch.save(coords.detach().cpu(), f"{morphing_params['save_cache_path']}/coords_morphing{morphing_params['morphing_idx']}.pt")

        if "dual_tar_img" in morphing_params:
            dual_tar_img = self.preprocess_image(morphing_params["dual_tar_img"])
            dual_tar_cond = self.get_cond([dual_tar_img])
            morphing_params["tar_cond"] = dual_tar_cond["cond"]
            morphing_params["tar_load_cache_path"] = morphing_params["dual_tar_load_cache_path"]

        slat = self.sample_slat_morphing(src_cond, coords, slat_sampler_params, morphing_params)
        outputs = self.decode_slat(slat, formats)

        return outputs
    
    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
