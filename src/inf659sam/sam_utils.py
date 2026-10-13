"""Envoltorio minimo de SAM (transformers) que separa codificador y decodificador.

Separar `embed` (codificador de imagen, ~95% del computo) de `decode` (codificador de
prompts + decodificador de mascaras) permite cachear el embedding de cada chip una sola
vez y luego ejecutar muchas variantes de prompts/ablaciones y el ajuste del decodificador
(H2) sin volver a pagar el codificador.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import config
from .baseline import normalize_band


class SamWrapper:
    def __init__(self, model_dir: Path | None = None, attn_implementation: str | None = None,
                 device: str | None = None):
        from transformers import SamModel, SamProcessor

        model_dir = str(model_dir or config.MODEL_DIR)
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self.device = device or config.get_device()
        self.model = SamModel.from_pretrained(model_dir, **kw).eval().to(self.device)
        self.processor = SamProcessor.from_pretrained(model_dir)
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ entrada
    @staticmethod
    def sar_to_rgb(vv: np.ndarray, vh: np.ndarray, mode: str = "vh3") -> np.ndarray:
        """Representacion de entrada de 3 canales uint8.

        vh3     : VH replicada en 3 canales (pipeline v2 del equipo).
        vv_vh_d : [VV, VH, VV-VH] (composicion de bandas, Anexo A del E1).
        """
        if mode == "vh3":
            ch = normalize_band(vh)
            return np.stack([ch, ch, ch], axis=-1)
        if mode == "vv_vh_d":
            diff = np.where(np.isfinite(vv) & np.isfinite(vh), vv - vh, np.nan)
            return np.stack([normalize_band(vv), normalize_band(vh), normalize_band(diff)], axis=-1)
        raise ValueError(mode)

    def preprocess(self, rgb: np.ndarray) -> dict:
        inputs = self.processor(images=rgb, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"],
            "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs["reshaped_input_sizes"],
        }

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embedding del codificador (se devuelve en CPU para cachearlo o reutilizarlo)."""
        return self.model.get_image_embeddings(pixel_values.to(self.device)).cpu()

    # ------------------------------------------------------------------ prompts
    @staticmethod
    def scale_points(points, original_size, reshaped_size) -> list[list[float]]:
        """(x, y) en pixeles del chip -> marco de 1024 que ve SAM (igual que SamProcessor)."""
        oh, ow = [int(v) for v in original_size]
        rh, rw = [int(v) for v in reshaped_size]
        return [[x * (rw / ow), y * (rh / oh)] for x, y in points]

    def prompt_tensors(self, points, labels, original_size, reshaped_size):
        pts = torch.tensor([[self.scale_points(points, original_size, reshaped_size)]], dtype=torch.float32)
        lbl = torch.tensor([[list(labels)]], dtype=torch.int64)
        return pts, lbl

    # ------------------------------------------------------------------ decodificador
    def decode(self, emb: torch.Tensor, points, labels, original_sizes, reshaped_sizes,
               multimask: bool = True, input_masks: torch.Tensor | None = None, model=None,
               grad: bool = False) -> dict:
        """Ejecuta prompt encoder + mask decoder sobre un embedding ya calculado.

        Devuelve mascaras binarias a la resolucion del chip, puntajes de IoU predicho y logits
        de baja resolucion (256x256).
        """
        model = model or self.model
        pts, lbl = self.prompt_tensors(points, labels, original_sizes[0], reshaped_sizes[0])
        dev = self.device
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(image_embeddings=emb.to(dev), input_points=pts.to(dev), input_labels=lbl.to(dev),
                        multimask_output=multimask,
                        input_masks=None if input_masks is None else input_masks.to(dev))
        low = out.pred_masks.detach().float().cpu()
        masks = self.processor.image_processor.post_process_masks(low, original_sizes.cpu(), reshaped_sizes.cpu())[0]
        return {
            "masks": masks[0].numpy().astype(bool),                # (k, H, W)
            "scores": out.iou_scores[0, 0].detach().float().cpu().numpy(),  # (k,)
            "low_res_logits": low[0, 0],                           # (k, 256, 256)
        }


def mask_to_prompt_logits(mask: np.ndarray, size: int = 256, pos: float = 8.0) -> torch.Tensor:
    """Mascara binaria del chip -> prompt denso de SAM (logits 1x1x256x256, +pos dentro / -pos fuera)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")
    return (t * 2 - 1) * pos
