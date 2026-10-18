"""Shared request contract for the panel and the pinned Viggle runtime."""

import base64
import binascii
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 65 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
LORA_FILE = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
SIGMAS = {
    4: [1.0, 0.75, 0.5, 0.25],
    6: [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
    # The author's dense-text preset, explicitly documented in Known limitations.
    8: [1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125],
}


def decode_reference(value: str) -> bytes:
    header, separator, encoded = value.partition(",")
    if not separator or header not in {
        "data:image/png;base64",
        "data:image/jpeg;base64",
        "data:image/webp;base64",
    }:
        raise ValueError("参考图仅支持 PNG、JPEG、WebP")
    if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("每张参考图不能超过 16 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("参考图编码无效") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("参考图为空或超过 16 MiB")
    return raw


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    images: list[str] = Field(default_factory=list, max_length=3)
    steps: Literal[4, 6, 8] = 6
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    resolution: Literal[1024, 1536, 2048] = 1024
    aspect_ratio: Literal["auto", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "custom"] = (
        "auto"
    )
    width: int = Field(default=1024, ge=512, le=2816, multiple_of=32)
    height: int = Field(default=1024, ge=512, le=2816, multiple_of=32)
    operation: Literal["generate", "refine", "texture", "upscale"] = "generate"
    scale: Literal[1.5, 2.0] = 2.0
    strength: float = Field(default=0.35, ge=0, le=0.5, allow_inf_nan=False)
    enhance_steps: int = Field(default=8, ge=4, le=20)
    tile_size: Literal[512, 768, 1024] = 1024
    tile_overlap: Literal[64, 128, 192, 256] = 128
    texture_amount: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    sr_engine: Literal["realesrgan", "supir"] = "realesrgan"
    sr_resolution: Literal[2048, 4096] = 4096
    sr_steps: int = Field(default=30, ge=20, le=60)
    sr_cfg: float = Field(default=4.0, ge=1, le=8, allow_inf_nan=False)
    sr_control: float = Field(default=1.0, ge=0.8, le=1.0, allow_inf_nan=False)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入提示词或编辑指令")
        return value.strip()

    @field_validator("images")
    @classmethod
    def valid_references(cls, values: list[str]) -> list[str]:
        for value in values:
            decode_reference(value)
        return values

    @model_validator(mode="after")
    def valid_size(self):
        if self.operation != "generate":
            if len(self.images) != 1:
                raise ValueError("放大处理需要且仅支持一张源图")
            return self
        area_limit = 1536**2 if self.images else 2048**2
        if self.images and self.resolution == 2048:
            raise ValueError("图片编辑支持 1024 或 1536 档")
        if self.aspect_ratio == "custom" and self.width * self.height > area_limit:
            raise ValueError("自定义尺寸超出当前模式的像素上限")
        return self

    def dimensions(self, reference_size: tuple[int, int] | None = None) -> tuple[int, int]:
        if self.operation != "generate":
            if reference_size is None:
                raise ValueError("请选择需要放大的源图")
            if self.operation == "upscale":
                factor = self.sr_resolution / max(reference_size)
                width, height = (round(n * factor / 32) * 32 for n in reference_size)
                if not 1 <= factor <= 4 or min(width, height) < 512:
                    raise ValueError("超分支持最多 4 倍放大，源图长边不能大于目标尺寸")
                return width, height
            factor = self.scale if self.operation == "refine" else 4096 / max(reference_size)
            if self.operation == "texture" and not 1 <= factor <= 4:
                raise ValueError("4K 纹理增强的源图长边应在 1024–4096 像素之间")
            width, height = (round(n * factor / 32) * 32 for n in reference_size)
            pixel_limit = 2048**2 * 1.02 if self.operation == "refine" else 4096**2
            if (
                min(width, height) < 512 or max(width, height) > 4096
                or width * height > pixel_limit
            ):
                raise ValueError("放大尺寸超出当前阶段限制：精修约 2K，纹理增强长边 4K")
            return width, height
        if self.aspect_ratio == "custom":
            return self.width, self.height
        if self.aspect_ratio == "auto":
            ratio = reference_size[0] / reference_size[1] if reference_size else 1.0
        else:
            w, h = self.aspect_ratio.split(":")
            ratio = int(w) / int(h)
        # Same area/32-pixel alignment as the reference pipeline. Bound extreme
        # uploaded panoramas to the tested output aspect ratios.
        ratio = min(16 / 9, max(9 / 16, ratio))
        width = round(math.sqrt(self.resolution**2 * ratio) / 32) * 32
        height = round(math.sqrt(self.resolution**2 / ratio) / 32) * 32
        return width, height
