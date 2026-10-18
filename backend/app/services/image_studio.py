import base64
import hashlib
import hmac
import json
from pathlib import Path
from uuid import UUID

import httpx

from app.image_schema import MAX_OUTPUT_BYTES, ImageRequest, decode_reference
from app.models import Deployment


def image_runtime_token(secret_key: str) -> str:
    return hmac.new(secret_key.encode(), b"dgx-viggle-image-runtime-v1", hashlib.sha256).hexdigest()


class ImageStudioService:
    def __init__(self, *, data_dir, session_factory, runtime_token):
        self.root = Path(data_dir) / "image-studio"
        self.session_factory = session_factory
        self.runtime_token = runtime_token

    def directory(self, task_id: str) -> Path:
        if str(UUID(task_id)) != task_id:
            raise ValueError("Invalid image task ID")
        directory = self.root / task_id
        if directory.is_symlink() or not directory.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Invalid image directory")
        return directory

    def save_inputs(self, task_id: str, request: ImageRequest) -> list[str]:
        directory = self.directory(task_id)
        directory.mkdir(parents=True, exist_ok=False)
        headers = []
        for index, value in enumerate(request.images):
            (directory / f"input-{index}.bin").write_bytes(decode_reference(value))
            headers.append(value.partition(",")[0])
        return headers

    def generate(self, context, payload) -> dict:
        directory = self.directory(context.task_id)
        request = dict(payload["request"])
        request["images"] = [
            header
            + ","
            + base64.b64encode((directory / f"input-{index}.bin").read_bytes()).decode()
            for index, header in enumerate(payload.get("image_headers", []))
        ]
        request = ImageRequest.model_validate(request).model_dump(mode="json")
        with self.session_factory() as db:
            deployment = db.get(Deployment, payload["deployment_id"])
            if not deployment or deployment.runtime != "viggle" or deployment.status != "running":
                raise ValueError("图片模型未运行，请从部署实例启动后重试")
            endpoint = deployment.endpoint_url
        context.check_control()
        with httpx.stream(
            "POST",
            f"{endpoint.rstrip('/')}/generate",
            json=request,
            headers={"Authorization": f"Bearer {self.runtime_token}"},
            timeout=httpx.Timeout(1800, connect=10),
            trust_env=False,
        ) as response:
            if not response.is_success:
                response.read()
                raise ValueError(f"图片服务返回 {response.status_code}，请检查模型日志后重试")
            for line in response.iter_lines():
                context.check_control()
                if not line:
                    continue
                if len(line) > MAX_OUTPUT_BYTES * 4 // 3 + 1024**2:
                    raise ValueError("生成结果超过大小上限")
                event = json.loads(line)
                if event.get("type") == "progress":
                    context.update(progress=event["progress"], message=event["message"])
                elif event.get("type") == "error":
                    raise ValueError(str(event.get("message", "图片生成失败"))[:500])
                elif event.get("type") == "result":
                    raw = base64.b64decode(event.pop("image"), validate=True)
                    if len(raw) > MAX_OUTPUT_BYTES:
                        raise ValueError("生成结果超过 64 MiB")
                    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                        raise ValueError("图片服务未返回有效 PNG")
                    temporary = directory / "output.tmp"
                    temporary.write_bytes(raw)
                    temporary.replace(directory / "output.png")
                    return {
                        **{
                            key: event[key]
                            for key in ("seed", "width", "height", "steps", "seconds")
                        },
                        "image_url": f"/api/images/{context.task_id}/output",
                        "prompt": request["prompt"],
                        "reference_count": len(request["images"]),
                        "operation": request["operation"],
                        "sr_engine": event.get("sr_engine"),
                        "strength": request["strength"],
                        "tile_count": event.get("tile_count", 0),
                        "source_task_id": payload.get("source_task_id"),
                    }
        raise ValueError("图片生成连接意外结束，请重试")
