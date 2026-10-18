"""Serve the official Laya Router; this is a decision API, not a chat API."""

import argparse
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
Label = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "score", "noul"]
    instructions: Text
    criteria: dict[Label, Label | None] | list[Label] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        if self.type == "noul":
            if self.criteria is not None:
                raise ValueError("noul questions do not accept criteria")
        else:
            if self.criteria is None or not 2 <= len(self.criteria) <= 20:
                raise ValueError("choice/score require 2-20 criteria at the default head budget")
            if self.type == "score" and not isinstance(self.criteria, list):
                raise ValueError("score criteria must be an ordered list")
            if isinstance(self.criteria, list) and len(set(self.criteria)) != len(self.criteria):
                raise ValueError("criteria must be distinct")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Text
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: dict[Label, Question] = Field(min_length=1, max_length=16)
    checkpoint: Literal["english", "multilingual", "typed-decisions"] | None = None

    @model_validator(mode="after")
    def bound_state(self):
        if len(json.dumps(self.state, ensure_ascii=False).encode()) > 65536:
            raise ValueError("state exceeds the 64 KiB request limit")
        return self


def create_app(model_path: str, served_name: str) -> FastAPI:
    gate = Lock()

    @asynccontextmanager
    async def lifespan(app):
        import laya
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Laya service requires a working CUDA device")
        torch.set_num_threads(4)
        paths = {
            "english": model_path,
            "multilingual": str(Path(model_path) / "multilingual"),
            "typed-decisions": str(Path(model_path) / "typed-decisions"),
        }
        router = laya.Router(models=paths, device="cuda", preload=True)
        for name in paths:
            router.predict(
                "A customer requests a refund.",
                {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
                model=name,
            )
            if router.load(name).device.type != "cuda":
                raise RuntimeError(f"{name} fell back to CPU during warmup")
        app.state.router = router
        yield
        router.unload()

    app = FastAPI(title="Laya decisions", lifespan=lifespan)

    @app.get("/health")
    def health():
        router = app.state.router
        devices = {name: str(router.load(name).device) for name in router.loaded}
        if any(not value.startswith("cuda") for value in devices.values()):
            raise HTTPException(503, "An SDK checkpoint fell back to CPU")
        return {"status": "ok", "checkpoints": devices}

    @app.get("/v1/models")
    def models():
        health()
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "owned_by": "convaiinnovations",
                    "capabilities": ["decision"],
                    "max_output_tokens": 0,
                    "context_windows": {
                        "english": 512,
                        "multilingual": 1024,
                        "typed-decisions": 1024,
                    },
                }
            ],
        }

    @app.post("/v1/decisions")
    def decide(body: DecisionRequest):
        if body.model != served_name:
            raise HTTPException(404, "Unknown model")
        # ponytail: serialize GPU forwards; add a measured batching queue if demand warrants it.
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Laya is busy; retry after the current decision")
        try:
            started = time.perf_counter()
            router = app.state.router
            result = router.predict(
                body.state,
                {
                    name: value.model_dump(exclude_none=True)
                    for name, value in body.questions.items()
                },
                model=body.checkpoint,
            )
            health()
            agent = router.load(result["routing"]["model"])
            result["model"] = served_name
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            result["limits"] = {
                "context_length": agent.cfg["max_len"],
                "head_max_len": agent.cfg["head_max_len"],
            }
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            gate.release()

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="laya")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model_path, args.served_model_name), host="0.0.0.0", port=args.port)
