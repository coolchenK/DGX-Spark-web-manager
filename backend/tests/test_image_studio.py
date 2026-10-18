import base64
import json
from uuid import uuid4

import pytest
import respx
from app.image_schema import SIGMAS, ImageRequest
from app.models import Deployment
from app.tasks.engine import TaskContext
from pydantic import ValidationError


def test_image_studio_authenticated_generation_and_worker_lane(authenticated_client):
    c = authenticated_client
    app = c.app
    app.state.task_engine.stop()
    session = app.state.database.session_factory
    with session() as db:
        d = Deployment(
            name="Viggle",
            runtime="viggle",
            api_model_name="viggle",
            managed=True,
            status="running",
            health="healthy",
            capabilities=["image_generation", "image"],
            endpoint_url="http://127.0.0.1:8123",
            config={},
        )
        db.add(d)
        db.commit()
        did = d.id
    png = b"\x89PNG\r\n\x1a\nexample"
    reference = "data:image/png;base64," + base64.b64encode(png).decode()
    payload = {"deployment_id": did, "prompt": "change the background", "images": [reference]}
    csrf = c.headers.pop("X-CSRF-Token")
    assert c.post("/api/images", json=payload).status_code == 403
    c.headers["X-CSRF-Token"] = csrf
    r = c.post("/api/images", json=payload)
    assert r.status_code == 202, r.text
    tid = r.json()["id"]
    engine = app.state.task_engine
    assert engine._claim(lane="control") is None
    assert engine._claim(lane="download") is None
    claim = engine._claim(lane="image")
    assert claim[0] == tid
    assert "images" not in claim[2]["request"]
    with respx.mock as mock:
        route = mock.post("http://127.0.0.1:8123/generate").respond(
            200,
            text="\n".join(
                [
                    json.dumps({"type": "progress", "progress": 50, "message": "sampling"}),
                    json.dumps(
                        {
                            "type": "result",
                            "image": base64.b64encode(png).decode(),
                            "seed": 12,
                            "width": 1024,
                            "height": 1024,
                            "steps": 6,
                            "seconds": 10,
                        }
                    ),
                ]
            ),
        )
        result = app.state.image_studio_service.generate(TaskContext(session, tid), claim[2])
        assert json.loads(route.calls.last.request.content)["images"] == [reference]
        assert route.calls.last.request.headers["authorization"].startswith("Bearer ")
    engine._finish(tid, "succeeded", result=result)
    assert c.get(result["image_url"]).content == png
    assert (
        "attachment" in c.get(result["image_url"] + "?download=true").headers["content-disposition"]
    )
    assert c.get("/api/images").json()[0]["result"]["seed"] == 12
    upscale = {"deployment_id": did, "prompt": "refine details", "operation": "upscale",
               "source_task_id": tid, "sr_engine": "supir", "sr_control": 0.93}
    assert c.post("/api/images", json={**upscale, "source_task_id": "../etc"}).status_code == 422
    assert c.post("/api/images", json={**upscale, "images": [reference]}).status_code == 422
    child = c.post("/api/images", json=upscale)
    assert child.status_code == 202, child.text
    child_id = child.json()["id"]
    assert child.json()["title"] == "SUPIR 细节重建"
    child_record = c.get("/api/images").json()[0]
    assert child_record["source_task_id"] == tid
    assert child_record["request"]["operation"] == "upscale"
    assert child_record["request"]["sr_engine"] == "supir"
    assert child_record["request"]["sr_control"] == 0.93
    copied = app.state.image_studio_service.directory(child_id) / "input-0.bin"
    assert copied.read_bytes() == png
    assert c.delete("/api/images/" + tid).status_code == 204
    assert not app.state.image_studio_service.directory(tid).exists()
    assert copied.read_bytes() == png  # Removing the parent cannot break the queued child.
    with pytest.raises(ValueError):
        app.state.image_studio_service.directory("../etc")
    other = str(uuid4())
    app.state.image_studio_service.root.mkdir(exist_ok=True)
    (app.state.image_studio_service.root / other).symlink_to("/tmp", target_is_directory=True)
    with pytest.raises(ValueError):
        app.state.image_studio_service.directory(other)
    key = c.post("/api/keys", json={"name": "image test"}).json()["key"]
    assert (
        c.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer " + key},
            json={"model": "viggle", "messages": [{"role": "user", "content": "x"}]},
        ).status_code
        == 400
    )
    c.post("/api/auth/logout")
    assert c.get("/api/images").status_code == 401


def test_image_request_limits_and_recipe():
    assert SIGMAS[6] == [1, 0.9375, 0.875, 0.75, 0.5, 0.25]
    assert ImageRequest(prompt="test").dimensions() == (1024, 1024)
    w, h = ImageRequest(prompt="test").dimensions((4000, 100))
    assert w % 32 == h % 32 == 0 and w / h < 1.85
    for changes in (
        {"prompt": " "},
        {"steps": 7},
        {"width": 1025},
        {"images": ["https://example.com/x"]},
        {"images": ["data:image/png;base64,eA=="], "resolution": 2048},
        {"aspect_ratio": "custom", "width": 2816, "height": 2816},
        {"operation": "texture"},
        {"strength": float("nan")},
    ):
        with pytest.raises(ValidationError):
            ImageRequest.model_validate({"prompt": "test", **changes})
    source = "data:image/png;base64,eA=="
    assert ImageRequest(prompt="test", images=[source], operation="refine").dimensions(
        (1024, 1024)
    ) == (2048, 2048)
    assert ImageRequest(prompt="test", images=[source], operation="texture").dimensions(
        (2048, 2048)
    ) == (4096, 4096)
    with pytest.raises(ValueError, match="阶段限制"):
        ImageRequest(prompt="test", images=[source], operation="refine").dimensions((2048, 2048))


def test_learned_upscaling_dimensions_and_limits():
    source = "data:image/png;base64,eA=="
    values = dict(prompt="wood grain", images=[source], operation="upscale")
    assert ImageRequest(**values).dimensions((1024, 768)) == (4096, 3072)
    assert ImageRequest(**values, sr_resolution=2048).dimensions((1024, 768)) == (2048, 1536)
    for invalid in [
        {"sr_engine": "unknown"},
        {"sr_steps": 5},
        {"sr_cfg": float("nan")},
        {"sr_control": 0.5},
    ]:
        with pytest.raises(ValidationError):
            ImageRequest(**values, **invalid)
    with pytest.raises(ValueError):
        ImageRequest(**values).dimensions((512,512))
