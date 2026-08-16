from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.models import Deployment, OperationPlan
from app.tasks.engine import TaskContext


class OperationExecutor:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        deployment_service,
        discovery_service,
    ):
        self.session_factory = session_factory
        self.deployment_service = deployment_service
        self.discovery_service = discovery_service

    def handler(self, context: TaskContext, payload: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(payload["plan_id"])
        with self.session_factory() as db:
            plan = db.get(OperationPlan, plan_id)
            if not plan or plan.status not in {"approved", "executing"}:
                raise ValueError("Operation plan is not approved")
            plan.status = "executing"
            steps = list(plan.steps)
            db.commit()
        results: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            context.check_control()
            if not step.get("executable"):
                results.append({"index": index, "status": "skipped", "reason": "Not executable"})
                continue
            operation = step.get("operation")
            if operation == "rescan_inventory":
                with self.session_factory() as db:
                    result = self.discovery_service.scan_all(db)
            else:
                action = {
                    "start_deployment": "start",
                    "stop_deployment": "stop",
                    "restart_deployment": "restart",
                }.get(operation)
                if not action:
                    results.append({"index": index, "status": "skipped", "reason": "Not allowed"})
                    continue
                deployment_id = str(step["deployment_id"])
                with self.session_factory() as db:
                    deployment = db.get(Deployment, deployment_id)
                    if deployment is None:
                        raise ValueError("Deployment was not found; retry the operation")
                    action_payload = {
                        "deployment_id": deployment_id,
                        "action": action,
                        "expected_container_id": deployment.container_id,
                        "expected_container_name": deployment.container_name,
                    }
                result = self.deployment_service.action_handler(
                    context,
                    action_payload,
                )
            results.append({"index": index, "status": "succeeded", "result": result})
            context.update(progress=(index + 1) / max(len(steps), 1) * 100)
        with self.session_factory() as db:
            plan = db.get(OperationPlan, plan_id)
            if plan:
                plan.status = "completed"
                plan.result_json = {"steps": results}
                db.commit()
        return {"plan_id": plan_id, "steps": results}
