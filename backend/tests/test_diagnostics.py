from app.services.diagnostics import parse_diagnostic_content, sanitize_steps


def test_diagnostic_parser_accepts_json_fence():
    result = parse_diagnostic_content(
        '```json\n{"summary":"High memory", "diagnosis":"Two models", '
        '"risk":"low", "steps":[]}\n```'
    )

    assert result["summary"] == "High memory"


def test_unknown_and_shell_operations_are_not_executable():
    steps = sanitize_steps(
        [
            {
                "operation": "restart_deployment",
                "deployment_id": "dep-1",
                "reason": "Recover health",
            },
            {"operation": "shell", "command": "rm -rf /", "reason": "unsafe"},
        ],
        known_deployment_ids={"dep-1"},
    )

    assert steps[0]["executable"] is True
    assert steps[1]["executable"] is False
    assert "command" not in steps[1]


def test_plan_must_be_approved_before_execution(authenticated_client):
    from app.models import OperationPlan

    with authenticated_client.app.state.database.session_factory() as db:
        plan = OperationPlan(
            summary="Rescan inventory",
            diagnosis="Inventory may be stale",
            risk="low",
            steps=[{"operation": "rescan_inventory", "reason": "Refresh", "executable": True}],
        )
        db.add(plan)
        db.commit()
        plan_id = plan.id

    response = authenticated_client.post(f"/api/diagnostics/{plan_id}/approve")

    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "running", "succeeded"}
    with authenticated_client.app.state.database.session_factory() as db:
        approved = db.get(OperationPlan, plan_id)
        assert approved.status in {"approved", "executing", "completed"}
        assert approved.approved_by == "admin"
