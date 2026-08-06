from architecture_a.inspection import decide_condition_route
from shared.perception import Detection


def det(label, confidence):
    return Detection(label, confidence, (0, 0, 10, 10))


def test_damage_routes_to_rejection_with_priority():
    result = decide_condition_route(
        {"barcode": [det("package", 0.9), det("barcode", 0.8)],
         "damage": [det("package", 0.9), det("damage_mark", 0.7)]}
    )
    assert result.destination == "rejection_tray"
    assert not result.should_escalate


def test_barcode_routes_to_conveyor():
    result = decide_condition_route(
        {"barcode": [det("package", 0.9), det("barcode", 0.8)], "damage": []}
    )
    assert result.destination == "conveyor"


def test_barcode_in_damage_inspection_pose_routes_to_conveyor():
    result = decide_condition_route(
        {"barcode": [det("package", 0.9)],
         "damage": [det("package", 0.9), det("barcode", 0.92)]}
    )
    assert result.destination == "conveyor"


def test_grounded_package_without_barcode_routes_to_inspection():
    result = decide_condition_route(
        {"barcode": [det("package", 0.9)], "damage": [det("package", 0.8)]}
    )
    assert result.destination == "inspection_tray"


def test_missing_package_stops_and_escalates():
    result = decide_condition_route({"barcode": [], "damage": []})
    assert result.destination is None
    assert result.should_escalate


def test_low_confidence_damage_does_not_override_valid_barcode():
    result = decide_condition_route(
        {"barcode": [det("package", 0.9), det("barcode", 0.95)],
         "damage": [det("package", 0.9), det("damage_mark", 0.14)]}
    )
    assert result.destination == "conveyor"


def test_weak_barcode_does_not_count_as_present():
    result = decide_condition_route(
        {"barcode": [det("package", 0.95), det("barcode", 0.79)], "damage": []}
    )
    assert result.destination == "inspection_tray"


def test_weak_package_grounding_stops_instead_of_assuming_missing_barcode():
    result = decide_condition_route(
        {"barcode": [det("package", 0.79)], "damage": []}
    )
    assert result.should_escalate


def test_weak_damage_response_does_not_trigger_rejection():
    result = decide_condition_route(
        {"barcode": [det("package", 0.95), det("barcode", 0.95)],
         "damage": [det("damage_mark", 0.39)]}
    )
    assert result.destination == "conveyor"
