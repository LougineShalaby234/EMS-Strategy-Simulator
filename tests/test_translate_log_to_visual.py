import pytest
import json
import re
from translate_log_to_visual import (
    extract_simulation_state_blocks,
    process_frame,
    extract_remaining_route,
    build_marker,
    build_route_lines,
    parse_simulation_state
)

@pytest.fixture
def sample_log_text():
    """
    A small mock log snippet with:
    - A SimulationState(...) block
    - A line "Decisions made this step:"
    - Some route data
    """
    return """
=== DETAILED SIMULATION LOG ===

BEGINNING OF EXPERIMENT STATE:
SimulationState(
  time=2025-01-27 08:15:00,
  Ambulances=[
    <Ambulance 1, status=on the way to emergency, pos=(40.123, -75.456), remaining_route=[(1, (40.125, -75.458)), (2, (40.127, -75.460))], contains_patient=False>,
  ],
  RescueStations=[
    <RescueStation 2, loc=(40.200, -75.400)>,
  ],
  Hospitals=[
    <Hospital 5, loc=(40.300, -75.500), num_beds=20, free_beds=20>,
  ],
  EmergenciesRecent=[
    <Emergency 10, status=waiting_for_assignment, loc=(40.250, -75.450)>,
  ],
  EmergenciesPast=[
  ]
)
Decisions made this step: [('SendAmbulanceToEmergency', 1, 10)]

END OF STEP at simulation time 2025-01-27 08:15:00
Current Simulation State:
...
"""


def test_process_frame(sample_log_text):
    blocks = extract_simulation_state_blocks(sample_log_text)
    state_block, decisions_str = blocks[0]
    frame = process_frame(state_block, decisions_str)
    assert isinstance(frame, dict), "process_frame should return a dictionary"
    assert "description" in frame
    assert "tags" in frame
    assert "lines" in frame
    # Check if there's an ambulance in tags
    ambulance_tags = [t for t in frame["tags"] if t["icon"] == "ambulance"]
    assert len(ambulance_tags) == 1, "Should have exactly one ambulance marker"
    # Check the line segments
    assert len(frame["lines"]) >= 1, "Should have route line segments for the ambulance"

def test_extract_remaining_route():
    mock_str = "remaining_route=[(1, (40.001, -75.001)), (2, (40.002, -75.002))]"
    route = extract_remaining_route(mock_str)
    assert len(route) == 2
    assert route[0] == [40.001, -75.001]
    assert route[1] == [40.002, -75.002]

def test_build_marker_ambulance():
    """
    Verify that build_marker picks the correct color/icon for an ambulance with a certain status.
    """
    obj_str = "Ambulance 3, status=relocating, pos=(40.500, -75.500)"
    marker = build_marker(obj_str, "Ambulance")
    assert marker["icon"] == "ambulance"
    assert marker["color"] != "#000000", "Relocating should map to a known color, not black"
    assert marker["position"] == [40.500, -75.500]

def test_build_route_lines():
    """
    Ensure we generate colored line segments from an ambulance route, 
    with status-based color logic.
    """
    # Ambulance is on the way to hospital => #FF4500 (example in code)
    obj_str = "Ambulance 7, status=on the way to hospital, remaining_route=[(1, (40.000, -75.000)), (2, (40.001, -75.001))]"
    lines = build_route_lines(obj_str)
    assert len(lines) == 1, "Should produce one line segment from two points"
    assert lines[0]["start"] == [40.000, -75.000]
    assert lines[0]["end"] == [40.001, -75.001]
    assert lines[0]["color"] == "#FF4500", "on the way to hospital should map to #FF4500"

def test_parse_simulation_state():
    """
    Basic check that parse_simulation_state returns the right structure.
    """
    mock_state_str = """
    SimulationState(
      time=2023-01-01 10:00:00,
      Ambulances=[
        <Ambulance 1, status=idle, pos=(40.111, -75.222), remaining_route=[]>,
      ],
      RescueStations=[
        <RescueStation 7, loc=(40.200, -75.300)>,
      ],
      Hospitals=[
        <Hospital 2, loc=(40.250, -75.350)>,
      ],
      EmergenciesRecent=[
      ],
      EmergenciesPast=[
        <Emergency 10, status=finished, loc=(40.150, -75.280)>,
      ]
    )
    """
    state = parse_simulation_state(mock_state_str)
    assert state["time"] == "2023-01-01 10:00:00"
    assert len(state["Ambulances"]) == 1
    assert len(state["RescueStations"]) == 1
    assert len(state["Hospitals"]) == 1
    assert len(state["EmergenciesRecent"]) == 0
    assert len(state["EmergenciesPast"]) == 1
