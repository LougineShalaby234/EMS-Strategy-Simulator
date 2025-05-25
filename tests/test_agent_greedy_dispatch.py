import pytest
from unittest.mock import MagicMock
from agents.agent_GreedyDispatch import NearestAgent

AMBULANCE_STATUS_IDLE = "idle"
AMBULANCE_STATUS_EN_ROUTE_HOSPITAL = "on the way to hospital"
AMBULANCE_STATUS_RELOCATING = "relocating"
EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT = "waiting_for_assignment"

@pytest.fixture
def dummy_state():
    # Dummy emergency
    emergency = MagicMock()
    emergency.id = "E1"
    emergency.status = EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT
    emergency.location = (0, 0)
    emergency.hospitals = ["H1", "H2"]

    # Dummy hospitals
    hospital1 = MagicMock()
    hospital1.id = "H1"
    hospital1.location = (1, 1)
    hospital1.free_beds = 2

    hospital2 = MagicMock()
    hospital2.id = "H2"
    hospital2.location = (5, 5)
    hospital2.free_beds = 0

    # Dummy rescue station
    station = MagicMock()
    station.location = (10, 10)

    # Dummy ambulances
    amb1 = MagicMock()
    amb1.id = "A1"
    amb1.status = AMBULANCE_STATUS_IDLE
    amb1.position = (2, 2)
    amb1.emergency_assigned = None
    amb1.hospital_assigned = None
    amb1.contains_patient = False
    amb1.remaining_route = []

    amb2 = MagicMock()
    amb2.id = "A2"
    amb2.status = AMBULANCE_STATUS_RELOCATING
    amb2.position = (10, 10)
    amb2.emergency_assigned = None
    amb2.hospital_assigned = None
    amb2.contains_patient = False
    amb2.remaining_route = []

    state = MagicMock()
    state.emergencies_past = [emergency]
    state.emergencies_recent = []
    state.ambulances = [amb1, amb2]
    state.hospitals = [hospital1, hospital2]
    state.rescue_stations = [station]

    return state, emergency, amb1, hospital1

def test_assigns_nearest_ambulance(dummy_state):
    state, emergency, _, _ = dummy_state
    agent = NearestAgent(state)

    agent.compute_route = lambda src, dst: [(5, dst)]  # mock travel time
    decisions = agent.get_actions(state)

    assert any(
        dec[0] == "SendAmbulanceToEmergency" and dec[2] == emergency.id
        for dec in decisions
    )

def test_assigns_patient_to_nearest_hospital(dummy_state):
    state, emergency, amb, hospital1 = dummy_state

    amb.contains_patient = True
    amb.emergency_assigned = emergency.id
    amb.remaining_route = []
    state.emergencies_past = [emergency]

    agent = NearestAgent(state)
    agent.compute_route = lambda src, dst: [(len(str(dst)), dst)]  # dummy distance

    decisions = agent.get_actions(state)

    assert any(
        dec[0] == "SendAmbulanceToHospital" and dec[1] == amb.id
        for dec in decisions
    )

def test_redirects_from_full_hospital(dummy_state):
    state, _, amb, hospital1 = dummy_state

    amb.status = AMBULANCE_STATUS_EN_ROUTE_HOSPITAL
    amb.hospital_assigned = hospital1.id
    amb.position = (0, 0)

    hospital1.free_beds = 0

    
    hospital2 = MagicMock()
    hospital2.id = "H2"
    hospital2.location = (2, 2)
    hospital2.free_beds = 2
    state.hospitals = [hospital1, hospital2]

    agent = NearestAgent(state)
    agent.compute_route = lambda src, dst: [(1, dst)]  # mock travel time

    decisions = agent.get_actions(state)

    assert any(dec[0] == "ChangeTargetHospital" and dec[1] == amb.id for dec in decisions)


def test_does_not_relocate_without_probability(monkeypatch, dummy_state):
    state, _, amb, _ = dummy_state
    amb.status = AMBULANCE_STATUS_IDLE
    amb.emergency_assigned = None

    monkeypatch.setattr("random.random", lambda: 1.0)  # never trigger relocation

    agent = NearestAgent(state)
    agent.compute_route = lambda src, dst: [(1, dst)]
    decisions = agent.get_actions(state)

    assert all(dec[0] != "RelocateAmbulance" for dec in decisions)
