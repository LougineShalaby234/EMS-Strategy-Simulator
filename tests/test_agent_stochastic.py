import pytest
from unittest.mock import MagicMock
from agents.agent_Stochastic import RandomAgent

# Constants to match your script
AMBULANCE_STATUS_IDLE = "idle"
AMBULANCE_STATUS_RELOCATING = "relocating"
EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT = "waiting_for_assignment"

@pytest.fixture
def dummy_state():
    # Create dummy emergencies
    emergency1 = MagicMock()
    emergency1.id = "E1"
    emergency1.status = EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT
    emergency1.hospitals = ["H1", "H2"]

    emergency2 = MagicMock()
    emergency2.id = "E2"
    emergency2.status = "resolved"

    # Dummy hospitals
    hospital1 = MagicMock()
    hospital1.id = "H1"
    hospital1.free_beds = 2
    hospital1.location = (10.0, 10.0)

    hospital2 = MagicMock()
    hospital2.id = "H2"
    hospital2.free_beds = 0
    hospital2.location = (20.0, 20.0)

    # Dummy rescue station
    station = MagicMock()
    station.id = "S1"
    station.location = (5.0, 5.0)

    # Dummy ambulances
    amb1 = MagicMock()
    amb1.id = "A1"
    amb1.status = AMBULANCE_STATUS_IDLE
    amb1.emergency_assigned = None
    amb1.hospital_assigned = None
    amb1.contains_patient = False
    amb1.remaining_route = []

    amb2 = MagicMock()
    amb2.id = "A2"
    amb2.status = AMBULANCE_STATUS_RELOCATING
    amb2.emergency_assigned = None
    amb2.hospital_assigned = None
    amb2.contains_patient = False
    amb2.remaining_route = []

    # Simulation state
    state = MagicMock()
    state.emergencies_past = [emergency1]
    state.emergencies_recent = [emergency2]
    state.hospitals = [hospital1, hospital2]
    state.rescue_stations = [station]
    state.ambulances = [amb1, amb2]

    return state, emergency1, hospital1, amb1

def test_assigns_ambulance_to_emergency(dummy_state):
    state, emergency1, _, _ = dummy_state
    agent = RandomAgent(state)
    decisions = agent.get_actions(state)

    # Extract all ambulances assigned in decisions
    assigned_ambulances = [dec[1] for dec in decisions if dec[0] == "SendAmbulanceToEmergency"]

    assert any(dec[0] == "SendAmbulanceToEmergency" and dec[2] == emergency1.id for dec in decisions)
    assert all(amb_id in [amb.id for amb in state.ambulances] for amb_id in assigned_ambulances)

def test_assigns_hospital_to_ambulance_with_patient(dummy_state):
    state, emergency1, hospital1, amb1 = dummy_state

    # Modify ambulance to have picked up a patient
    amb1.contains_patient = True
    amb1.remaining_route = []
    amb1.emergency_assigned = emergency1.id
    emergency1.status = "active"

    state.emergencies_past = [emergency1]
    agent = RandomAgent(state)
    decisions = agent.get_actions(state)

    # Either H1 or fallback hospital should be used
    assert any(dec[0] == "SendAmbulanceToHospital" for dec in decisions)

def test_does_not_relocate_randomly_when_prob_zero(dummy_state, monkeypatch):
    state, _, _, amb1 = dummy_state

    amb1.status = AMBULANCE_STATUS_IDLE
    amb1.emergency_assigned = None

    # Override random.random to always return 1.0 (so relocation doesn't trigger)
    monkeypatch.setattr("random.random", lambda: 1.0)

    agent = RandomAgent(state)
    decisions = agent.get_actions(state)

    assert all(dec[0] != "RelocateAmbulance" for dec in decisions)
