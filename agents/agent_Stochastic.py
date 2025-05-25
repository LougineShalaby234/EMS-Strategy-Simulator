#!/usr/bin/env python3
"""
Custom agent that:
1.Randomly assigns available ambulances to emergencies

2. Randomly selects hospitals (preferring those with free beds)

3. occasionally relocates idle ambulances (10%)
"""

import random
from agents.base_class import AgentBase

# Status string constants (should match those used in the simulator)
AMBULANCE_STATUS_IDLE = "idle"
AMBULANCE_STATUS_BROKEN = "broken"
AMBULANCE_STATUS_EN_ROUTE_HOSPITAL = "on the way to hospital"
AMBULANCE_STATUS_EN_ROUTE_EMERGENCY = "on the way to emergency"
AMBULANCE_STATUS_RELOCATING = "relocating"

EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT = "waiting_for_assignment"
EMERGENCY_STATUS_WAITING_FOR_AMBULANCE = "waiting_for_ambulance"

class RandomAgent(AgentBase):
    """
    An agent that assigns ambulances to emergencies and hospitals, and occasionally relocates idle ambulances.
    """

    def __init__(self, state):
        """
        Initialize the agent with the current simulation state.
        
        :param state: The SimulationState object.
        """
        self.state = state

    def get_actions(self, state):
        """
        Generate actions based on the simulation state.
        
        :param state: The current SimulationState object.
        :return: A list of decision tuples.
        """
        decisions = []

        # 1) Process emergencies (past first, then recent) for ambulance assignment.
        for emergency in state.emergencies_past + state.emergencies_recent:
            if emergency.status == EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT:
                available_ambulances = [
                    amb for amb in state.ambulances
                    if amb.emergency_assigned is None and amb.status in [AMBULANCE_STATUS_IDLE, AMBULANCE_STATUS_RELOCATING]
                ]
                if available_ambulances:
                    selected_amb = random.choice(available_ambulances)
                    decisions.append(("SendAmbulanceToEmergency", selected_amb.id, emergency.id))
        
        # 2) Assign ambulances that have arrived at an emergency to a hospital
        for amb in state.ambulances:
            if amb.emergency_assigned is not None and amb.hospital_assigned is None and amb.contains_patient:
                if not amb.remaining_route:
                    associated_emergency = next((em for em in state.emergencies_past + state.emergencies_recent if em.id == amb.emergency_assigned), None)
                    if associated_emergency:
                        viable_hospitals = [
                            h for h in state.hospitals if h.id in associated_emergency.hospitals and h.free_beds > 0
                        ]
                        chosen_hospital = random.choice(viable_hospitals) if viable_hospitals else random.choice(state.hospitals) if state.hospitals else None
                        if chosen_hospital:
                            decisions.append(("SendAmbulanceToHospital", amb.id, chosen_hospital.id))
        
        # 3) Relocate idle ambulances with a small probability
        relocation_destinations = state.hospitals + state.rescue_stations
        for amb in state.ambulances:
            if amb.status == AMBULANCE_STATUS_IDLE and amb.emergency_assigned is None:
                if random.random() <= 0.1 and relocation_destinations:
                    dest = random.choice(relocation_destinations)
                    decisions.append(("RelocateAmbulance", amb.id, dest.location[0], dest.location[1]))
        
        return decisions
