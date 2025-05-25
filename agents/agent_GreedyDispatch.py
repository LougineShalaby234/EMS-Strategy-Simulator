"""
Custom agent that:
1) Assigns the nearest available ambulance to each emergency.
2) Assigns patients to the nearest hospital with free beds (or the nearest possible hospital if none have free beds).
3) Occasionally relocates idle ambulances to a random hospital or rescue station.
4) Redirects ambulances to a new hospital if their current target hospital becomes full.
"""

import random
from agents.base_class import AgentBase

# Status string constants
AMBULANCE_STATUS_IDLE = "idle"
AMBULANCE_STATUS_BROKEN = "broken"
AMBULANCE_STATUS_EN_ROUTE_HOSPITAL = "on the way to hospital"
AMBULANCE_STATUS_EN_ROUTE_EMERGENCY = "on the way to emergency"
AMBULANCE_STATUS_RELOCATING = "relocating"

EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT = "waiting_for_assignment"
EMERGENCY_STATUS_WAITING_FOR_AMBULANCE = "waiting_for_ambulance"

class NearestAgent(AgentBase):
    """
    An agent that assigns ambulances and hospitals based on nearest distance using travel time estimation.
    """
    def __init__(self, state):
        self.state = state

    def get_actions(self, state):
        decisions = []
        
        def travel_time(source, destination):
            route = self.compute_route(source, destination)
            return route[-1][0] if route else float("inf")
        
        # Assign nearest available ambulance only to emergencies that are waiting for assignment
        for emergency in state.emergencies_past + state.emergencies_recent:
            if emergency.status == EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT:
                available_ambulances = [
                    amb for amb in state.ambulances
                    if amb.emergency_assigned is None and amb.status in [AMBULANCE_STATUS_IDLE, AMBULANCE_STATUS_RELOCATING]
                ]
                
                if available_ambulances:
                    nearest_ambulance = min(available_ambulances, key=lambda amb: travel_time(amb.position, emergency.location))
                    decisions.append(("SendAmbulanceToEmergency", nearest_ambulance.id, emergency.id))
        
        # Assign ambulances that have arrived at emergencies to the nearest hospital with a free bed
        for amb in state.ambulances:
            if amb.emergency_assigned is not None and amb.hospital_assigned is None and amb.contains_patient:
                if not amb.remaining_route:
                    associated_emergency = next((em for em in state.emergencies_past + state.emergencies_recent if em.id == amb.emergency_assigned), None)
                    
                    if associated_emergency:
                        viable_hospitals = [h for h in state.hospitals if h.id in associated_emergency.hospitals and h.free_beds > 0]
                        
                        if viable_hospitals:
                            chosen_hospital = min(viable_hospitals, key=lambda h: travel_time(amb.position, h.location))
                        else:
                            chosen_hospital = min(state.hospitals, key=lambda h: travel_time(amb.position, h.location)) if state.hospitals else None
                        
                        if chosen_hospital:
                            decisions.append(("SendAmbulanceToHospital", amb.id, chosen_hospital.id))
        
        # Relocate idle ambulances with a small probability
        relocation_destinations = state.hospitals + state.rescue_stations
        for amb in state.ambulances:
            if amb.status == AMBULANCE_STATUS_IDLE and amb.emergency_assigned is None:
                if random.random() < 0.01 and relocation_destinations:
                    dest = random.choice(relocation_destinations)
                    decisions.append(("RelocateAmbulance", amb.id, dest.location[0], dest.location[1]))
        
        # Check if any ambulances are en route to a full hospital
        for amb in state.ambulances:
            if amb.status == AMBULANCE_STATUS_EN_ROUTE_HOSPITAL:
                current_hospital = next((h for h in state.hospitals if h.id == amb.hospital_assigned), None)
                if current_hospital and current_hospital.free_beds == 0:
                    alternative_hospitals = [h for h in state.hospitals if h.free_beds > 0]
                    if alternative_hospitals:
                        new_hospital = min(alternative_hospitals, key=lambda h: travel_time(amb.position, h.location))
                        decisions.append(("ChangeTargetHospital", amb.id, new_hospital.id))
        
        return decisions
