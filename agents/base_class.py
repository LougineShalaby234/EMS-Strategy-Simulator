#!/usr/bin/env python3
"""
Base class for all agent implementations.
"""


class AgentBase:
    """
    A base class that all agent implementations should inherit from.
    
    Agents have:
      - An __init__ that accepts the current SimulationState
      - A get_actions that accepts the current SimulationState
      - Access to compute_route (via direct import or other means)
    """

    def __init__(self, state):
        """
        Initialize the agent with the current simulation state.
        
        :param state: The entire SimulationState object.
        """
        self.state = state
        # If you just need to store a reference to compute_route:
        self.compute_route = None
        # Perform any other agent-specific setup here.

    def get_actions(self, state):
        """
        Return a list of actions/decisions based on the current simulation state.
        
        :param state: The current SimulationState object.
        :return: A list of decisions, e.g.:
            [
               ("SendAmbulanceToEmergency", ambulance_id, emergency_id),
               ("RelocateAmbulance", ambulance_id, (lat, long)),
               ...
            ]
        """
        # Example usage of compute_route within an agent:
        # route_info = self.compute_route((lat1, lng1), (lat2, lng2))
        
        # For now, return an empty list of decisions:
        return []
