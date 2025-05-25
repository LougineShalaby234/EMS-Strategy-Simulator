#!/usr/bin/env python3
"""
Simulator for running multiple experiments with different scenarios and agents.

In this version:
1) Ambulance objects track the currently assigned emergency and hospital.
2) SimulationState uses two lists for emergencies: 'emergencies_recent' and 'emergencies_past'.
3) The agent's get_actions(...) now receives just the SimulationState instead of (events_this_minute, state).
4) Route computation is implemented using Google Maps API
5) Patient waiting times are properly tracked and reported
"""

import os
import glob
import csv
import random
import json
import importlib.util
import inspect
from datetime import datetime, timedelta
import requests
import math
import statistics
import copy
import re
import here_search.polyline_converter as pc
from dotenv import load_dotenv
import base64
import struct
import argparse


from agents.base_class import AgentBase

# ----------------------------
# String Constants for Statuses
# ----------------------------
AMBULANCE_STATUS_IDLE = "idle"
AMBULANCE_STATUS_BROKEN = "broken"
AMBULANCE_STATUS_EN_ROUTE_HOSPITAL = "on the way to hospital"
AMBULANCE_STATUS_EN_ROUTE_EMERGENCY = "on the way to emergency"
AMBULANCE_STATUS_RELOCATING = "relocating"
AMBULANCE_STATUS_AT_EMERGENCY = "at emergency"
AMBULANCE_STATUS_BEING_CLEANED = "being cleaned"

EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT = "waiting_for_assignment"
EMERGENCY_STATUS_WAITING_FOR_AMBULANCE = "waiting_for_ambulance"
EMERGENCY_STATUS_EN_ROUTE_HOSPITAL = "en_route_hospital"
EMERGENCY_STATUS_FINISHED = "finished"

#----------
# ENV APIs
#----------
load_dotenv()
GOOGLE_MAPS_API_KEY = ''
OPENROUTESERVICE_API_KEY = ''
VALHALLA_API_URL = ''
HERE_MAPS_API_KEY = os.getenv('HERE_MAPS_API_KEY')


#---------
# hyperparameters
#---------
RoutingMethod = "LinearInterpolation" # which routing method to be used
assert RoutingMethod in ["Google", "Here", "Openrouteservice", "LinearInterpolation", "Valhalla" ]
ProbNoTransport = 0.3 # how often emergencies will not need to be transported to a hospital
ProbDelay = 0.01 # how often an ambulance encounter a 1 minute random delay while moving forward
assert all(0 <= p <= 1 for p in [ProbNoTransport, ProbDelay]), "Probabilities must be between 0 and 1"
OnsiteTimeMax = 10 # maximum time the ambulance remain at Emergency scene 
CleaningTimeMax = 5 # maximum time the ambulance takes to clean after Hospital transport
AvgSpeedKph = 40 # average speed the vehichle would move , used in estimate_travel_time
# -------------
# Entity Classes
# -------------

class Ambulance:
    def __init__(self, amb_id, lat, lng, status=AMBULANCE_STATUS_IDLE,zone=0):
        self.id = amb_id
        self.position = (lat, lng)
        self.status = status
        self.zone=zone 
        # Track current route: list of (time_offset_minutes, (lat, lng)) tuples,
        # now generated for each full minute from start to end
        self.remaining_route = []
        # Which emergency is assigned (if any), and which hospital
        self.emergency_assigned = None
        self.hospital_assigned = None
        self.arrival_estimate = None  # estimated time of arrival at destination
        # Flag indicating if the ambulance currently contains a patient
        self.contains_patient = False
        # time remaining for an ambulance to get cleaned after the pattient was transported to the hospital
        self.cleaning_time_remaining = 0
        # how many minutes ambulance remains on-scene 
        self.onsite_time_remaining = 0
        
    def __repr__(self):
        return (f"<Ambulance {self.id}, "
                f"status={self.status}, "
                f"pos={self.position}, "
                f"contains_patient={self.contains_patient}, "
                f"emergency={self.emergency_assigned}, "
                f"remaining_route={str(self.remaining_route)}, "
                f"hospital={self.hospital_assigned}>")

class Emergency:
    def __init__(self, emerg_id, timestamp, lat, lng, possible_hospitals):
        self.id = emerg_id
        self.timestamp = timestamp  # When the emergency call was received
        self.location = (lat, lng)
        self.status = EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT
        self.hospitals = possible_hospitals  
        self.resolved = False  # True when patient arrives at hospital
        self.transport_decision_made = False # We only want to do the random “no‐transport” check *once*.
        
        # Timing metrics
        self.ambulance_assigned_time = None  # When an ambulance was assigned
        self.ambulance_arrival_time = None  # When ambulance reached the emergency
        self.hospital_arrival_time = None  # When patient arrived at hospital
        
        # Assigned ambulance and hospital
        self.assigned_ambulance = None
        self.assigned_hospital = None

    def get_wait_time_for_ambulance(self, current_time):
        """Return the time waited for an ambulance to arrive (in minutes)"""
        if self.ambulance_arrival_time:
            return (self.ambulance_arrival_time - self.timestamp).total_seconds() / 60
        return (current_time - self.timestamp).total_seconds() / 60
    
    def get_total_wait_time(self, current_time):
        """Return total time from call to hospital arrival (in minutes)"""
        if self.hospital_arrival_time:
            return (self.hospital_arrival_time - self.timestamp).total_seconds() / 60
        return (current_time - self.timestamp).total_seconds() / 60
        
    def __repr__(self):
        return (f"<Emergency {self.id}, "
                f"status={self.status}, "
                f"time={self.timestamp}, "
                f"loc={self.location}, "
                f"resolved={self.resolved}, "
                f"hospitals={self.hospitals}, "
                f"assignedAmb={self.assigned_ambulance}, "
                f"assignedHosp={self.assigned_hospital}>")

class Hospital:
    def __init__(self, hosp_id, lat, lng, num_beds, free_beds,zone=0):
        self.id = hosp_id
        self.location = (lat, lng)
        self.num_beds = num_beds
        self.free_beds = free_beds
        self.zone=zone

    def __repr__(self):
        return (f"<Hospital {self.id}, "
                f"loc={self.location}, "
                f"num_beds={self.num_beds}, "
                f"free_beds={self.free_beds}>")

class RescueStation:
    def __init__(self, station_id, lat, lng, available_ambulances,zone=0):
        self.id = station_id
        self.location = (lat, lng)
        self.available_ambulances = available_ambulances
        self.zone=zone

    def __repr__(self):
        return (f"<RescueStation {self.id}, "
                f"loc={self.location}, "
                f"availableAmb={self.available_ambulances}>")

# -------------------
# The Simulation State
# -------------------

class SimulationState:
    """
    Holds all relevant state information for the simulation.
    """
    def __init__(self):
        self.global_clock = None  # a datetime object
        self.ambulances = []
        self.hospitals = []
        self.rescue_stations = []
        self.emergencies_recent = []  # those that began in the last minute
        self.emergencies_past = []    # older or resolved emergencies

    # --- UPDATED __repr__ TO LIST ALL PROPERTIES IN DETAIL ---
    def __repr__(self):
        rep = []
        rep.append(f"SimulationState(\n  time={self.global_clock},")
        
        rep.append("  Ambulances=[")
        for a in self.ambulances:
            rep.append(f"    {repr(a)},")
        rep.append("  ],")
        
        rep.append("  RescueStations=[")
        for rs in self.rescue_stations:
            rep.append(f"    {repr(rs)},")
        rep.append("  ],")

        rep.append("  Hospitals=[")
        for h in self.hospitals:
            rep.append(f"    {repr(h)},")
        rep.append("  ],")

        rep.append("  EmergenciesRecent=[")
        for e in self.emergencies_recent:
            rep.append(f"    {repr(e)},")
        rep.append("  ],")

        rep.append("  EmergenciesPast=[")
        for e in self.emergencies_past:
            rep.append(f"    {repr(e)},")
        rep.append("  ]\n)")
        
        return "\n".join(rep)

# ----------------------
# Utility / Core Functions
# ----------------------

def parse_time_str(timestr):
    """
    Convert a string like '2025-01-27-08-15-00' into a datetime object.
    """
    return datetime.strptime(timestr, "%Y-%m-%d-%H-%M-%S")


def read_map(path):
    """
    Reads the map data from a JSON file and returns a SimulationState with
    ambulances, hospitals, rescue stations, etc.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    state = SimulationState()

    # Build list of hospitals
    for h_data in data.get("hospitals", []):
        hosp_id = h_data["id"]
        lat = h_data["location"]["lat"]
        lng = h_data["location"]["lng"]
        num_beds = h_data.get("capacity", 0)
        free_beds = num_beds
        zone= h_data.get("zone", 0)
        hospital_obj = Hospital(hosp_id, lat, lng, num_beds, free_beds)
        state.hospitals.append(hospital_obj)

    # Build list of rescue stations
    for rs_data in data.get("rescueStations", []):
        station_id = rs_data["id"]
        lat = rs_data["location"]["lat"]
        lng = rs_data["location"]["lng"]
        zone = rs_data.get("zone", 0)
        station_obj = RescueStation(station_id, lat, lng, 0,zone)
        state.rescue_stations.append(station_obj)

    # Build list of ambulances
    for amb_data in data.get("ambulances", []):
        amb_id = amb_data["id"]
        lat = amb_data["location"]["lat"]
        lng = amb_data["location"]["lng"]
        reported_status = amb_data.get("status", "")
        zone = amb_data.get("zone",0)
        if reported_status == "outOfService":
            status = AMBULANCE_STATUS_BROKEN
        elif reported_status == "idle":
            status = AMBULANCE_STATUS_IDLE
        else:
            status = AMBULANCE_STATUS_IDLE
        ambulance_obj = Ambulance(amb_id, lat, lng, status,zone)
        state.ambulances.append(ambulance_obj)

    return state


def load_scenario(scenario_path):
    """
    Read the scenario JSON and return a sorted list of Emergency objects.
    """
    with open(scenario_path, 'r') as f:
        data = json.load(f)

    emergencies_data = data.get("emergencies", [])
    emergencies = []
    for e_data in emergencies_data:
        e_id = e_data["id"]
        dt = parse_time_str(e_data["time"])
        lat = e_data["location"]["lat"]
        lng = e_data["location"]["lng"]
        possible_hosps = e_data.get("hospitals", [])
        emerg_obj = Emergency(e_id, dt, lat, lng, possible_hosps)
        emergencies.append(emerg_obj)

    emergencies.sort(key=lambda x: x.timestamp)
    return emergencies


def haversine_distance(point1, point2):
    """
    Calculate the distance between two lat/lng points in kilometers.
    Used as a fallback when the API is not available.
    """
    lat1, lon1 = point1
    lat2, lon2 = point2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    r = 6371
    return c*r


def estimate_travel_time(distance_km, AvgSpeedKph=AvgSpeedKph):
    """
    Estimate travel time based on distance and average speed.
    Returns travel time in minutes.
    """
    return (distance_km / AvgSpeedKph) * 60


def interpolate_position(start_point, end_point, fraction):
    """
    Linearly interpolate between two points based on fraction [0,1].
    """
    lat1, lng1 = start_point
    lat2, lng2 = end_point
    return (lat1 + fraction*(lat2-lat1), lng1 + fraction*(lng2-lng1))


def get_emergency_by_id(emergency_id, state):
    for emergency in state.emergencies_recent + state.emergencies_past:
        if emergency.id == emergency_id:
            return emergency
    return None


def get_hospital_by_id(hospital_id, state):
    for hospital in state.hospitals:
        if hospital.id == hospital_id:
            return hospital
    return None


def apply_decisions(decisions, state, event_log):
    """
    Test and apply the agent's decisions to the SimulationState.
    Records status changes and ETAs in the event log.

    Also returns a dictionary indicating which decisions were applied and which were ignored.
    """
    current_time = state.global_clock
    event_log.write(f"[{current_time}] Applying {len(decisions)} decision(s): {decisions}\n")

    applied_decisions = []
    ignored_decisions = []

    for decision in decisions:
        decision_type = decision[0]

        if decision_type == "SendAmbulanceToEmergency":
            _, ambulance_id, emergency_id = decision
            ambulance = next((a for a in state.ambulances if a.id == ambulance_id), None)
            emergency = get_emergency_by_id(emergency_id, state)

            if emergency is None or emergency.status != EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT:
                event_log.write(f"[{current_time}] Invalid or finished emergency: {ambulance_id}/{emergency_id}\n")
                ignored_decisions.append(decision)
                continue

            if ambulance is None or emergency is None or emergency.resolved:
                event_log.write(f"[{current_time}] Invalid or already resolved ambulance/emergency: {ambulance_id}/{emergency_id}. Ignoring decision.\n")
                ignored_decisions.append(decision)
                continue

            if ambulance.status in [AMBULANCE_STATUS_IDLE, AMBULANCE_STATUS_RELOCATING]:
                ambulance.status = AMBULANCE_STATUS_EN_ROUTE_EMERGENCY
                ambulance.emergency_assigned = emergency_id
                ambulance.hospital_assigned = None
                emergency.status = EMERGENCY_STATUS_WAITING_FOR_AMBULANCE
                emergency.ambulance_assigned_time = current_time
                emergency.assigned_ambulance = ambulance_id
                route = compute_route(ambulance.position, emergency.location)
                ambulance.remaining_route = route
                if route:
                    last_waypoint_time = route[-1][0]
                    ambulance.arrival_estimate = current_time + timedelta(minutes=last_waypoint_time)
                applied_decisions.append(decision)
            else:
                ignored_decisions.append(decision)

        elif decision_type == "SendAmbulanceToHospital":
            _, ambulance_id, hospital_id = decision
            ambulance = next((a for a in state.ambulances if a.id == ambulance_id), None)
            if ambulance is not None:
                if ambulance.status == AMBULANCE_STATUS_AT_EMERGENCY and ambulance.onsite_time_remaining > 0:
                    event_log.write(f"[{current_time}] Ignoring 'SendAmbulanceToHospital' for {ambulance_id}, "
                                    f"on-site time not yet finished.\n")
                    ignored_decisions.append(decision)
                    continue
            hospital = get_hospital_by_id(hospital_id, state)
            emergency = get_emergency_by_id(ambulance.emergency_assigned, state) if ambulance else None

            if ambulance is None or hospital is None or emergency is None:
                ignored_decisions.append(decision)
                continue

            if ambulance.status in [AMBULANCE_STATUS_BROKEN, AMBULANCE_STATUS_EN_ROUTE_EMERGENCY] or not ambulance.contains_patient:
                ignored_decisions.append(decision)
                continue
            if not ambulance.remaining_route:
                ambulance.status = AMBULANCE_STATUS_EN_ROUTE_HOSPITAL
                ambulance.hospital_assigned = hospital_id
                emergency.status = EMERGENCY_STATUS_EN_ROUTE_HOSPITAL
                emergency.assigned_hospital = hospital_id
                route = compute_route(emergency.location, hospital.location)
                ambulance.remaining_route = route
                if route:
                    last_waypoint_time = route[-1][0]
                    ambulance.arrival_estimate = current_time + timedelta(minutes=last_waypoint_time)
                applied_decisions.append(decision)
            else:
                ignored_decisions.append(decision)

        elif decision_type == "RelocateAmbulance":
            _, ambulance_id, dest_lat, dest_lng = decision
            ambulance = next((a for a in state.ambulances if a.id == ambulance_id), None)

            if ambulance is None or ambulance.status in [AMBULANCE_STATUS_BROKEN, AMBULANCE_STATUS_EN_ROUTE_EMERGENCY]:
                ignored_decisions.append(decision)
                continue

            if ambulance.status in [AMBULANCE_STATUS_IDLE, AMBULANCE_STATUS_RELOCATING]:
                ambulance.status = AMBULANCE_STATUS_RELOCATING
                route = compute_route(ambulance.position, (dest_lat, dest_lng))
                ambulance.remaining_route = route
                if route:
                    last_waypoint_time = route[-1][0]
                    ambulance.arrival_estimate = current_time + timedelta(minutes=last_waypoint_time)
                applied_decisions.append(decision)
            else:
                ignored_decisions.append(decision)

        elif decision_type == "ChangeTargetHospital":
            _, ambulance_id, new_hospital_id = decision
            ambulance = next((a for a in state.ambulances if a.id == ambulance_id), None)
            new_hospital = get_hospital_by_id(new_hospital_id, state)

            if ambulance is None or new_hospital is None:
                ignored_decisions.append(decision)
                continue

            if ambulance.status in [AMBULANCE_STATUS_EN_ROUTE_HOSPITAL, AMBULANCE_STATUS_AT_EMERGENCY]:
                ambulance.status = AMBULANCE_STATUS_EN_ROUTE_HOSPITAL
                ambulance.hospital_assigned = new_hospital_id
                route = compute_route(ambulance.position, new_hospital.location)
                ambulance.remaining_route = route
                if route:
                    last_waypoint_time = route[-1][0]
                    ambulance.arrival_estimate = current_time + timedelta(minutes=last_waypoint_time)
                applied_decisions.append(decision)
            else:
                ignored_decisions.append(decision)

        else:
            ignored_decisions.append(decision)
    for ambulance in state.ambulances:
        if ambulance.status == AMBULANCE_STATUS_AT_EMERGENCY and ambulance.onsite_time_remaining ==1 :
            em = get_emergency_by_id(ambulance.emergency_assigned, state)
            if em and not em.resolved and not em.transport_decision_made:
                # We haven't decided yet if the patient needs a hospital
                em.transport_decision_made = True  # Mark we've done it exactly once # as it will remain like 10 minutes at emergencies so the decision is never re-made
                if random.random() < ProbNoTransport:
                    # => No hospital transport is needed
                    em.status = "finished"
                    em.resolved = True
                    em.hospital_arrival_time = current_time  # We can treat “arrival_time” as on‐site resolution
                    ambulance.contains_patient = False

                    # Keep ambulance on‐scene for some random minutes:
                    event_log.write(f"[{current_time}] No hospital transport required for emergency {em.id}.\n"
                        f"[{current_time}] Ambulance {ambulance.id} is now idle.\n"
                    )
                else:
                    event_log.write(f"[{current_time}] Patient at emergency {em.id} requires hospital transport.\n")
        

    return {
        "applied": applied_decisions,
        "ignored": ignored_decisions
    }



def move_ambulances_forward(state, event_log):
    """
    Move all ambulances forward by 1 minute.
    Now uses the minute-by-minute route so that each call to this function
    moves the ambulance exactly to the next position in its route if one is available.
    
    If an ambulance reaches its destination, update its status accordingly.
    """
    current_time = state.global_clock
    next_minute = current_time + timedelta(minutes=1)
    event_log.write(f"[{current_time}] Moving ambulances forward by 1 minute.\n")

    for ambulance in state.ambulances:
        # 1) If the ambulance is simply 'at emergency' with onsite_time_remaining > 0,
        #    decrement that time. (Means no hospital trip needed, or waiting on-scene.)
        if (ambulance.status == AMBULANCE_STATUS_AT_EMERGENCY and 
            ambulance.onsite_time_remaining > 0 and  ambulance.contains_patient):
            ambulance.onsite_time_remaining -= 1
            event_log.write(f"[{next_minute}] Ambulance {ambulance.id} remains on scene, "
                            f"{ambulance.onsite_time_remaining} minute(s) left.\n")

            # If done, go idle
            if ambulance.onsite_time_remaining <= 0:
                ambulance.status = AMBULANCE_STATUS_IDLE
                event_log.write(f"[{next_minute}] Ambulance {ambulance.id} finished on-site work, now idle.\n")

            # Skip route movement, we remain on-scene
            continue
        if ambulance.status == AMBULANCE_STATUS_BEING_CLEANED:
            if ambulance.cleaning_time_remaining > 0:
                ambulance.cleaning_time_remaining -= 1
                event_log.write(f"[{next_minute}] Ambulance {ambulance.id} cleaning... "
                                f"{ambulance.cleaning_time_remaining} minute(s) left.\n")

            if ambulance.cleaning_time_remaining <= 0:
                ambulance.status = AMBULANCE_STATUS_IDLE
                event_log.write(f"[{next_minute}] Ambulance {ambulance.id} finished cleaning, now idle.\n")
            continue
        if ambulance.remaining_route:
            if random.random() < ProbDelay:  
                event_log.write(f"[{next_minute}] Ambulance {ambulance.id} delayed for 1 minute due to random delay.\n")
                continue
            else :
                route = ambulance.remaining_route
                updated_route = [(offset - 1, waypoint) for (offset, waypoint) in route]
                ambulance.remaining_route = updated_route
            arrived_points = [r for r in updated_route if r[0] <= 0]
            if arrived_points:
                new_pos = arrived_points[-1][1]
                ambulance.position = new_pos
                ambulance.remaining_route = [r for r in updated_route if r[0] > 0]
                reached_end = (not ambulance.remaining_route)
            else:
                new_pos = ambulance.position
                reached_end = False

            if ambulance.status == AMBULANCE_STATUS_EN_ROUTE_EMERGENCY:
                desc = f"driving to emergency {ambulance.emergency_assigned}"
            elif ambulance.status == AMBULANCE_STATUS_EN_ROUTE_HOSPITAL:
                desc = f"driving to hospital {ambulance.hospital_assigned}"
            elif ambulance.status == AMBULANCE_STATUS_RELOCATING:
                desc = "relocating"
            else:
                desc = f"status {ambulance.status}"

            event_log.write(f"[{next_minute}] Ambulance {ambulance.id} moved to {ambulance.position} ({desc}).\n")

            # If we have reached the destination
            if reached_end:
                if ambulance.status == AMBULANCE_STATUS_EN_ROUTE_EMERGENCY:
                    emergency = get_emergency_by_id(ambulance.emergency_assigned, state)
                    if emergency:
                        emergency.ambulance_arrival_time = next_minute
                        wait_time = emergency.get_wait_time_for_ambulance(next_minute)
                        event_log.write(f"[{next_minute}] Ambulance {ambulance.id} arrived at emergency {emergency.id}. Wait time: {wait_time:.2f} minutes.\n")
                    old_status = ambulance.status
                    ambulance.status = AMBULANCE_STATUS_AT_EMERGENCY
                    event_log.write(f"[{next_minute}] Ambulance {ambulance.id} status changed from {old_status} to {AMBULANCE_STATUS_AT_EMERGENCY}.\n")
                    ambulance.contains_patient = True
                    ambulance.onsite_time_remaining = random.randint(2, OnsiteTimeMax)
                    event_log.write(
                        f"[{next_minute}] Ambulance {ambulance.id} will stay on-scene for "
                        f"{ambulance.onsite_time_remaining} minute(s).\n"
                    )


                elif ambulance.status == AMBULANCE_STATUS_EN_ROUTE_HOSPITAL:
                    hospital = get_hospital_by_id(ambulance.hospital_assigned, state)
                    emergency = get_emergency_by_id(ambulance.emergency_assigned, state)
                    if hospital and emergency:
                        emergency.status = EMERGENCY_STATUS_FINISHED
                        emergency.resolved = True
                        emergency.hospital_arrival_time = next_minute
                        total_wait_time = emergency.get_total_wait_time(next_minute)
                        event_log.write(f"[{next_minute}] Patient from emergency {emergency.id} arrived at hospital {hospital.id}. Total time: {total_wait_time:.2f} minutes.\n")
                        if hospital.free_beds > 0:
                            hospital.free_beds -= 1
                        # event_log.write(f"[{next_minute}] Ambulance {ambulance.id} delivered patient for emergency {emergency.id} and is now idle.\n")
                        old_status = ambulance.status
                        ambulance.status = AMBULANCE_STATUS_BEING_CLEANED
                        ambulance.cleaning_time_remaining = random.randint(2, CleaningTimeMax)
                        event_log.write(
                            f"[{next_minute}] Ambulance {ambulance.id} delivered patient for emergency {emergency.id}. "
                            f"Status changed from {old_status} to {AMBULANCE_STATUS_BEING_CLEANED}, cleaning for "
                            f"{ambulance.cleaning_time_remaining} minute(s).\n"
                        )               
                        ambulance.contains_patient = False
                        ambulance.emergency_assigned = None
                        ambulance.hospital_assigned = None

                elif ambulance.status == AMBULANCE_STATUS_RELOCATING:
                    event_log.write(f"[{next_minute}] Ambulance {ambulance.id} completed relocation to {new_pos}.\n")
                    old_status = ambulance.status
                    ambulance.status = AMBULANCE_STATUS_IDLE
                    event_log.write(f"[{next_minute}] Ambulance {ambulance.id} status changed from {old_status} to {AMBULANCE_STATUS_IDLE}.\n")
        else:
            # No route to follow; ambulance stays put
            pass


def all_emergencies_resolved(state):
    all_recent_resolved = all(e.resolved for e in state.emergencies_recent)
    all_past_resolved = all(e.resolved for e in state.emergencies_past)
    return all_recent_resolved and all_past_resolved


def compute_wait_time_statistics(state):
    ambulance_wait_times = []
    total_wait_times = []
    for emergency in state.emergencies_past + state.emergencies_recent:
        if emergency.resolved:
            if emergency.ambulance_arrival_time:
                amb_wait = (emergency.ambulance_arrival_time - emergency.timestamp).total_seconds()/60
                ambulance_wait_times.append(amb_wait)
            if emergency.hospital_arrival_time:
                total_wait = (emergency.hospital_arrival_time - emergency.timestamp).total_seconds()/60
                total_wait_times.append(total_wait)
    results = {
        "resolved_emergencies": len(total_wait_times),
        "avg_ambulance_wait_time": statistics.mean(ambulance_wait_times) if ambulance_wait_times else 0,
        "avg_total_wait_time": statistics.mean(total_wait_times) if total_wait_times else 0,
        "median_total_wait_time": statistics.median(total_wait_times) if total_wait_times else 0,
        "min_total_wait_time": min(total_wait_times) if total_wait_times else 0,
        "max_total_wait_time": max(total_wait_times) if total_wait_times else 0,
        "all_wait_times": total_wait_times
    }
    return results
    
def compute_route(start_point, end_point, api=RoutingMethod):
    """assert ROUTING_METHOD in ["Google", "Here", "Openrouteservice", "LinearInterpolation", "Valhalla"] .....
    Compute the route between two lat/long points, returning a list of
    (time_offset_minutes, (lat, lng)) for each full minute of travel.
    """
    start_lat, start_lng = start_point
    end_lat, end_lng = end_point

    if api == 'Here':
        return compute_route_here(start_point, end_point)
    elif api == 'Google':
        return compute_route_google(start_point, end_point)
    elif api == 'Openrouteservice':
        return compute_route_openrouteservice(start_point, end_point)
    elif api == 'Valhalla':
        return compute_route_valhalla(start_point, end_point)
    elif api == 'LinearInterpolation':
        return compute_route_linear_interpolation(start_point, end_point)
    else:
        raise ValueError(f"Unsupported routing API: {api}")

def compute_route_here(start_point, end_point):
    print("compute_route was called")
    here_url = f"https://router.hereapi.com/v8/routes?transportMode=car&origin={start_point[0]},{start_point[1]}&destination={end_point[0]},{end_point[1]}&return=polyline,travelSummary&apikey={HERE_MAPS_API_KEY}"
    try:
        response = requests.get(here_url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching here data: {e}")
        return compute_route_linear_interpolation(start_point, end_point)

    if not data.get("routes"):
        print("here API returned no routes. Falling back to direct distance estimate.")
        return compute_route_linear_interpolation(start_point, end_point)

    encoded_polyline = data["routes"][0]["sections"][0]["polyline"]
    duration = data["routes"][0]["sections"][0]["travelSummary"]["duration"]
    decoded_raw = pc.decode_legacy(encoded_polyline)

    decoded_route = [(lat / 10.0, lng / 10.0) for lat, lng in decoded_raw]

    total_minutes = int(math.ceil(duration / 60.0))
    route = [
        (minute, decoded_route[min(int(minute / max(total_minutes, 1) * len(decoded_route)), len(decoded_route) - 1)])
        for minute in range(total_minutes + 1)
    ]
    return route


def compute_route_linear_interpolation(start_point, end_point):
    distance_km = haversine_distance(start_point, end_point)
    travel_time_mins = int(math.ceil(estimate_travel_time(distance_km)))
    route = []
    for minute in range(travel_time_mins + 1):
        fraction = minute / max(travel_time_mins, 1)
        latlng = interpolate_position(start_point, end_point, fraction)
        route.append((minute, latlng))
    return route


def compute_route_google(start_point, end_point):
    google_url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={start_point[0]},{start_point[1]}&"
        f"destination={end_point[0]},{end_point[1]}&"
        f"mode=driving&key={GOOGLE_MAPS_API_KEY}"
    )
    try:
        response = requests.get(google_url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching Google Maps data: {e}")
        return compute_route_linear_interpolation(start_point, end_point)

    if not data.get("routes"):
        print("Google Maps returned no routes. Falling back.")
        return compute_route_linear_interpolation(start_point, end_point)

    polyline = data["routes"][0]["overview_polyline"]["points"]
    decoded_route = polyline.decode(polyline) 
    duration = data["routes"][0]["legs"][0]["duration"]["value"]  # in seconds

    total_minutes = int(math.ceil(duration / 60.0))
    route = [
        (minute, decoded_route[min(int(minute / max(total_minutes, 1) * len(decoded_route)), len(decoded_route) - 1)])
        for minute in range(total_minutes + 1)
    ]
    return route

def compute_route_openrouteservice(start_point, end_point):
    ors_url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {
        "Authorization": OPENROUTESERVICE_API_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [
            [start_point[1], start_point[0]],
            [end_point[1], end_point[0]]
        ]
    }
    try:
        response = requests.post(ors_url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching OpenRouteService data: {e}")
        return compute_route_linear_interpolation(start_point, end_point)

    if "features" not in data or not data["features"]:
        print("OpenRouteService returned no routes. Falling back.")
        return compute_route_linear_interpolation(start_point, end_point)

    geometry = data["features"][0]["geometry"]["coordinates"]
    decoded_route = [(lat, lng) for lng, lat in geometry]
    duration = data["features"][0]["properties"]["summary"]["duration"]  # in seconds

    total_minutes = int(math.ceil(duration / 60.0))
    route = [
        (minute, decoded_route[min(int(minute / max(total_minutes, 1) * len(decoded_route)), len(decoded_route) - 1)])
        for minute in range(total_minutes + 1)
    ]
    return route


def compute_route_valhalla(start_point, end_point):
    valhalla_url = f"{VALHALLA_API_URL}/route"
    body = {
        "locations": [
            {"lat": start_point[0], "lon": start_point[1]},
            {"lat": end_point[0], "lon": end_point[1]}
        ],
        "costing": "auto",
        "directions_options": {"units": "kilometers"}
    }

    try:
        response = requests.post(valhalla_url, json=body)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching Valhalla data: {e}")
        return compute_route_linear_interpolation(start_point, end_point)

    if "trip" not in data or "legs" not in data["trip"]:
        print("Valhalla returned no routes. Falling back.")
        return compute_route_linear_interpolation(start_point, end_point)

    polyline = data["trip"]["legs"][0]["shape"]
    decoded_route = decode_valhalla(polyline) 
    duration = data["trip"]["summary"]["time"] 

    total_minutes = int(math.ceil(duration / 60.0))
    route = [
        (minute, decoded_route[min(int(minute / max(total_minutes, 1) * len(decoded_route)), len(decoded_route) - 1)])
        for minute in range(total_minutes + 1)
    ]
    return route

def decode_valhalla(encoded_str):
    decoded = []
    prev_lat, prev_lon = 0, 0
    index = 0
    while index < len(encoded_str):
        result = 1
        shift = 0
        b = 0x20
        while b >= 0x20:
            b = ord(encoded_str[index]) - 63
            index += 1
            result += (b & 0x1f) << shift
            shift += 5
        dlat = ~(result >> 1) if result & 1 else (result >> 1)
        prev_lat += dlat

        result = 1
        shift = 0
        b = 0x20
        while b >= 0x20:
            b = ord(encoded_str[index]) - 63
            index += 1
            result += (b & 0x1f) << shift
            shift += 5
        dlng = ~(result >> 1) if result & 1 else (result >> 1)
        prev_lon += dlng

        decoded.append((prev_lat * 1e-6, prev_lon * 1e-6))
    return decoded


def start_experiment(map_path, scenario_path, agent_path, experiment_index=0):
    state = read_map(map_path)
    scenario_emergencies = load_scenario(scenario_path)
    if not scenario_emergencies:
        return 0
    pending_emergencies = list(scenario_emergencies)
    state.global_clock = pending_emergencies[0].timestamp
    state.emergencies_recent = []
    state.emergencies_past = []

    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    agent_class = None
    for name, obj in inspect.getmembers(agent_module, inspect.isclass):
        if issubclass(obj, AgentBase) and obj is not AgentBase:
            agent_class = obj
            break
    if agent_class is None:
        raise ImportError(f"No AgentBase subclass found in {agent_path}")
    agent = agent_class(state)
    agent.compute_route = compute_route

    os.makedirs("output", exist_ok=True)
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    agent_name = os.path.splitext(os.path.basename(agent_path))[0]
    log_filename = f"output/{scenario_name}__{agent_name}__run{experiment_index}.txt"
    detailed_log_filename = f"output/{scenario_name}__{agent_name}__run{experiment_index}_detailed.txt"

    with open(log_filename, 'w') as event_log, open(detailed_log_filename, 'w') as detailed_log:
        event_log.write("=== SIMULATION EVENT LOG ===\n\n")
        event_log.write("MAP DATA:\n")
        map_dump = {
            "ambulances": [
                {"id": a.id, "position": a.position, "status": a.status}
                for a in state.ambulances
            ],
            "hospitals": [
                {"id": h.id, "location": h.location,
                 "beds": h.num_beds, "free_beds": h.free_beds}
                for h in state.hospitals
            ],
            "rescueStations": [
                {"id": rs.id, "location": rs.location,
                 "available_ambulances": rs.available_ambulances}
                for rs in state.rescue_stations
            ]
        }
        event_log.write(json.dumps(map_dump, indent=2))
        event_log.write("\n\n")

        # Detailed log start
        detailed_log.write("=== DETAILED SIMULATION LOG ===\n\n")
        detailed_log.write("BEGINNING OF EXPERIMENT STATE:\n")
        detailed_log.write(str(state) + "\n\n")

        last_event_time = pending_emergencies[-1].timestamp
        current_time = state.global_clock

        while True:
            # Move 'recent' emergencies to 'past'
            state.emergencies_past.extend(state.emergencies_recent)
            state.emergencies_recent = []

            next_minute = current_time + timedelta(minutes=1)
            # Activate new emergencies
            while pending_emergencies and pending_emergencies[0].timestamp < next_minute:
                new_em = pending_emergencies.pop(0)
                event_log.write(f"[{current_time}] New emergency activated: {new_em}\n")
                new_em.status = EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT
                state.emergencies_recent.append(new_em)

            #decisions = agent.get_actions(state)
            decisions = agent.get_actions(copy.deepcopy(state))

            # <<< CHANGE START: Capture which decisions are applied/ignored >>>
            applied_ignored = apply_decisions(decisions, state, event_log)
            applied_decisions = applied_ignored["applied"]
            ignored_decisions = applied_ignored["ignored"]
            # <<< CHANGE END >>>

            move_ambulances_forward(state, event_log)

            # <<< CHANGE START: Log applied and ignored decisions >>>
            detailed_log.write(f"END OF STEP at simulation time {current_time}\n")
            detailed_log.write("Current Simulation State:\n")
            detailed_log.write(str(state) + "\n")
            detailed_log.write(f"Decisions made this step: {decisions}\n")
            detailed_log.write(f"Decisions applied this step: {applied_decisions}\n")
            detailed_log.write(f"Decisions ignored this step: {ignored_decisions}\n\n")
            # <<< CHANGE END

            if next_minute >= last_event_time + timedelta(minutes=120) and not all_emergencies_resolved(state):
                event_log.write(f"[{next_minute}] Force quitting simulation: 120 minutes past last emergency call with unresolved emergencies.\n")
                return 10000

            if all_emergencies_resolved(state) and next_minute > last_event_time:
                event_log.write(f"[{next_minute}] All emergencies resolved. Simulation ends.\n")
                break

            current_time = next_minute
            state.global_clock = current_time

        # Final stats
        wait_stats = compute_wait_time_statistics(state)
        event_log.write("\n=== FINAL WAIT TIME STATISTICS ===\n")
        event_log.write(f"Resolved emergencies: {wait_stats['resolved_emergencies']}\n")
        event_log.write(f"Average time from call to ambulance arrival: {wait_stats['avg_ambulance_wait_time']:.2f} minutes\n")
        event_log.write(f"Average time from call to hospital arrival: {wait_stats['avg_total_wait_time']:.2f} minutes\n")
        return wait_stats['avg_total_wait_time']


def infer_scenario_pattern(map_path):
    map_basename = os.path.basename(map_path)
    map_name = re.sub(r'^map_|\.json$', '', map_basename)
    scenario_pattern = f'input/calls_{map_name}_*.json'
    return scenario_pattern

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="input/map_ReinlandPfalz.json", help="Path to map JSON")
    parser.add_argument("--scenario", help="Path to scenario JSON or pattern (inferred from map if not provided)")
    parser.add_argument("--agent", default="agents/agent_SpatiotemporalOptimizer.py", help="Path to agent file pattern")
    parser.add_argument("--outfolder", default="output", help="Output folder for logs/results")
    parser.add_argument("--num-exp", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--result-filename", default="final_results.csv", help="CSV file to save results")
    args = parser.parse_args()

    map_path = args.map
    scenario_pattern = args.scenario if args.scenario else infer_scenario_pattern(map_path)
    agent_pattern = args.agent
    num_exp = args.num_exp
    result_filename = args.result_filename

    scenario_files = glob.glob(scenario_pattern)
    agent_files = glob.glob(agent_pattern)
    results = []

    for scenario_path in scenario_files:
        for agent_path in agent_files:
            for i in range(num_exp):
                score = start_experiment(map_path, scenario_path, agent_path, experiment_index=i)
                results.append({
                    'scenario': os.path.basename(scenario_path),
                    'agent': os.path.basename(agent_path),
                    'experiment_index': i,
                    'score': score
                })

    os.makedirs(args.outfolder, exist_ok=True)
    result_path = os.path.join(args.outfolder, result_filename)

    with open(result_path, mode='w', newline='') as csvfile:
        fieldnames = ['scenario', 'agent', 'experiment_index', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results saved to {result_path}")
    print(f"Event logs created in the '{args.outfolder}/' folder.")

if __name__ == "__main__":
    main()
