"""spatiotemporal_optimizer.py

Policy agent that combines real‑time dispatch logic with *every‑three‑hour*
proactive relocations driven by the :pyclass:`~prediction_model.
demand_prediction.LightGBMRLPEventPrediction` demand model.

The proactive step works on a **20 km hex‑grid** approximation:

* Each base (hospital or rescue station) is mapped to an integer grid cell of
  ≈20 km × 20 km. If multiple bases fall into the same cell, that cell is
  evaluated only once – avoiding duplicated work.
* For the three‑hour prediction horizon starting at the next full 3‑hour slot
  (00:00, 03:00, 06:00 …) the agent forecasts the number of emergency calls at
  every grid cell.
* If the predicted calls exceed the currently available ambulances inside the
  **20 km neighbourhood** (all bases whose centres are ≤ 20 km away), the
  deficit is evenly distributed over these neighbouring bases. Idle ambulances
  outside the neighbourhood are then reassigned to cover the deficit.

This strategy ensures that shortages are detected once per grid‑cell and that
ambulances are allocated **proportionally** across all bases inside the
20 km cluster experiencing the shortfall.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# Optional LightGBM dependency handled in the model module itself
from prediction_model.demand_prediction import LightGBMRLPEventPrediction

# ---------------------------------------------------------------------------
# Framework imports supplied by the simulator (type stubs only for linters)
# ---------------------------------------------------------------------------
from agents.base_class import AgentBase  # type: ignore

# -----------------------------
# Simulator status constants
# -----------------------------
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

# ---------------------------------------------------------------------------
# Helper constants – 20 km grid approximation
# ---------------------------------------------------------------------------
_GRID_KM: float = 20.0  # neighbourhood radius & grid size
_GRID_DEG: float = _GRID_KM / 111.0  # ≈ degrees latitude per 20 km
_TWO_PI: float = 2.0 * math.pi


class SpatioTemporalOptimizer(AgentBase):
    """Ambulance dispatch & relocation policy using spatio‑temporal forecasts."""

    #: seconds between two proactive relocation waves (3 h)
    _RELOCATION_PERIOD_S: int = 3 * 3600

    def __init__(self, state):
        super().__init__(state)
        self.predictor = LightGBMRLPEventPrediction(r"prediction_model/emergency_call_model.pkl")
        self._last_relocation_ts: float | None = None  # unix timestamp when last relocation ran

    # ------------------------------------------------------------------
    # Main entry point (called every simulation tick)
    # ------------------------------------------------------------------
    def get_actions(self, state):  # noqa: D401 – required simulator signature
        now: datetime = getattr(state, "current_time", datetime.now())
        decisions: List[Tuple] = []

        # ------------------------------------------------------------------
        # 1) Reactive – send nearest idle ambulance to every unassigned call
        # ------------------------------------------------------------------
        waiting = [e for e in (state.emergencies_past + state.emergencies_recent)
                   if e.status == EMERGENCY_STATUS_WAITING_FOR_AMBULANCE_ASSIGNMENT]
        idle_pool = [a for a in state.ambulances
                     if a.emergency_assigned is None and a.status in {
                         AMBULANCE_STATUS_IDLE, AMBULANCE_STATUS_RELOCATING}]

        for em in waiting:
            if not idle_pool:
                break
            amb = min(idle_pool, key=lambda a: self._travel_time(a.position, em.location))
            decisions.append(("SendAmbulanceToEmergency", amb.id, em.id))
            idle_pool.remove(amb)

        # ------------------------------------------------------------------
        # 2) Post‑pickup – choose hospital once scene is cleared
        # ------------------------------------------------------------------
        for amb in state.ambulances:
            if (amb.emergency_assigned is not None and amb.hospital_assigned is None and
                    amb.contains_patient and not amb.remaining_route):
                assoc_em = next((e for e in (state.emergencies_past + state.emergencies_recent)
                                  if e.id == amb.emergency_assigned), None)
                if not state.hospitals:
                    continue
                prefs = [h for h in state.hospitals
                         if assoc_em and h.id in assoc_em.hospitals and h.free_beds > 0]
                dest = min(prefs or state.hospitals,
                           key=lambda h: self._travel_time(amb.position, h.location))
                decisions.append(("SendAmbulanceToHospital", amb.id, dest.id))

        # ------------------------------------------------------------------
        # 3) Redirect en‑route if hospital becomes full
        # ------------------------------------------------------------------
        for amb in state.ambulances:
            if amb.status == AMBULANCE_STATUS_EN_ROUTE_HOSPITAL:
                curr = next((h for h in state.hospitals if h.id == amb.hospital_assigned), None)
                if curr and curr.free_beds == 0:
                    alt = [h for h in state.hospitals if h.free_beds > 0]
                    if alt:
                        dest = min(alt, key=lambda h: self._travel_time(amb.position, h.location))
                        decisions.append(("ChangeTargetHospital", amb.id, dest.id))

        # ------------------------------------------------------------------
        # 4) Proactive – every 3 h relocation wave
        # ------------------------------------------------------------------
        if self._should_relocate(now):
            self._last_relocation_ts = now.timestamp()
            decisions.extend(self._relocate_idle_ambulances(state, now))

        return decisions

    # ------------------------------------------------------------------
    # Helper – determine if we are on a relocation tick (once / 3 h)
    # ------------------------------------------------------------------
    def _should_relocate(self, now: datetime) -> bool:
        if self._last_relocation_ts is None:
            return True  # first ever call
        return (now.timestamp() - self._last_relocation_ts) >= self._RELOCATION_PERIOD_S

    # ------------------------------------------------------------------
    # Core proactive relocation logic (20 km neighbourhood)
    # ------------------------------------------------------------------
    def _relocate_idle_ambulances(self, state, now: datetime):
        """Return relocation decisions for currently *idle* ambulances."""
        decisions: List[Tuple] = []

        bases = state.hospitals + state.rescue_stations
        if not bases:
            return decisions

        # ------------------------------------------------------------------
        # 1) Forecast demand per grid‑cell (unique lat/lon 20 km bins)
        # ------------------------------------------------------------------
        grid_to_bases: Dict[Tuple[int, int], List] = {}
        for b in bases:
            key = self._grid_key(b.location[0], b.location[1])
            grid_to_bases.setdefault(key, []).append(b)

        # Predict demand at the *grid centroid* for the next three‑hour window
        # We call the model once per grid to avoid duplicate work.
        grid_demand: Dict[Tuple[int, int], int] = {}
        lookahead_time = now + timedelta(hours=3)
        for key, group in grid_to_bases.items():
            lat_c, lon_c = self._grid_center(*key)
            y_hat = self.predictor.inference((lat_c, lon_c, lookahead_time))[0]
            grid_demand[key] = max(1, int(round(y_hat)))  # at least 1 expected call

        # ------------------------------------------------------------------
        # 2) Identify idle ambulances and map them to nearest base & grid
        # ------------------------------------------------------------------
        idle = [a for a in state.ambulances
                if a.status == AMBULANCE_STATUS_IDLE and a.emergency_assigned is None]
        if not idle:
            return decisions

        def nearest_base(amb):
            return min(bases, key=lambda b: self._travel_time(amb.position, b.location))

        base_to_idle: Dict[int, List] = {b.id: [] for b in bases}
        for amb in idle:
            b = nearest_base(amb)
            base_to_idle[b.id].append(amb)

        # ------------------------------------------------------------------
        # 3) Compute deficit per **20 km neighbourhood**
        # ------------------------------------------------------------------
        def neighbourhood(key):
            lat_c, lon_c = self._grid_center(*key)
            return [k for k in grid_to_bases.keys()
                    if self._haversine_km(lat_c, lon_c, *self._grid_center(*k)) <= _GRID_KM]

        # Prepare structures to track surpluses and deficits
        neighbourhood_deficit: Dict[Tuple[int, int], int] = {}
        neighbourhood_bases: Dict[Tuple[int, int], List] = {}

        for key, bases_in_cell in grid_to_bases.items():
            # Count ambulances currently at bases in *this* neighbourhood
            nbh_keys = neighbourhood(key)
            bases_nbh: List = sum((grid_to_bases[k] for k in nbh_keys), [])
            neighbourhood_bases[key] = bases_nbh
            ambulances_available = sum(len(base_to_idle[b.id]) for b in bases_nbh)
            calls_expected = sum(grid_demand[k] for k in nbh_keys)
            deficit = max(0, calls_expected - ambulances_available)
            neighbourhood_deficit[key] = deficit

        # ------------------------------------------------------------------
        # 4) Select ambulances from *outside* the deficit neighbourhoods
        # ------------------------------------------------------------------
        # Build a list of surplus idle ambulances not needed in their own neighbourhoods
        surplus_ambs: List = []
        for key, bases_in_cell in grid_to_bases.items():
            nbh_keys = neighbourhood(key)
            bases_nbh = sum((grid_to_bases[k] for k in nbh_keys), [])
            ambulances_available = sum(len(base_to_idle[b.id]) for b in bases_nbh)
            calls_expected = sum(grid_demand[k] for k in nbh_keys)
            surplus_here = max(0, ambulances_available - calls_expected)
            if surplus_here <= 0:
                continue
            # harvest surplus ambulances – but keep one spare per base if possible
            for b in bases_nbh:
                keep = 1  # leave at least one idle ambulance per base
                movable = max(0, len(base_to_idle[b.id]) - keep)
                surplus_ambs.extend(base_to_idle[b.id][:movable])

        if not surplus_ambs:
            return decisions  # nothing to relocate

        # ------------------------------------------------------------------
        # 5) Fill deficits – distribute *evenly* across bases in neighbourhood
        # ------------------------------------------------------------------
        for key, deficit in neighbourhood_deficit.items():
            if deficit == 0:
                continue
            bases_nbh = neighbourhood_bases[key]
            if not bases_nbh:
                continue
            # Round‑robin over bases so that extras are spread as evenly as possible
            base_cycle = list(bases_nbh)
            idx = 0
            while deficit > 0 and surplus_ambs:
                amb = surplus_ambs.pop(0)
                tgt_base = base_cycle[idx % len(base_cycle)]
                decisions.append(("RelocateAmbulance", amb.id,
                                   tgt_base.location[0], tgt_base.location[1]))
                idx += 1
                deficit -= 1

        return decisions

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_key(lat: float, lon: float) -> Tuple[int, int]:
        """Return integer grid indices for ~20 km × 20 km bins."""
        return int(lat / _GRID_DEG), int(lon / _GRID_DEG)

    @staticmethod
    def _grid_center(ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid indices back to lat/lon of the cell centre."""
        return (ix + 0.5) * _GRID_DEG, (iy + 0.5) * _GRID_DEG

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great‑circle distance on Earth in km."""
        lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2 * math.asin(math.sqrt(a))

    @staticmethod
    def _travel_time(src, dst) -> float:  # noqa: D401 – simple helper
        """Fallback travel time (km straight‑line / 60 km h⁻¹)."""
        dx = src[0] - dst[0]
        dy = src[1] - dst[1]
        dist_km = 111.0 * math.hypot(dx, dy)
        return (dist_km / 60.0) * 60.0  # minutes
