import re
import sys
import json
import ast
import glob

# ----------------------------
# Updated Mapping for Statuses
# ----------------------------
# Make sure these exactly match what appears in the logs!
AMBULANCE_STATUS_COLORS = {
    "idle": "#00FF00",                          # green
    "broken": "#808080",                        # red
    "on the way to hospital": "#ff6666",        # must match logs exactly
    "on the way to emergency": "#000099",       # must match logs exactly
    "relocating": "#FFFF00",                    # yellow
    "at emergency": "#00FFFF",                  # cyan
}

EMERGENCY_STATUS_COLORS = {
    "waiting_for_assignment": "#FF6347",  # tomato
    "waiting_for_ambulance": "#FF0000",   # red
    "en_route_hospital": "#0000FF",       # blue
    "finished": "#00FF00",                # green
}

# Distinct colors for hospitals and rescue stations
HOSPITAL_COLOR = "#800080"       # purple
RESCUE_STATION_COLOR = "#00CED1" # dark turquoise

# For remaining routes, we will determine color based on status
# but we keep this constant in case we need a default
ROUTE_LINE_COLOR = "#0000FF" # unused

def extract_coordinates(obj_str, key):
    """
    Extract coordinates from a substring such as pos=(lat, lon) or loc=(lat, lon).
    """
    pattern = key + r"=\(([^)]+)\)"
    m = re.search(pattern, obj_str)
    if m:
        coords = m.group(1).split(',')
        try:
            return [float(coords[0].strip()), float(coords[1].strip())]
        except Exception:
            return None
    return None

def extract_remaining_route(obj_str):
    """
    Extract the remaining_route list from an object string.
    The format is something like:
      remaining_route=[(1, (lat1, lon1)), (2, (lat2, lon2)), ...]
    We extract the coordinate pairs.
    """
    pattern = r"remaining_route=\["
    start_match = re.search(pattern, obj_str)
    if not start_match:
        return []
    start_index = start_match.end()
    count = 1
    i = start_index
    while i < len(obj_str) and count > 0:
        if obj_str[i] == '[':
            count += 1
        elif obj_str[i] == ']':
            count -= 1
        i += 1
    route_str = obj_str[start_index:i-1]
    route = []
    # Find all tuples of the form: (number, (lat, lon))
    tuple_pattern = r"\(\s*\d+\s*,\s*\(([^)]+)\)\s*\)"
    for match in re.finditer(tuple_pattern, route_str):
        coords = match.group(1).split(',')
        if len(coords) >= 2:
            try:
                lat = float(coords[0].strip())
                lon = float(coords[1].strip())
                route.append([lat, lon])
            except Exception:
                continue
    return route

def parse_object_list(state_str, obj_key):
    """
    Given the simulation state string and an object key (e.g., "Ambulances"),
    extract a list of raw object strings.
    Each object is enclosed in < ... > inside that list.
    """
    pattern = obj_key + r"=\["
    start_match = re.search(pattern, state_str)
    if not start_match:
        return []
    start_index = start_match.end()
    count = 1
    i = start_index
    while i < len(state_str) and count > 0:
        if state_str[i] == '[':
            count += 1
        elif state_str[i] == ']':
            count -= 1
        i += 1
    list_str = state_str[start_index:i-1]
    # Now extract objects enclosed in < ... >.
    objects = re.findall(r"<([^>]+)>", list_str)
    return objects

def parse_simulation_state(state_str):
    """
    Given a block of text representing one simulation state (the content inside SimulationState(...))
    extract a dictionary with keys:
       time, Ambulances, RescueStations, Hospitals, EmergenciesRecent, EmergenciesPast.
    The lists contain raw object strings.
    """
    state = {}
    # Extract the simulation time (assumed format: time=YYYY-MM-DD HH:MM:SS,)
    time_match = re.search(r"time=([0-9\-: ]+),", state_str)
    if time_match:
        state["time"] = time_match.group(1).strip()
    else:
        state["time"] = "unknown"

    state["Ambulances"] = parse_object_list(state_str, "Ambulances")
    state["RescueStations"] = parse_object_list(state_str, "RescueStations")
    state["Hospitals"] = parse_object_list(state_str, "Hospitals")
    state["EmergenciesRecent"] = parse_object_list(state_str, "EmergenciesRecent")
    state["EmergenciesPast"] = parse_object_list(state_str, "EmergenciesPast")
    return state

def build_marker(obj_str, obj_type):
    """
    Build a marker dictionary from an object raw string.
    The marker will have:
       - position: extracted from pos=... (if ambulance) or loc=... (others)
       - icon: chosen based on object type
       - color: for ambulances/emergencies based on status; for others fixed colors
       - description: the raw string text.
    """
    marker = {}
    if obj_type == "Ambulance":
        pos = extract_coordinates(obj_str, "pos")
        marker["position"] = pos if pos else [0, 0]
        marker["icon"] = "ambulance"
        status_match = re.search(r"status=([^,>]+)", obj_str)
        status = status_match.group(1).strip() if status_match else ""
        marker["color"] = AMBULANCE_STATUS_COLORS.get(status, "#000000")

    elif obj_type == "RescueStation":
        pos = extract_coordinates(obj_str, "loc")
        marker["position"] = pos if pos else [0, 0]
        marker["icon"] = "life-ring"
        marker["color"] = RESCUE_STATION_COLOR

    elif obj_type == "Hospital":
        pos = extract_coordinates(obj_str, "loc")
        marker["position"] = pos if pos else [0, 0]
        marker["icon"] = "hospital"
        marker["color"] = HOSPITAL_COLOR

    elif obj_type == "Emergency":
        pos = extract_coordinates(obj_str, "loc")
        marker["position"] = pos if pos else [0, 0]
        marker["icon"] = "exclamation-triangle"
        # Extract the emergency status:
        status_match = re.search(r"status=([^,>]+)", obj_str)
        status = status_match.group(1).strip() if status_match else ""
        marker["color"] = EMERGENCY_STATUS_COLORS.get(status, "#FF00FF")

    else:
        marker["position"] = [0, 0]
        marker["icon"] = "info"
        marker["color"] = "#FFFFFF"

    marker["description"] = obj_str.strip()
    return marker

def build_route_lines(obj_str):
    """
    For an ambulance object string, if there is a non-empty remaining_route,
    build line segments from successive points with colors
    depending on the ambulance status:
      relocating => yellow (#FFFF00)
      on the way to emergency => blue (#0000FF)
      on the way to hospital => red (#FF0000)
      otherwise => black (#000000)
    """
    # First, parse the status
    status_match = re.search(r"status=([^,>]+)", obj_str)
    status = status_match.group(1).strip() if status_match else ""

    # Determine the line color
    if status == "relocating":
        line_color = "#E99B48"  # yellow
    elif status == "on the way to emergency":
        line_color = "#2B669E"  # blue
    elif status == "on the way to hospital":
        line_color = "#FF4500"  # orange red
    else:
        line_color = "#000000"  # black (default if not one of the above)

    route = extract_remaining_route(obj_str)
    lines = []
    if len(route) >= 2:
        for i in range(len(route) - 1):
            line = {
                "start": route[i],
                "end": route[i + 1],
                "color": line_color
            }
            lines.append(line)
    return lines

def process_frame(state_block, decisions_str):
    """
    Given the content of a simulation state block (inside SimulationState(...))
    and an optional decisions string, build a frame dictionary with keys:
    description, tags, and lines.
    """
    state = parse_simulation_state(state_block)

    # Attempt to parse decisions from a Python literal if present
    if decisions_str:
        try:
            decisions = ast.literal_eval(decisions_str)
        except Exception:
            decisions = []
    else:
        decisions = []

    # Build a description using simulation time, recent emergencies, and decisions.
    recent_emergencies = state.get("EmergenciesRecent", [])
    emergencies_text = "; ".join([e.strip() for e in recent_emergencies]) if recent_emergencies else "None"
    decisions_text = "; ".join([str(dec) for dec in decisions]) if decisions else "None"
    description = f"Time: {state.get('time')}; Recent Emergencies: {emergencies_text}; Decisions: {decisions_text}"

    tags = []
    lines = []

    # Include all ambulances
    for amb in state.get("Ambulances", []):
        marker = build_marker(amb, "Ambulance")
        tags.append(marker)
        amb_lines = build_route_lines(amb)
        lines.extend(amb_lines)

    # Add markers for rescue stations
    for rs in state.get("RescueStations", []):
        marker = build_marker(rs, "RescueStation")
        tags.append(marker)

    # Add markers for hospitals
    for hosp in state.get("Hospitals", []):
        marker = build_marker(hosp, "Hospital")
        tags.append(marker)

    # Add markers for emergencies (recent and past)
    for emerg in state.get("EmergenciesRecent", []):
        marker = build_marker(emerg, "Emergency")
        tags.append(marker)
    for emerg in state.get("EmergenciesPast", []):
        marker = build_marker(emerg, "Emergency")
        tags.append(marker)

    return {
        "description": description,
        "tags": tags,
        "lines": lines
    }

def extract_simulation_state_blocks(text):
    """
    Extract all simulation state blocks from the log text.
    Each block starts at "SimulationState(" and ends at the matching closing parenthesis.
    Immediately following the state block, if a line "Decisions made this step:" exists,
    that line (up to newline) is captured.
    Returns a list of (state_block, decisions_str) tuples.
    """
    blocks = []
    pos = 0
    while True:
        idx = text.find("SimulationState(", pos)
        if idx == -1:
            break
        start_index = idx + len("SimulationState(")
        count = 1
        i = start_index
        while i < len(text) and count > 0:
            if text[i] == '(':
                count += 1
            elif text[i] == ')':
                count -= 1
            i += 1
        state_block = text[start_index:i-1].strip()

        # Look for "Decisions made this step:" after the end of SimulationState(...)
        decisions_str = ""
        dec_idx = text.find("Decisions made this step:", i)
        next_state = text.find("SimulationState(", i)
        next_end = text.find("END OF STEP", i)
        if dec_idx != -1 and (next_state == -1 or dec_idx < next_state) and (next_end == -1 or dec_idx < next_end):
            end_line = text.find("\n", dec_idx)
            if end_line == -1:
                end_line = len(text)
            decisions_str = text[dec_idx + len("Decisions made this step:"): end_line].strip()

        blocks.append((state_block, decisions_str))
        pos = i
    return blocks

def main():
    # Process all files matching the pattern "output/calls_simple_*_run*_detailed.txt"
    input_pattern = "output/call_*_run*_detailed.txt"
    input_files = glob.glob(input_pattern)
    if not input_files:
        print("No files matching pattern found:", input_pattern)
        sys.exit(1)

    for input_file in input_files:
        output_file = input_file.replace("_detailed.txt", "_visualization.json")
        print(f"Processing {input_file} -> {output_file}")
        with open(input_file, "r") as f:
            log_text = f.read()

        blocks = extract_simulation_state_blocks(log_text)
        frames = []
        for state_block, decisions_str in blocks:
            frame = process_frame(state_block, decisions_str)
            if frame:
                frames.append(frame)

        output_dict = {"frames": frames}
        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Wrote {len(frames)} frames to {output_file}")

if __name__ == "__main__":
    main()
