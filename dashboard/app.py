"""Streamlit dashboard application.

This dashboard allows users to monitor live camera feeds, view detection
results, and inspect logged events. Running this script with Streamlit
will start a local web server accessible in your browser. The dashboard
interacts with the underlying modules (camera adapters, detection,
tracking, recognition, rule engine, storage) to display real‑time data.

Note: This is a minimal implementation to illustrate the structure. You
will need to extend it to handle live video streaming and integrate the
full pipeline.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import yaml
import pandas as pd


def load_config() -> dict:
    """Load the dashboard configuration file.

    Returns
    -------
    config : dict
        The parsed YAML configuration dictionary.
    """
    # Determine the path to the default configuration relative to this file
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def load_events(log_dir: str) -> List[Dict[str, Any]]:
    """Read logged events from the JSON lines log file.

    Parameters
    ----------
    log_dir : str
        Directory where events.jsonl resides.

    Returns
    -------
    events : list of dict
        A list of event dictionaries parsed from the log file.
    """
    events: List[Dict[str, Any]] = []
    log_file = Path(log_dir) / "events.jsonl"
    if not log_file.exists():
        return events
    with log_file.open("r") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def update_watchlist(new_plate: str, config_path: Path) -> None:
    """Append a new license plate to the watchlist in the configuration.

    Parameters
    ----------
    new_plate : str
        The plate number to add.
    config_path : Path
        Path to the YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    rules = config.get("routing_rules", [])
    updated = False
    for rule in rules:
        if rule.get("type") == "license_plate_watchlist":
            watchlist = rule.get("watchlist", [])
            if new_plate not in watchlist:
                watchlist.append(new_plate)
                rule["watchlist"] = watchlist
                updated = True
                break
    if updated:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)



def main() -> None:
    """Main entry point for the Streamlit dashboard.

    Displays the current watchlist, allows adding new plates, shows logged
    events, and provides a simple interface for monitoring the system.
    """
    st.set_page_config(page_title="Road Security Dashboard", layout="wide")
    st.title("Road Security Dashboard")

    # Load configuration and events
    config = load_config()
    storage_cfg = config.get("storage", {})
    log_dir = storage_cfg.get("log_dir", "logs")
    events = load_events(log_dir)

    # Sidebar: watchlist management
    st.sidebar.header("Watchlist")
    # Extract current watchlist from the first matching rule
    watchlist: List[str] = []
    rules = config.get("routing_rules", [])
    for rule in rules:
        if rule.get("type") == "license_plate_watchlist":
            wl = rule.get("watchlist", [])
            if isinstance(wl, list):
                watchlist = wl
            break
    st.sidebar.write(watchlist or "No plates in watchlist.")
    new_plate = st.sidebar.text_input("Add plate to watchlist")
    if st.sidebar.button("Add"):
        if new_plate:
            # Update YAML config with new plate
            config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
            update_watchlist(new_plate.strip(), config_path)
            st.sidebar.success(f"Added {new_plate} to watchlist. Restart the pipeline to apply.")
        else:
            st.sidebar.warning("Please enter a plate number.")

    # Main area: Live feed placeholder
    st.header("Live Feed")
    st.write(
        "The live camera feed will appear here when the pipeline is running. "
        "Use the demo script to start processing a sample video."
    )

    # Detection & Recognition results placeholder
    st.header("Detection & Recognition Results")
    st.write(
        "Detection and recognition results will be displayed here in real-time once "
        "integrated."
    )

    # Event logs table
    st.header("Event Logs")
    if events:
        # Flatten rule info for better display
        flattened: List[Dict[str, Any]] = []
        for e in events:
            row = {
                "timestamp": e.get("timestamp"),
                "rule_type": e.get("rule", {}).get("type"),
                "license_plate": e.get("context", {}).get("license_plate"),
                "confidence": e.get("context", {}).get("license_plate_confidence"),
                "track_id": e.get("context", {}).get("track_id"),
            }
            flattened.append(row)
        df = pd.DataFrame(flattened)
        st.dataframe(df)
    else:
        st.write("No events logged yet.")

    # Refresh button to reload events
    if st.button("Refresh Logs"):
        st.experimental_rerun()


if __name__ == "__main__":
    main()
