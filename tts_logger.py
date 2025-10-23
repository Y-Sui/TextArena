"""
Simple Trajectory Logger for Test-Time Scaling
==============================================

Clean and simple logging of LLM inputs/outputs for each TTS method.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import textarena as ta


class TrajectoryLogger:
    """Simple logger that tracks LLM inputs and outputs for each step"""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.trajectory = []
        self.current_turn = None

    def start_turn(self, observation: str):
        """Start logging a new turn"""
        self.current_turn = {
            "observation": observation,
            "steps": []
        }

    def log_llm_call(self, input_text: str, output_text: str, step_type: str = "generation"):
        """Log a single LLM input/output"""
        if self.current_turn is not None:
            self.current_turn["steps"].append({
                "step_type": step_type,
                "input": input_text,
                "output": output_text
            })

    def end_turn(self, final_action: str):
        """End the current turn and save it"""
        if self.current_turn is not None:
            self.current_turn["final_action"] = final_action
            self.trajectory.append(self.current_turn)
            self.current_turn = None

    def save(self, output_dir: str = "trajectories", filename: str = None):
        """Save trajectory to JSON file"""
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.method_name}_{timestamp}.json"

        filepath = os.path.join(output_dir, filename)

        data = {
            "method": self.method_name,
            "timestamp": datetime.now().isoformat(),
            "turns": self.trajectory
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Trajectory saved to: {filepath}")
        return filepath