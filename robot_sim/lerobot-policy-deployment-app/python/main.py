"""
FILE: main.py

DESCRIPTION:
  Loads a trained NumPy MLP policy for robot control and provides action prediction.

BRIEF:
  Provides functions to predict robot actions based on state observations.
  Allows a microcontroller to call these functions via Bridge to drive servos.

AUTHOR: Kevin Thomas
CREATION DATE: February 28, 2026
UPDATE DATE: February 28, 2026
"""

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import numpy as np

# Status and device configuration
# Map status strings for LED matrix display
STATUS_MAP = {"running": "running", "success": "success", "timeout": "timeout"}

# Load pre-trained MLP policy weights
# Initialize weights from .npz archive and hold in memory for fast inference
_data = np.load("/app/python/best_policy.npz")
W1, b1 = _data["W1"], _data["b1"]
W2, b2 = _data["W2"], _data["b2"]
W3, b3 = _data["W3"], _data["b3"]


# Private helper functions for predict_action (in call order)
def _prepare_raw_state(
    agent_x: float,
    agent_y: float,
    block_x: float,
    block_y: float,
    target_x: float,
    target_y: float,
) -> list:
    """
    Prepare raw state features in model input order.

    PARAMETERS:
      agent_x (float): Agent x position.
      agent_y (float): Agent y position.
      block_x (float): Block x position.
      block_y (float): Block y position.
      target_x (float): Target x position.
      target_y (float): Target y position.

    RETURNS:
      list: Features in order [agent_x, agent_y, block_x, block_y,
            target_x, target_y].
    """
    return [
        agent_x,
        agent_y,
        block_x,
        block_y,
        target_x,
        target_y,
    ]


def _relu(x: np.ndarray) -> np.ndarray:
    """
    Compute ReLU activation element-wise.

    PARAMETERS:
      x (np.ndarray): Input array.

    RETURNS:
      np.ndarray: max(0, x) applied element-wise.
    """
    return np.maximum(0, x)


def _forward_pass(state: np.ndarray) -> np.ndarray:
    """
    Run three-layer MLP forward pass.

    Layer 1: state @ W1 + b1, ReLU
    Layer 2: h1 @ W2 + b2, ReLU
    Layer 3: h2 @ W3 + b3, tanh

    PARAMETERS:
      state (np.ndarray): 1-D state vector of shape (6,).

    RETURNS:
      np.ndarray: Action vector of shape (2,) clamped to [-1, 1].
    """
    h1 = _relu(state @ W1 + b1)
    h2 = _relu(h1 @ W2 + b2)
    action = np.tanh(h2 @ W3 + b3)
    return action


def _format_action(action: np.ndarray) -> str:
    """
    Format action array to comma-separated string for Bridge.

    PARAMETERS:
      action (np.ndarray): Action vector of shape (2,).

    RETURNS:
      str: Formatted string (e.g., "0.1234,-0.5678").
    """
    return f"{action[0]:.4f},{action[1]:.4f}"


def predict_action(
    agent_x: float,
    agent_y: float,
    block_x: float,
    block_y: float,
    target_x: float,
    target_y: float,
) -> str:
    """
    Predict robot action from state observation.

    Prepares features, runs the MLP forward pass, and returns the
    predicted action as a formatted string.

    PARAMETERS:
      agent_x (float): Agent x position.
      agent_y (float): Agent y position.
      block_x (float): Block x position.
      block_y (float): Block y position.
      target_x (float): Target x position.
      target_y (float): Target y position.

    RETURNS:
      str: Predicted action string (e.g., "0.1234,-0.5678") or error message.
    """
    try:
        raw_state = _prepare_raw_state(
            agent_x,
            agent_y,
            block_x,
            block_y,
            target_x,
            target_y,
        )
        state = np.array(raw_state, dtype=np.float32)
        action = _forward_pass(state)
        return _format_action(action)
    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Application interface setup
# Expose predict_action function for direct microcontroller calls via Bridge protocol
Bridge.provide("predict_action", predict_action)

# Initialize web UI server on port 7000 for user interaction
ui = WebUI(port=7000)


# Private helper functions for on_predict (in call order)
def _extract_state_measurements(data: dict) -> tuple:
    """
    Extract state measurements from request data.

    PARAMETERS:
      data (dict): Request data containing state measurements.

    RETURNS:
      tuple: (agent_x, agent_y, block_x, block_y,
              target_x, target_y) with 0.0 defaults.
    """
    return (
        data.get("agent_x", 0.0),
        data.get("agent_y", 0.0),
        data.get("block_x", 0.0),
        data.get("block_y", 0.0),
        data.get("target_x", 0.0),
        data.get("target_y", 0.0),
    )


def _send_result_to_clients(action_str: str) -> None:
    """
    Send prediction result to web client and servo hardware.

    PARAMETERS:
      action_str (str): Predicted action string (e.g., "0.1234,-0.5678").

    RETURNS:
      None
    """
    ui.send_message("action_result", {"action": action_str})
    Bridge.call("set_servos", action_str)
    Bridge.call("display_status", "running")


def on_predict(client, data):
    """
    Handle action prediction request from web interface.

    Extracts state measurements, runs prediction, and broadcasts result to web
    client, servo hardware, and LED matrix display.

    PARAMETERS:
      client: The client connection requesting prediction.
      data (dict): Request data with agent_x, agent_y, block_x, block_y,
                   target_x, target_y.

    RETURNS:
      None
    """
    (
        agent_x,
        agent_y,
        block_x,
        block_y,
        target_x,
        target_y,
    ) = _extract_state_measurements(data)
    action_str = predict_action(
        agent_x,
        agent_y,
        block_x,
        block_y,
        target_x,
        target_y,
    )
    _send_result_to_clients(action_str)


# Script-level event handling and application initialization
# Register the on_predict handler to receive prediction requests via web socket
ui.on_message("predict", on_predict)

# Start the application main loop
App.run()
