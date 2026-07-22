# LeRobot Policy Deployment App

**AUTHOR:** [Kevin Thomas](ket189@pitt.edu)

**CREATION DATE:** February 28, 2026  
**UPDATE DATE:** February 28, 2026

Trained Policy Source: [HERE](https://github.com/mytechnotalent/lerobot)

Deploy a behaviour-cloned NumPy MLP robot-control policy to the Arduino UNO Q and drive servos in real time while monitoring actions live through the web interface.

## Description

The App uses a from-scratch NumPy Multi-Layer Perceptron (MLP) policy trained entirely in simulation with the [LeRobot Simulation Environment](https://github.com/mytechnotalent/lerobot) to control a servo arm on the [Arduino UNO Q](https://www.arduino.cc/product-uno-q). It takes a 6-D state observation (agent x, agent y, block x, block y, target x, target y) and outputs a 2-D action (dx, dy) to drive two servos for the PushT task. Users can enter state measurements through a web interface, and the predicted action is sent to the servos while the episode status is visualized on the 8 x 13 LED matrix with unique patterns for success, timeout, running, and manual-override states.

The `assets` folder contains the **frontend** components of the application, including the HTML, CSS, and JavaScript files that make up the web user interface. The `python` folder contains the application **backend** with NumPy model inference and WebUI handling. The Arduino sketch manages servo PWM output and LED matrix display.

## Bricks Used

The LeRobot Policy Deployment App uses the following Bricks:

- `arduino:web_ui`: Brick to create a web interface for entering state observations and displaying predicted actions.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)
- SG90 micro servos (x2)
- External 5 V / 2 A servo power supply (x1)

### Software

- Arduino App Lab
- NumPy (for neural network inference)

## How to Use the Example

### Clone the Example

1. Clone the example to your workspace.

### Run the App

1. Click the **Run** button in App Lab to start the application.
2. Open the App in your browser at `<UNO-Q-IP-ADDRESS>:7000`
3. Enter the six state features:
   - **Agent X**: agent x position (e.g., `0.5`)
   - **Agent Y**: agent y position (e.g., `0.5`)
   - **Block X**: block x position (e.g., `0.3`)
   - **Block Y**: block y position (e.g., `0.7`)
   - **Target X**: target x position (e.g., `0.5`)
   - **Target Y**: target y position (e.g., `0.5`)
4. Click **Predict Action** to see the result

### Input Validation

The web interface validates that all inputs are proper floats:

- Integers are rejected (e.g., `1` must be entered as `1.0`)
- Text/strings are rejected
- Valid format examples: `0.5`, `0.3`, `-0.2`, `1.0`

### Example Measurements

Try these sample state observations to test action predictions:

| Scenario        | Agent X | Agent Y | Block X | Block Y | Target X | Target Y |
| --------------- | ------- | ------- | ------- | ------- | -------- | -------- |
| Push Right      | 0.3     | 0.5     | 0.5     | 0.5     | 0.7      | 0.5      |
| Push Up         | 0.5     | 0.7     | 0.5     | 0.5     | 0.5      | 0.3      |
| Already On Goal | 0.4     | 0.5     | 0.5     | 0.5     | 0.5      | 0.5      |

## How it Works

Once the application is running, the device performs the following operations:

- **Serving the web interface and handling WebSocket communication.**

  The `web_ui` Brick provides the web server and WebSocket communication:

  ```python
  from arduino.app_bricks.web_ui import WebUI

  ui = WebUI(port=7000)
  ui.on_message("predict", on_predict)
  ```

- **Loading the trained NumPy MLP policy.**

  The application loads the pre-trained policy weights from the `.npz` archive:

  ```python
  from arduino.app_utils import *
  import numpy as np

  _data = np.load("/app/python/best_policy.npz")
  W1, b1 = _data["W1"], _data["b1"]
  W2, b2 = _data["W2"], _data["b2"]
  W3, b3 = _data["W3"], _data["b3"]
  ```

  The weights are loaded once at startup and held in memory for fast inference.

- **Running the MLP forward pass to compute actions.**

  The `predict_action()` function takes six state features and returns the action string:

  ```python
  def predict_action(agent_x: float, agent_y: float,
                     block_x: float, block_y: float,
                     target_x: float, target_y: float) -> str:
      raw_state = _prepare_raw_state(agent_x, agent_y, block_x, block_y,
                                     target_x, target_y)
      state = np.array(raw_state, dtype=np.float32)
      action = _forward_pass(state)
      return _format_action(action)
  ```

  The model outputs a 2-D action clamped to [-1, 1] by the tanh output layer.

- **Handling web interface predictions and updating the hardware.**

  When a user submits state measurements through the web interface:

  ```python
  def on_predict(client, data):
      (agent_x, agent_y, block_x, block_y,
       target_x, target_y) = _extract_state_measurements(data)
      action_str = predict_action(agent_x, agent_y, block_x, block_y,
                                  target_x, target_y)
      _send_result_to_clients(action_str)
  ```

- **Sending actions to the MCU and driving servos.**

  Actions are sent to the STM32 MCU via Bridge, which maps them to servo angles:

  ```python
  def _send_result_to_clients(action_str: str) -> None:
      ui.send_message("action_result", {"action": action_str})
      Bridge.call("set_servos", action_str)
      Bridge.call("display_status", "running")
  ```

- **Displaying status patterns on the LED matrix.**

  The sketch receives the status and displays the corresponding pattern:

  ```cpp
  void display_status(String status) {
    _load_status_frame(status);
  }
  ```

The high-level data flow looks like this:

```
Web Browser Input -> WebSocket -> Python Backend -> NumPy MLP -> Bridge -> LED Matrix
                                                        |
                                                        v
                                                 Bridge -> Servos
```

- **`ui = WebUI(port=7000)`**: Initializes the web server that serves the HTML interface and handles WebSocket communication.

- **`ui.on_message("predict", on_predict)`**: Registers a WebSocket message handler that responds when the user submits measurements.

- **`ui.send_message("action_result", ...)`**: Sends prediction results to the web client in real-time.

- **`Bridge.provide("predict_action", predict_action)`**: Exposes the prediction function for direct microcontroller calls via Bridge protocol.

- **`Bridge.call("set_servos", action_str)`**: Sends servo angle commands from the Linux side to the STM32 MCU.

- **`Bridge.call("display_status", status)`**: Calls the Arduino function to update the LED matrix display.

- **`predict_action()`**: Takes six float features (agent x, agent y, block x, block y, target x, target y), runs the three-layer MLP forward pass (ReLU -> ReLU -> tanh), and returns the predicted action string.

- **`_relu()`**: Computes `np.maximum(0, x)` -- the hidden-layer activation function.

- **`_forward_pass()`**: Runs the full three-layer MLP: hidden1 (256 ReLU) -> hidden2 (256 ReLU) -> output (2 tanh).

- **`_prepare_raw_state()`**: Assembles the six input floats into the correct model input order.

- **`_format_action()`**: Converts the NumPy action array to a comma-separated string for Bridge.

- **`App.run()`**: Starts the application main loop.

### 🔧 Frontend (`index.html` + `app.js`)

The web interface provides a form for entering state observations with validation.

- **Socket.IO connection**: Establishes WebSocket communication with the Python backend through the `web_ui` Brick.

- **`socket.emit("predict", data)`**: Sends state data to the backend when the user clicks the predict button.

- **`socket.on("action_result", ...)`**: Receives prediction results and updates the UI accordingly.

- **`_is_valid_float()`**: Validates that inputs are proper floats (rejects integers and strings).

### 🔧 Hardware (`sketch.ino`)

The Arduino code manages servo control and LED matrix feedback. It runs on the STM32U585 MCU.

- **`matrix.begin()`**: Initializes the matrix driver, making the LED display ready to show patterns.

- **`Bridge.begin()`**: Opens the serial communication bridge to the host Python® runtime.

- **`Bridge.provide("set_servos", set_servos)`**: Registers the servo control function to be callable from Python.

- **`Bridge.provide("display_status", display_status)`**: Registers the LED display function to be callable from Python.

- **`set_servos(String angles)`**: Parses comma-separated action values, maps them from [-1, 1] to [0, 180] degrees, and writes them to PWM pins D9 and D10.

- **`display_status(String status)`**: Receives the episode status and displays the corresponding 8 x 13 frame on the LED matrix.

- **`_parse_angles(String, int&, int&)`**: Private helper that converts action floats to constrained servo angles.

- **`_load_status_frame(String status)`**: Private helper that selects the correct frame array based on status.

- **`loop()`**: Waits for commands from the Python backend and updates hardware accordingly.

- **`status_frames.h`**: Header file that stores the pixel patterns for each status:
  - **Success**: Checkmark pattern
  - **Timeout**: X mark pattern
  - **Running**: Right arrow pattern
  - **Manual**: Pause bars pattern
  - **Unknown**: Question mark for error cases

## Neural Network Architecture

The MLP policy consists of:

- **fc1**: Input (6) -> Output (256), ReLU activation
- **fc2**: Input (256) -> Output (256), ReLU activation
- **out**: Input (256) -> Output (2), tanh activation

The model takes 6 input features (agent x, agent y, block x, block y, target x, target y) and outputs 2 continuous actions (dx, dy) clamped to [-1, 1] by the tanh output layer. The entire forward pass runs in pure NumPy -- no PyTorch, no TensorFlow, no ONNX runtime. Weights are stored in a single `.npz` archive (~528 KB).

## Author

**Kevin Thomas**

- Creation Date: February 28, 2026
- Last Updated: February 28, 2026
