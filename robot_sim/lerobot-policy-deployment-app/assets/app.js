/**
 * FILE: app.js
 *
 * DESCRIPTION:
 *   Web UI for LeRobot policy deployment app.
 *   Handles form submission, validation, socket communication, and result display.
 *
 * BRIEF:
 *   Manages client-side interactions for robot state input and action output.
 *   Uses WebSocket (Socket.IO) to communicate with Arduino backend.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 28, 2026
 * UPDATE DATE: February 28, 2026
 *
 * SPDX-FileCopyrightText: Copyright (C) Kevin Thomas
 * SPDX-License-Identifier: Apache-2.0
 */

// Initialize Socket.IO connection to server
const socket = io(`http://${window.location.host}`);

// Reference to error container element for connection status messages
let errorContainer;

/**
 * Initialize event listeners when DOM is fully loaded.
 */
document.addEventListener('DOMContentLoaded', () => {
    errorContainer = document.getElementById('error-container');
    _init_socket_events();
    _init_form_handler();
});

/**
 * Private helper to handle connection established event.
 */
function _handle_connect() {
    if (errorContainer) {
        errorContainer.style.display = 'none';
        errorContainer.textContent = '';
    }
}

/**
 * Private helper to handle action result received.
 *
 * PARAMETERS:
 *   message (object): Message object containing action prediction.
 */
function _handle_action_result(message) {
    _display_result(message.action);
}

/**
 * Private helper to handle connection lost event.
 */
function _handle_disconnect() {
    if (errorContainer) {
        errorContainer.textContent = 'Connection to the board lost. Please check the connection.';
        errorContainer.style.display = 'block';
    }
}

/**
 * Initialize Socket.IO event listeners for connection and messages.
 */
function _init_socket_events() {
    socket.on('connect', _handle_connect);
    socket.on('action_result', _handle_action_result);
    socket.on('disconnect', _handle_disconnect);
}

/**
 * Initialize form submission event handler.
 */
function _init_form_handler() {
    const form = document.getElementById('state-form');
    form.addEventListener('submit', _on_form_submit);
}

/**
 * Validate if value is a valid floating-point number.
 *
 * Must have decimal point; rejects pure integers and empty strings.
 *
 * PARAMETERS:
 *   value (string): The value to validate.
 *
 * RETURN:
 *   boolean: true if valid float, false otherwise.
 */
function _is_valid_float(value) {
    const trimmed = value.trim();
    if (trimmed === '') return false;
    if (/^-?\d+$/.test(trimmed)) return false;
    return /^-?\d*\.\d+$/.test(trimmed);
}

/**
 * Private helper to validate all form input fields.
 *
 * Updates error display states for invalid fields.
 *
 * PARAMETERS:
 *   fields (array): Array of field objects with id and errorId.
 *
 * RETURN:
 *   boolean: true if all fields valid, false otherwise.
 */
function _validate_form_fields(fields) {
    let valid = true;
    fields.forEach(field => {
        const input = document.getElementById(field.id);
        const error = document.getElementById(field.errorId);
        if (!_is_valid_float(input.value)) {
            input.classList.add('error');
            error.style.display = 'block';
            valid = false;
        } else {
            input.classList.remove('error');
            error.style.display = 'none';
        }
    });
    return valid;
}

/**
 * Private helper to collect and parse form input values.
 *
 * RETURN:
 *   object: Object with agent_x, agent_y, block_x, block_y,
 *           target_x, target_y properties.
 */
function _collect_form_data() {
    return {
        agent_x: parseFloat(document.getElementById('agent-x').value),
        agent_y: parseFloat(document.getElementById('agent-y').value),
        block_x: parseFloat(document.getElementById('block-x').value),
        block_y: parseFloat(document.getElementById('block-y').value),
        target_x: parseFloat(document.getElementById('target-x').value),
        target_y: parseFloat(document.getElementById('target-y').value)
    };
}

/**
 * Handle form submission and validate before sending prediction request.
 *
 * PARAMETERS:
 *   e (event): The form submission event.
 */
function _on_form_submit(e) {
    e.preventDefault();
    const fields = [
        { id: 'agent-x', errorId: 'ax-error' },
        { id: 'agent-y', errorId: 'ay-error' },
        { id: 'block-x', errorId: 'bx-error' },
        { id: 'block-y', errorId: 'by-error' },
        { id: 'target-x', errorId: 'tx-error' },
        { id: 'target-y', errorId: 'ty-error' }
    ];
    if (_validate_form_fields(fields)) {
        socket.emit('predict', _collect_form_data());
        document.getElementById('result').style.display = 'none';
    }
}

/**
 * Display action result to user on the web UI.
 *
 * PARAMETERS:
 *   action (string): The predicted action string (dx, dy).
 */
function _display_result(action) {
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    resultText.textContent = `Predicted Action: (${action})`;
    resultDiv.style.display = 'block';
}
