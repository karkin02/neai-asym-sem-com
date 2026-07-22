/**
 * FILE: sketch.ino
 *
 * DESCRIPTION:
 *   Drives servos and displays episode status on an 8x13 LED matrix using Arduino.
 *
 * BRIEF:
 *   Receives action values from Python backend via Bridge to set servo positions.
 *   Displays episode status frames (running, success, timeout) on the LED matrix.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 28, 2026
 * UPDATE DATE: February 28, 2026
 */

#include <Arduino_RouterBridge.h>
#include <Servo.h>
#include "status_frames.h"
#include "led_matrix.h"

// Servo objects for X and Y axis control
Servo servoX;
Servo servoY;

// PWM pins for servo output
const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;

/**
 * Private helper function to parse action string into servo angles.
 *
 * Maps action values from [-1, 1] to [0, 180] degrees.
 *
 * PARAMETERS:
 *   angles (String): Comma-separated action values (e.g., "0.1234,-0.5678").
 *   angleX (int&): Output servo X angle in degrees.
 *   angleY (int&): Output servo Y angle in degrees.
 *
 * RETURN:
 *   void
 */
void _parse_angles(String angles, int &angleX, int &angleY)
{
    int commaIdx = angles.indexOf(',');
    float actionX = angles.substring(0, commaIdx).toFloat();
    float actionY = angles.substring(commaIdx + 1).toFloat();
    angleX = (int)(90.0 + actionX * 90.0);
    angleY = (int)(90.0 + actionY * 90.0);
    angleX = constrain(angleX, 0, 180);
    angleY = constrain(angleY, 0, 180);
}

/**
 * Private helper function to load status frame by status name.
 *
 * PARAMETERS:
 *   status (String): The episode status name.
 *
 * RETURN:
 *   void
 */
void _load_status_frame(String status)
{
    if (status == "success")
        loadFrame8x13(success_frame);
    else if (status == "timeout")
        loadFrame8x13(timeout_frame);
    else if (status == "running")
        loadFrame8x13(running_frame);
    else if (status == "manual")
        loadFrame8x13(manual_frame);
    else
        loadFrame8x13(unknown);
}

/**
 * Initialize LED matrix, servos, and Bridge communication.
 *
 * RETURN:
 *   void
 */
void setup()
{
    matrix.begin();
    matrix.clear();
    servoX.attach(SERVO_X_PIN);
    servoY.attach(SERVO_Y_PIN);
    servoX.write(90);
    servoY.write(90);
    Bridge.begin();
    Bridge.provide("set_servos", set_servos);
    Bridge.provide("display_status", display_status);
}

/**
 * Main loop waiting for commands from Python backend.
 *
 * RETURN:
 *   void
 */
void loop()
{
    delay(100);
}

/**
 * Set servo positions from action string received via Bridge.
 *
 * PARAMETERS:
 *   angles (String): Comma-separated action values (e.g., "0.1234,-0.5678").
 *
 * RETURN:
 *   void
 */
void set_servos(String angles)
{
    int angleX, angleY;
    _parse_angles(angles, angleX, angleY);
    servoX.write(angleX);
    servoY.write(angleY);
}

/**
 * Display episode status on LED matrix.
 *
 * PARAMETERS:
 *   status (String): The episode status name (running, success, timeout, manual).
 *
 * RETURN:
 *   void
 */
void display_status(String status)
{
    _load_status_frame(status);
}
