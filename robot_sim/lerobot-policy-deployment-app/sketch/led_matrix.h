/*
 * FILE: led_matrix.h
 *
 * DESCRIPTION:
 *   LED matrix interface for 8x13 status display.
 *   Provides functions to convert frame arrays and load them onto the matrix.
 *
 * BRIEF:
 *   Declares the LED matrix object and helper functions for frame handling.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 28, 2026
 * UPDATE DATE: February 28, 2026
 *
 * NOTES:
 *   Requires Arduino_LED_Matrix library.
 */

#ifndef LED_MATRIX_H
#define LED_MATRIX_H

#include <Arduino_LED_Matrix.h>
#include <stdint.h>

/**
 * OBJECT: matrix
 *
 * TYPE:
 *   Arduino_LED_Matrix
 *
 * DESCRIPTION:
 *   Represents the 8x13 LED matrix used for displaying episode status frames.
 *   Provides methods to initialize the matrix, clear it, and load frames.
 *
 * USAGE:
 *   matrix.begin()        - Initialize the hardware.
 *   matrix.clear()        - Clear all LEDs.
 *   matrix.loadFrame(...) - Load a 4-element framebuffer representing an 8x13 frame.
 */
extern Arduino_LED_Matrix matrix;

/**
 * Converts 8x13 frame array to 4-element 32-bit framebuffer.
 *
 * PARAMETERS:
 *   src  - 8x13 array of frame pixels (1 = on, 0 = off)
 *   dest - 4-element uint32_t framebuffer to fill
 *
 * RETURN:
 *   None
 */
void convertToFrameBuffer(const uint8_t src[8][13], uint32_t dest[4]);

/**
 * Loads a single frame to the LED matrix.
 *
 * PARAMETERS:
 *   frame - 8x13 frame array
 *
 * RETURN:
 *   None
 */
void loadFrame8x13(const uint8_t frame[8][13]);

#endif /* LED_MATRIX_H */
