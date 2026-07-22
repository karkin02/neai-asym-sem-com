/*
 * FILE: led_matrix.cpp
 *
 * DESCRIPTION:
 *   LED matrix implementation for 8x13 status display.
 *   Provides functions to convert frame arrays and load them onto the matrix.
 *
 * BRIEF:
 *   Implements the LED matrix object and helper functions for frame handling.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 28, 2026
 * UPDATE DATE: February 28, 2026
 *
 * NOTES:
 *   Requires Arduino_LED_Matrix library.
 */

#include "led_matrix.h"

Arduino_LED_Matrix matrix;

/**
 * Private helper to initialize framebuffer array to zero.
 *
 * PARAMETERS:
 *   dest (uint32_t[4]): Framebuffer array to clear.
 *
 * RETURN:
 *   void
 */
void _init_framebuffer(uint32_t dest[4])
{
    for (int i = 0; i < 4; i++)
        dest[i] = 0;
}

/**
 * Private helper to set a single bit in framebuffer.
 *
 * PARAMETERS:
 *   dest (uint32_t[4]): Framebuffer array.
 *   bitPos (int): Absolute bit position (0-103 for 8x13).
 *
 * RETURN:
 *   void
 */
void _set_bit(uint32_t dest[4], int bitPos)
{
    int idx = bitPos / 32;
    int shift = 31 - (bitPos % 32);
    dest[idx] |= (1UL << shift);
}

/**
 * Convert 8x13 frame array to 4-element 32-bit framebuffer.
 *
 * Iterates through 2D frame array and sets corresponding bits in
 * the framebuffer for LED matrix display.
 *
 * PARAMETERS:
 *   src (const uint8_t[8][13]): Source frame array (1=on, 0=off).
 *   dest (uint32_t[4]): Destination 4-element framebuffer.
 *
 * RETURN:
 *   void
 */
void convertToFrameBuffer(const uint8_t src[8][13], uint32_t dest[4])
{
    _init_framebuffer(dest);
    int bitPos = 0;
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 13; col++)
        {
            if (src[row][col])
                _set_bit(dest, bitPos);
            bitPos++;
        }
    }
}

/**
 * Load a single 8x13 frame to the LED matrix.
 *
 * Converts the frame array to framebuffer format and loads it onto
 * the hardware matrix display.
 *
 * PARAMETERS:
 *   frame (const uint8_t[8][13]): Frame array to display.
 *
 * RETURN:
 *   void
 */
void loadFrame8x13(const uint8_t frame[8][13])
{
    uint32_t frameBuffer[4];
    convertToFrameBuffer(frame, frameBuffer);
    matrix.loadFrame(frameBuffer);
}
