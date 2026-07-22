#!/bin/bash

# Deploy script for LeRobot Policy Deployment App

set -e

# Path to ADB
ADB="/Applications/platform-tools/adb"

# Target deployment path
TARGET_PATH="/home/arduino/ArduinoApps/lerobot-policy-deployment-app"

# Echo header
echo ""
echo "=========================================="
echo "DEPLOY LeRobot Policy Deployment App"
echo "=========================================="
echo ""

# Check ADB
echo "[1/3] Checking Arduino connection..."
if ! $ADB devices 2>/dev/null | grep -q "device$"; then
    echo "ERROR: Arduino not connected"
    exit 1
fi
echo "✓ Connected"

# Deploy all files
echo ""
echo "[2/3] Deploying all files..."
$ADB shell "rm -rf $TARGET_PATH"
$ADB shell "mkdir -p $TARGET_PATH"
$ADB push python "$TARGET_PATH/" > /dev/null
$ADB push assets "$TARGET_PATH/" > /dev/null
$ADB push sketch "$TARGET_PATH/" > /dev/null
$ADB push app.yaml "$TARGET_PATH/" > /dev/null
$ADB push LICENSE "$TARGET_PATH/" > /dev/null
$ADB push README.md "$TARGET_PATH/" > /dev/null
echo "✓ All files deployed"

# Summary
echo ""
echo "[3/3] Deployment Complete"
echo ""
