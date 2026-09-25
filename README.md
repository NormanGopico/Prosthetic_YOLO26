# Prosthetic YOLO26 Object Detection

Computer vision subsystem for a prosthetic-hand research project using Ultralytics YOLO26 for real-time household-object detection.

## Current Object Classes

The current proof-of-concept detector uses 10 object classes:

0. Key
1. Wallet
2. Bottle
3. Can
4. Book
5. Smartphone
6. Fork
7. Spoon
8. Pencil
9. Apple

The object scope is temporary and may change during development.

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt