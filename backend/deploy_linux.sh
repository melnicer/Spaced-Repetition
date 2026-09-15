#!/bin/bash
# Linux VPS Deployment Script for FlutterStudy Backend
set -e

echo "Updating system and installing python3-pip & python3-venv..."
sudo apt update && sudo apt install -y python3-pip python3-venv python3-full

echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Configuring systemd service..."
SERVICE_PATH="/etc/systemd/system/flutterstudy.service"
CURRENT_DIR=$(pwd)
USER_NAME=$(whoami)

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=FlutterStudy FastAPI Backend Service
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$CURRENT_DIR
ExecStart=$CURRENT_DIR/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable flutterstudy.service
sudo systemctl restart flutterstudy.service

echo "FlutterStudy backend successfully deployed and running on port 8000!"
sudo systemctl status flutterstudy.service --no-pager
