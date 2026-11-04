#!/bin/sh
# Install the camera server and the WiFi watchdog as boot services.
#   install:    cd ~/trova-stream && sudo sh install.sh
#   uninstall:  cd ~/trova-stream && sudo sh install.sh --uninstall
set -e
cd "$(dirname "$0")"
UNITS="trova-stream.service trova-wifi-watchdog.service trova-wifi-watchdog.timer"

if [ "$1" = "--uninstall" ]; then
    systemctl disable --now trova-stream.service trova-wifi-watchdog.timer 2>/dev/null || true
    for u in $UNITS; do rm -f "/etc/systemd/system/$u"; done
    rm -f /usr/local/sbin/trova-wifi-watchdog
    systemctl daemon-reload
    echo "trova services removed"
    exit 0
fi

install -m 755 trova-wifi-watchdog.sh /usr/local/sbin/trova-wifi-watchdog
install -m 644 $UNITS /etc/systemd/system/
systemctl daemon-reload
pkill -f "^python3 -u stream.py" || true    # a copy started by hand would hold port 8080
systemctl enable --now trova-stream.service trova-wifi-watchdog.timer
systemctl is-active trova-stream.service trova-wifi-watchdog.timer
