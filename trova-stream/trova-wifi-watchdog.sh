#!/bin/sh
# trova-wifi-watchdog: bring WiFi back if NetworkManager gave up.
# NetworkManager stops retrying after a failed connection until the next
# reboot. trova-wifi-watchdog.timer runs this every minute.
CON=hackathon

state=$(nmcli -t -g GENERAL.STATE device show wlan0 2>/dev/null)
case "$state" in
    20*|30*) ;;          # unavailable / disconnected: reconnect
    *) exit 0 ;;         # connected or busy connecting: leave it alone
esac

logger -t trova-wifi "wlan0 is '$state', reconnecting to $CON"
nmcli --wait 40 connection up "$CON" >/dev/null 2>&1 \
    && logger -t trova-wifi "reconnected" \
    || logger -t trova-wifi "reconnect failed, will retry"
