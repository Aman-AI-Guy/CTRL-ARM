#!/bin/sh
# Deploy trova-stream/stream.py to the Pi and restart the service without sudo.
#   sh tools/deploy_stream.sh [host]          (default 192.168.5.160)
#   sh tools/deploy_stream.sh --rollback      (restore the stream.py saved before the first scan-mode deploy)
# Password login (trova/trova) through SSH_ASKPASS, as in trova-cam-setup-notes.md.
# The service runs as user trova with Restart=always: killing it (anchored pattern) makes systemd
# start the new file in ~3 s. Anyone watching the stream is cut off for those seconds.
set -e
cd "$(dirname "$0")/.."
HOST=192.168.5.160
ROLLBACK=0
for a in "$@"; do
    case "$a" in
        --rollback) ROLLBACK=1 ;;
        *) HOST="$a" ;;
    esac
done
ASK="${TMP:-/tmp}/trova-askpass.sh"
printf '#!/bin/sh\necho trova\n' > "$ASK" && chmod +x "$ASK"
export SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSHO="-o ConnectTimeout=8 -o PreferredAuthentications=password -o PubkeyAuthentication=no"
BACKUP=stream.py.bak-20261104      # the original server, saved on the Pi before the first deploy
RESTART='pkill -f "^/usr/bin/python3 -u /home/trova/trova-stream/stream.py" ; sleep 6; systemctl is-active trova-stream'

if [ "$ROLLBACK" = 1 ]; then
    ssh $SSHO "trova@$HOST" "cd ~/trova-stream && cp -p $BACKUP stream.py && md5sum stream.py && $RESTART"
else
    scp $SSHO trova-stream/stream.py "trova@$HOST:trova-stream/stream.py.new"
    ssh $SSHO "trova@$HOST" "cd ~/trova-stream && python3 -m py_compile stream.py.new && rm -rf __pycache__ \
        && { [ -f $BACKUP ] || cp -p stream.py $BACKUP; } && mv stream.py.new stream.py && md5sum stream.py && $RESTART"
    md5sum trova-stream/stream.py
fi
curl -s -m 5 "http://$HOST:8080/status.json"; echo
