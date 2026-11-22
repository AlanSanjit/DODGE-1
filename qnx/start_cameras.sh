#!/bin/sh
# Restart the QNX sensor service with a DODGE-1 USB camera config. Run as root:
#     sudo sh qnx/start_cameras.sh qnx/sensor_dodge_mono.conf    # ONE camera (CAMERA_MODE = "mono")
#     sudo sh qnx/start_cameras.sh                                # two cameras (qnx/sensor_dodge_usb.conf)
# The boot-time sensor service uses /etc/config/sensor/sensor_rpi5.conf, which expects two Pi ribbon
# cameras (IMX708); without them its platform init fails and it serves NO cameras - so it must be
# stopped first ("Another instance of SensorService is already running" otherwise).
# A reboot restores the image's default sensor configuration.
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/sensor_dodge_usb.conf}
DST=/etc/config/sensor/sensor_dodge_usb.conf

if grep -v '^[[:space:]]*#' "$SRC" | grep -q "_BUS\|_DEV"; then
    echo "Edit $SRC first: replace LEFT_BUS/LEFT_DEV/RIGHT_BUS/RIGHT_DEV with numbers (see the comments)."
    exit 1
fi
NUNITS=$(grep -c '^[[:space:]]*begin SENSOR_UNIT_' "$SRC")
cp "$SRC" "$DST" || exit 1
chmod 644 "$DST"

echo "stopping the running sensor service..."
slay -f sensor 2>/dev/null
sleep 1
slay -9 -f sensor 2>/dev/null
sleep 1

# NOTE: no "-b external": that mode needs the Pi 5 platform library declared in the config
# (sensor_rpi5.conf does; ours is USB-only) and otherwise fails with
# "Library cannot be found trying to dlopen library" / "Unable to initialize Platform".
echo "starting: sensor -U 521:521 -r /data/share/sensor -c $DST  ($NUNITS camera(s))"
sensor -U 521:521 -r /data/share/sensor -c "$DST" &
sleep 4
echo "--- /dev/sensor ---"
ls -l /dev/sensor 2>/dev/null
if ls /dev/sensor 2>/dev/null | grep -q "camera$NUNITS"; then
    echo "OK: $NUNITS camera(s) registered"
else
    echo "PROBLEM: cameras not registered - see the log below"
fi
echo "--- sensor service log (errors and camera detection) ---"
slog2info -b sensor_service 2>/dev/null | grep -i -E "error|fail|adding device|ignoring device|unit 0x|already running" | tail -n 20
if [ "$NUNITS" = "1" ]; then
    echo "Next (as qnxuser): ./stereo_capture -l   (expect: camera units available: 1)"
else
    echo "Next (as qnxuser): ./stereo_capture -l   (expect: camera units available: 1 2)"
fi
