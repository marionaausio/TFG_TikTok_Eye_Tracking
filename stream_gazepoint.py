import re
import socket

from pylsl import StreamInfo, StreamOutlet, local_clock

# ==== SETTINGS ====
HOST = "127.0.0.1"
PORT = 4242
GAZEPOINT_NOMINAL_SRATE = 150.0  # Use 60.0 if GP3HD high-speed mode is disabled.
SOCKET_TIMEOUT_S = 2.0
GAZE_STREAM_NAME = "GazepointEyeTracker"
EVENT_STREAM_NAME = "GazepointEvents"
# ===================

XML_TAG_RE = re.compile(r"^\s*<\s*([A-Za-z0-9_]+)")
XML_ATTR_RE = re.compile(r'([A-Za-z0-9_]+)="([^"]*)"')

# Keep the historical 0-21 order stable for existing consumers.  The six
# appended channels preserve the per-eye pupil values/validity that the old
# 22-channel serializer discarded.  PUPIL_LEFT/RIGHT are compatibility labels
# whose exact Gazepoint sources are now LPD/RPD.
GAZE_CHANNEL_LABELS = [
    "FPOGX",
    "FPOGY",
    "FPOGS",
    "FPOGD",
    "FPOGID",
    "FPOGV",
    "PUPIL_LEFT",
    "PUPIL_RIGHT",
    "EYE_LEFT",
    "EYE_RIGHT",
    "CURSOR_X",
    "CURSOR_Y",
    "BLINK",
    "PUPILMM",
    "DIAL",
    "TTL",
    "PIX",
    "USERDATA_ID",
    "COUNTER",
    "TIME",
    "TIME_TICK",
    "TIMESTAMP",
    "LPV",
    "RPV",
    "LPMM",
    "RPMM",
    "LPMMV",
    "RPMMV",
]


def create_gaze_outlet():
    """Create the primary Gazepoint continuous data outlet."""
    stream_info = StreamInfo(
        name=GAZE_STREAM_NAME,
        type="Gaze",
        channel_count=len(GAZE_CHANNEL_LABELS),
        nominal_srate=GAZEPOINT_NOMINAL_SRATE,
        channel_format="float32",
        source_id="gazepoint_all_fields",
    )

    channels = stream_info.desc().append_child("channels")
    for label in GAZE_CHANNEL_LABELS:
        channels.append_child("channel").append_child_value("label", label)

    stream_info.desc().append_child_value("timestamp_mode", "explicit_lsl_local_clock")
    stream_info.desc().append_child_value("schema_version", "2")
    stream_info.desc().append_child_value(
        "pupil_mapping",
        "PUPIL_LEFT=LPD;PUPIL_RIGHT=RPD;use LPV/RPV and LPMMV/RPMMV validity",
    )
    return StreamOutlet(stream_info)


def create_event_outlet():
    """Create a marker outlet for Gazepoint API/control/calibration events."""
    event_info = StreamInfo(
        name=EVENT_STREAM_NAME,
        type="Markers",
        channel_count=1,
        nominal_srate=0.0,
        channel_format="string",
        source_id="gazepoint_api_events",
    )
    event_info.desc().append_child_value(
        "notes",
        "Carries Gazepoint API events (CAL/ACK/NACK) with local_clock timestamps.",
    )
    return StreamOutlet(event_info)


def parse_xml_packet(packet):
    """Parse a single XML line into (tag, attributes)."""
    packet = str(packet or "").strip()
    if not packet.startswith("<"):
        return None, {}

    tag_match = XML_TAG_RE.search(packet)
    if not tag_match:
        return None, {}

    tag = tag_match.group(1).upper()
    attrs = {key: value for key, value in XML_ATTR_RE.findall(packet)}
    return tag, attrs


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_int(value):
    try:
        return int(float(value))
    except Exception:
        return float("nan")


def parse_cursor(value):
    """Parse Gazepoint CURSOR='x,y' value into two floats."""
    token = str(value or "")
    parts = token.split(",")
    if len(parts) >= 2:
        return parse_float(parts[0]), parse_float(parts[1])
    return float("nan"), float("nan")


def get_attr(attrs, *keys):
    for key in keys:
        if key in attrs:
            return attrs[key]
    return ""


def as_float(attrs, *keys):
    return parse_float(get_attr(attrs, *keys))


def as_int(attrs, *keys):
    return parse_int(get_attr(attrs, *keys))


def as_user_float(attrs):
    value = str(get_attr(attrs, "USER")).strip()
    if not value:
        return float("nan")
    return parse_float(value)


def build_gaze_sample(attrs, lsl_ts):
    cursor_x, cursor_y = parse_cursor(get_attr(attrs, "CURSOR"))
    sample = [
        as_float(attrs, "FPOGX"),
        as_float(attrs, "FPOGY"),
        as_float(attrs, "FPOGS"),
        as_float(attrs, "FPOGD"),
        as_float(attrs, "FPOGID"),
        as_float(attrs, "FPOGV"),
        as_float(attrs, "LPD"),
        as_float(attrs, "RPD"),
        as_float(attrs, "EYE_LEFT"),
        as_float(attrs, "EYE_RIGHT"),
        cursor_x,
        cursor_y,
        as_int(attrs, "BLINK", "BKID"),
        as_float(attrs, "PUPILMM", "LPMM", "RPMM"),
        as_float(attrs, "DIAL"),
        as_int(attrs, "TTL"),
        as_int(attrs, "PIX"),
        as_user_float(attrs),
        as_int(attrs, "COUNTER", "CNT"),
        as_float(attrs, "TIME"),
        as_float(attrs, "TIME_TICK"),
        float(lsl_ts),  # LSL-local timestamp mirror (authoritative timestamp is push_sample arg)
        as_int(attrs, "LPV"),
        as_int(attrs, "RPV"),
        as_float(attrs, "LPMM"),
        as_float(attrs, "RPMM"),
        as_int(attrs, "LPMMV"),
        as_int(attrs, "RPMMV"),
    ]
    if len(sample) != len(GAZE_CHANNEL_LABELS):
        raise RuntimeError(
            f"Gazepoint sample/schema mismatch: {len(sample)} values for "
            f"{len(GAZE_CHANNEL_LABELS)} channels"
        )
    return sample


def sanitize_token(value):
    text = str(value or "").replace(",", ";").replace("\r", " ").replace("\n", " ").strip()
    return text if text else "na"


def build_event_marker(tag, attrs):
    prefix_map = {
        "CAL": "gazepoint_calibration",
        "ACK": "gazepoint_ack",
        "NACK": "gazepoint_nack",
    }
    marker_parts = [prefix_map.get(tag, f"gazepoint_{tag.lower()}")]
    for key in sorted(attrs):
        marker_parts.append(f"{key.lower()}:{sanitize_token(attrs[key])}")
    return ",".join(marker_parts)


def send_enable_commands(sock):
    enable_cmds = [
        "ENABLE_SEND_DATA",
        "ENABLE_SEND_COUNTER",
        "ENABLE_SEND_TIME",
        "ENABLE_SEND_TIME_TICK",
        "ENABLE_SEND_POG_FIX",
        "ENABLE_SEND_POG_LEFT",
        "ENABLE_SEND_POG_RIGHT",
        "ENABLE_SEND_POG_BEST",
        "ENABLE_SEND_PUPIL_LEFT",
        "ENABLE_SEND_PUPIL_RIGHT",
        "ENABLE_SEND_EYE_LEFT",
        "ENABLE_SEND_EYE_RIGHT",
        "ENABLE_SEND_CURSOR",
        "ENABLE_SEND_KB",
        "ENABLE_SEND_BLINK",
        "ENABLE_SEND_PUPILMM",
        "ENABLE_SEND_DIAL",
        "ENABLE_SEND_TTL",
        "ENABLE_SEND_PIX",
        "ENABLE_SEND_USER_DATA",
        "ENABLE_SEND_POG_AAC",
    ]
    for cmd in enable_cmds:
        sock.sendall(f'<SET ID="{cmd}" STATE="1" />\r\n'.encode("ascii"))


def push_event(event_outlet, marker_text):
    try:
        event_outlet.push_sample([marker_text], float(local_clock()))
    except Exception as exc:
        print(f"[WARN] Failed to push event marker: {exc}")


def main():
    gaze_outlet = create_gaze_outlet()
    event_outlet = create_event_outlet()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((HOST, PORT))
        except OSError as exc:
            print(
                f"[WARN] Could not connect to Gazepoint Control at {HOST}:{PORT} ({exc})."
            )
            print(
                "[WARN] Gaze stream NOT started - this is EXPECTED if no eye-tracker is "
                "connected or Gazepoint Control is not running. Other streams (mouse) "
                "continue normally; you can close this window."
            )
            return
        sock.settimeout(SOCKET_TIMEOUT_S)
        send_enable_commands(sock)

        push_event(
            event_outlet,
            (
                "gazepoint_streamer_start,"
                f"host:{HOST},port:{PORT},"
                f"nominal_hz:{GAZEPOINT_NOMINAL_SRATE:.1f}"
            ),
        )
        print(
            "[INFO] Connected to Gazepoint. "
            f"Streaming '{GAZE_STREAM_NAME}' + '{EVENT_STREAM_NAME}'..."
        )

        buffer = ""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print("[WARN] Gazepoint socket closed by peer.")
                    break
                buffer += chunk.decode("utf-8", errors="ignore")

                while "\r\n" in buffer:
                    packet, buffer = buffer.split("\r\n", 1)
                    packet = packet.strip()
                    if not packet:
                        continue

                    tag, attrs = parse_xml_packet(packet)
                    if not tag:
                        continue

                    lsl_ts = float(local_clock())
                    if tag == "REC":
                        sample = build_gaze_sample(attrs, lsl_ts)
                        gaze_outlet.push_sample(sample, lsl_ts)
                    elif tag in {"CAL", "ACK", "NACK"}:
                        marker = build_event_marker(tag, attrs)
                        event_outlet.push_sample([marker], lsl_ts)
                        if tag == "CAL":
                            print(f"[CAL] {marker}")

            except socket.timeout:
                continue
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[WARN] Gazepoint stream loop warning: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Gazepoint stream stopped.")
