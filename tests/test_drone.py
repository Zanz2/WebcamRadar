import socket
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from pixel2voxel.drone import app
from pixel2voxel.drone.cot import CoTMessageGenerator, CoTNetworkHandler, Detection, format_time
from pixel2voxel.drone.detector import VoxelDroneDetector


def test_default_config_spawns_a_new_drone_every_step():
    """With the default cluster_tolerance = 1.0 the match radius
    is 2 voxels, but the simulated drone moves 2.5 voxels per step, so each step
    starts a new track and old ones linger until the 30 s timeout."""
    det = app.build(app.Config())
    sim = app.DroneSimulator()
    sim.step(det)
    sim.step(det)
    drones = det.get_detected_drones()
    assert len(drones) == 2 and drones[0].id != drones[1].id


def test_simulated_drone_is_tracked_with_a_wider_tolerance():
    cfg = app.Config(cluster_tolerance=2.0)  # match radius 4 voxels > 2.5 voxels/step
    det = app.build(cfg)
    sim = app.DroneSimulator()
    sim.step(det)
    first_id = det.get_detected_drones()[0].id
    time.sleep(0.05)
    sim.step(det)
    drones = det.get_detected_drones()
    assert len(drones) == 1 and drones[0].id == first_id
    # Position is on the simulated circle: centre (50, 50) voxels, radius 25, voxel 0.5 m.
    x, y = drones[0].x / cfg.voxel_size, drones[0].y / cfg.voxel_size
    assert abs(np.hypot(x - 50, y - 50) - 25) < 1.5
    assert drones[0].velocity_valid
    assert 0 < np.hypot(drones[0].vx, drones[0].vy)


def test_clusters_split_and_size_limits():
    det = VoxelDroneDetector(20, 20, 20, threshold=0.5, cluster_dist=1.0, min_voxels=3, max_voxels=10)
    det.set_voxel_size(1.0)
    for x in range(3):  # 3-voxel line -> kept
        det.set_voxel_value(x, 0, 0, 1.0)
    det.set_voxel_value(10, 10, 10, 1.0)  # lone voxel -> too small
    for x in range(12):  # 12-voxel line -> too big
        det.set_voxel_value(x, 15, 15, 1.0)
    det.process_voxel_grid()
    drones = det.get_detected_drones()
    assert len(drones) == 1
    assert drones[0].x == 1.5 and drones[0].y == 0.5  # centroid voxel 1 -> (1 + 0.5) * 1 m


def test_cot_message_is_valid_xml():
    gen = CoTMessageGenerator("sensor-1")
    gen.set_sensor_location(38.8895, -77.0352, 0.0)
    msg = gen.generate_cot_message(Detection(100.0, 200.0, 30.0, 0.5, 1, "drone-1-abc"), now=0.0)
    ev = ET.fromstring(msg.split("?>", 1)[1])
    assert ev.get("uid") == "drone-1-abc" and ev.get("type") == "a-u-A-M-F-Q-r"
    assert ev.get("time") == "1970-01-01T00:00:00.000Z" and ev.get("stale") == "1970-01-01T00:01:00.000Z"
    pt = ev.find("point")
    assert float(pt.get("lat")) > 38.8895 and float(pt.get("lon")) > -77.0352
    assert pt.get("hae") == "30.00" and pt.get("ce") == "2.00"
    assert ev.find("detail/contact").get("callsign") == "Drone-dron"
    ET.fromstring(gen.generate_status_message(now=0.0).split("?>", 1)[1])
    assert format_time(1.5) == "1970-01-01T00:00:01.500Z"


def test_network_handler_delivers_to_tcp_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    received = []

    def serve():
        conn, _ = srv.accept()
        conn.settimeout(3)
        buf = b""
        while b"</event>" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
        received.append(buf)
        conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    h = CoTNetworkHandler("127.0.0.1", port)
    h.start()
    h.queue_message("<event uid='x'></event>")
    t.join(timeout=5)
    h.stop()
    srv.close()
    assert received and received[0] == b"<event uid='x'></event>"
