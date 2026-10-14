"""Sonar tests against REAL PyBullet geometry (skipped if PyBullet is missing)."""
import numpy as np
import pytest

pb = pytest.importorskip("pybullet")

from src.sensors import Sonar, SonarConfig, SensorHealth            # noqa: E402
from src.simulation.truth import snapshot_from_pybullet             # noqa: E402

URDF = "assets/auv/auv.urdf"


@pytest.fixture()
def world():
    client = pb.connect(pb.DIRECT)
    auv = pb.loadURDF(URDF, basePosition=(0, 0, -5), physicsClientId=client)
    yield client, auv
    pb.disconnect(client)


def add_box(client, center, half):
    col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=client)
    return pb.createMultiBody(0, col, basePosition=center, physicsClientId=client)


def quiet(**kw):
    base = dict(horizontal_rays=21, vertical_rays=5, range_noise_std_m=0.0, dropout_probability=0.0,
                detection_probability=1.0, max_range_m=30.0, min_range_m=0.2)
    base.update(kw)
    return SonarConfig(**base)


def scan(world, cfg, t=0.0):
    client, auv = world
    return Sonar(auv, client, cfg).measure(snapshot_from_pybullet(auv, t, client))


def test_detects_real_wall_and_center_ray_range_is_exact(world):
    client, _ = world
    add_box(client, (10.5, 0, -5), (0.5, 20, 20))                 # wall, front face at x = 10
    m = scan(world, quiet())
    center = 2 * 21 + 10                                           # el index 2 (0 deg), az index 10 (0 deg)
    assert m.valid and m.detected[center]
    assert np.isclose(m.value[center], 10.0 - 0.8, atol=1e-3)      # sensor origin is 0.8 m ahead of CoG


def test_all_rays_hit_wall_with_planar_geometry(world):
    client, _ = world
    add_box(client, (10.5, 0, -5), (0.5, 40, 40))
    m = scan(world, quiet())
    assert m.value.shape == (105,) and m.detected.all()
    expected = 9.2 / (np.cos(m.elevations) * np.cos(m.azimuths))   # plane at x = 9.2 from sensor
    assert np.allclose(m.value, expected, atol=1e-3)
    assert m.points_sensor.shape == (105, 3)
    assert np.allclose(m.points_sensor[:, 0], 10.0, atol=1e-3)     # every point lies on the wall plane x=10


def test_no_hit_returns_max_range_and_not_detected(world):
    m = scan(world, quiet())
    assert (m.value == 30.0).all() and not m.detected.any() and np.isnan(m.points_sensor).all()
    assert m.valid                                                  # an empty scene is a valid scan


def test_obstacle_beyond_max_range_not_detected(world):
    client, _ = world
    add_box(client, (50.5, 0, -5), (0.5, 40, 40))
    assert not scan(world, quiet(max_range_m=30.0)).detected.any()
    assert scan(world, quiet(max_range_m=60.0)).detected.any()


def test_min_range_filters_close_returns(world):
    client, _ = world
    add_box(client, (1.15, 0, -5), (0.05, 40, 40))                  # face at x=1.1 -> 0.3 m from sensor
    assert scan(world, quiet(min_range_m=0.2)).detected.any()
    assert not scan(world, quiet(min_range_m=0.5)).detected.any()


def test_fov_limits_which_rays_see_an_off_axis_obstacle(world):
    client, _ = world
    # Small post at bearing +60 deg (left), 8 m away from the sensor origin, level.
    sx, a = 0.8, np.deg2rad(60)
    add_box(client, (sx + 8 * np.cos(a), 8 * np.sin(a), -5), (0.3, 0.3, 5))
    assert not scan(world, quiet(horizontal_fov_deg=90, horizontal_rays=91)).detected.any()      # +/-45 deg
    m = scan(world, quiet(horizontal_fov_deg=150, horizontal_rays=151))                            # +/-75 deg
    assert m.detected.any()
    assert (m.azimuths[m.detected] > 0).all()                       # +azimuth = LEFT (+Y)
    assert abs(np.rad2deg(m.azimuths[m.detected]).mean() - 60) < 3  # found at the right bearing


def test_right_side_obstacle_found_at_negative_azimuth(world):
    client, _ = world
    a = np.deg2rad(-30)
    add_box(client, (0.8 + 8 * np.cos(a), 8 * np.sin(a), -5), (0.3, 0.3, 5))
    m = scan(world, quiet(horizontal_rays=91))
    assert m.detected.any() and (m.azimuths[m.detected] < 0).all()


def test_vertical_fov_elevation_positive_up(world):
    client, _ = world
    add_box(client, (10.5, 0, -2.0), (0.5, 40, 0.2))                # thin slab ABOVE the vehicle (z about -2)
    m = scan(world, quiet())
    assert m.detected.any() and (m.elevations[m.detected] > 0).all()


def test_sensor_follows_vehicle_orientation(world):
    client, auv = world
    add_box(client, (0, 10.5, -5), (40, 0.5, 40))                   # wall to the world +Y side
    m0 = scan(world, quiet())                                       # nose along +X, wall on the left (+Y)
    center = 2 * 21 + 10
    assert not m0.detected[center]                                  # straight ahead sees nothing
    assert m0.detected.any() and (m0.azimuths[m0.detected] > 0).all()   # only left-looking rays see the wall
    pb.resetBasePositionAndOrientation(auv, (0, 0, -5), pb.getQuaternionFromEuler((0, 0, np.pi / 2)),
                                       physicsClientId=client)
    m = scan(world, quiet())                                        # now the nose points at +Y
    assert m.detected[center] and np.isclose(m.value[center], 10.0 - 0.8, atol=1e-3)


def test_own_hull_never_returned(world):
    m = scan(world, quiet(min_range_m=0.0))
    assert not m.detected.any()


def test_noise_statistics_and_determinism(world):
    client, _ = world
    add_box(client, (10.5, 0, -5), (0.5, 40, 40))
    cfg = quiet(range_noise_std_m=0.1, seed=3)
    s1, s2 = Sonar(world[1], client, cfg), Sonar(world[1], client, cfg)
    t = snapshot_from_pybullet(world[1], 0.0, client)
    a = np.array([s1.measure(t).value for _ in range(300)])
    b = np.array([s2.measure(t).value for _ in range(300)])
    assert np.array_equal(a, b)
    center = 2 * 21 + 10
    assert np.isclose(a[:, center].std(), 0.1, rtol=0.2) and np.isclose(a[:, center].mean(), 9.2, atol=0.03)
    s3 = Sonar(world[1], client, quiet(range_noise_std_m=0.1, seed=4))
    assert not np.array_equal(s3.measure(t).value, a[0])


def test_dropout_and_detection_probability(world):
    client, _ = world
    add_box(client, (10.5, 0, -5), (0.5, 40, 40))
    t = snapshot_from_pybullet(world[1], 0.0, client)
    s = Sonar(world[1], client, quiet(dropout_probability=0.3, seed=1))
    frac = np.mean([s.measure(t).detected.mean() for _ in range(200)])
    assert 0.66 < frac < 0.74                                       # ~70 % of rays survive
    s = Sonar(world[1], client, quiet(detection_probability=0.5, seed=2))
    assert 0.45 < np.mean([s.measure(t).detected.mean() for _ in range(200)]) < 0.55
    s = Sonar(world[1], client, quiet(dropout_probability=1.0))
    m = s.measure(t)
    assert not m.detected.any() and (m.value == 30.0).all()         # dropouts never leak true range


def test_scan_health_and_injected_failure(world):
    client, _ = world
    add_box(client, (10.5, 0, -5), (0.5, 40, 40))
    t = snapshot_from_pybullet(world[1], 0.0, client)
    s = Sonar(world[1], client, quiet(dropout_probability=1.0))
    hs = [s.measure(t).health for _ in range(40)]
    assert hs[0] == SensorHealth.HEALTHY and hs[-1] == SensorHealth.FAILED
    s2 = Sonar(world[1], client, quiet())
    s2.inject_fault("failed")
    m = s2.measure(t)
    assert not m.valid and not m.detected.any() and m.health == SensorHealth.FAILED


def test_single_ray_configuration_and_bad_config(world):
    m = scan(world, quiet(horizontal_rays=1, vertical_rays=1))
    assert m.value.shape == (1,) and m.azimuths[0] == 0 and m.elevations[0] == 0
    with pytest.raises(ValueError):
        Sonar(world[1], world[0], quiet(min_range_m=40.0))
    with pytest.raises(ValueError):
        Sonar(world[1], world[0], quiet(horizontal_rays=0))
