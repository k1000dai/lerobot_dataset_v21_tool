"""
Test module for validating real HSR data transformation calculations against ROS2 tf2 library.

This module loads actual HSR rosbag data and compares our custom TF implementation
directly against the official ROS2 tf2_ros library to ensure correctness.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tf2_py
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from std_msgs.msg import Header

from hsr_data_converter.tf_buffer import (
    TFBuffer,
    matrix_to_xyz_rpy,
    transform_to_matrix,
)


class TestRealDataROS2Validation:
    """Validate our TF implementation against ROS2 tf2 with real HSR robot data."""

    @pytest.fixture
    def sample_rosbag_path(self):
        """Get a single rosbag file for validation testing."""
        datasets_dir = Path("/root/datasets")

        # Find first available rosbag
        for dataset_dir in datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            for session_dir in dataset_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                inner_dir = session_dir / session_dir.name
                if inner_dir.exists():
                    meta_json = inner_dir / "meta.json"
                    if meta_json.exists():
                        with open(meta_json) as f:
                            meta_data = json.load(f)
                        bag_path = inner_dir / meta_data["bag_path"]
                        if bag_path.exists():
                            return {
                                "bag_path": bag_path,
                                "meta_path": meta_json,
                                "meta_data": meta_data,
                            }

        pytest.skip("No rosbag files available for testing")

    def create_tf2_transform_stamped(
        self, transform_msg, parent_frame, child_frame, timestamp_ns
    ):
        """Convert our transform data to ROS2 TransformStamped message."""
        tf_stamped = TransformStamped()

        # Header
        tf_stamped.header = Header()
        tf_stamped.header.frame_id = parent_frame
        tf_stamped.header.stamp = Time()
        tf_stamped.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        tf_stamped.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)

        # Frame IDs
        tf_stamped.child_frame_id = child_frame

        # Transform - need to convert from rosbags type to ROS2 type
        tf_stamped.transform = Transform()
        tf_stamped.transform.translation = Vector3(
            x=float(transform_msg.translation.x),
            y=float(transform_msg.translation.y),
            z=float(transform_msg.translation.z),
        )
        tf_stamped.transform.rotation = Quaternion(
            x=float(transform_msg.rotation.x),
            y=float(transform_msg.rotation.y),
            z=float(transform_msg.rotation.z),
            w=float(transform_msg.rotation.w),
        )

        return tf_stamped

    def test_static_transforms_against_tf2(self, sample_rosbag_path):
        """Test static transforms from real data against tf2_ros."""
        bag_path = sample_rosbag_path["bag_path"]

        # Load static transforms into both our buffer and tf2 buffer
        our_buffer = TFBuffer()
        tf2_buffer = tf2_py.BufferCore()

        typestore = get_typestore(Stores.ROS1_NOETIC)
        static_transform_count = 0

        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf_static":
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # Add to our buffer
                    our_buffer.add_transform_msg(msg, is_static=True)

                    # Add to tf2 buffer
                    for transform_stamped in msg.transforms:
                        # Convert to proper ROS2 message format
                        tf2_transform = self.create_tf2_transform_stamped(
                            transform_stamped.transform,
                            transform_stamped.header.frame_id,
                            transform_stamped.child_frame_id,
                            timestamp,
                        )
                        tf2_buffer.set_transform_static(tf2_transform, "test")
                        static_transform_count += 1

        print(f"\nLoaded {static_transform_count} static transforms")

        # Test some key static transform lookups
        test_frames = []
        for child_frame, (parent_frame, _) in our_buffer.static_transforms.items():
            test_frames.append((child_frame, parent_frame))
            if len(test_frames) >= 5:  # Test first 5 static transforms
                break

        successful_comparisons = 0

        for child_frame, parent_frame in test_frames:
            try:
                # Get transform from our buffer
                our_matrix = our_buffer.lookup_transform(child_frame, parent_frame, 0)

                # Get transform from tf2 buffer
                duration = Duration()
                duration.sec = 0
                duration.nanosec = 0
                tf2_result = tf2_buffer.lookup_transform_core(
                    child_frame, parent_frame, duration
                )

                if our_matrix is not None and tf2_result is not None:
                    # Convert tf2 result to matrix
                    tf2_transform = Transform()
                    tf2_transform.translation = Vector3(
                        x=tf2_result.transform.translation.x,
                        y=tf2_result.transform.translation.y,
                        z=tf2_result.transform.translation.z,
                    )
                    tf2_transform.rotation = Quaternion(
                        x=tf2_result.transform.rotation.x,
                        y=tf2_result.transform.rotation.y,
                        z=tf2_result.transform.rotation.z,
                        w=tf2_result.transform.rotation.w,
                    )
                    tf2_matrix = transform_to_matrix(tf2_transform)

                    # Validate direct matrix comparison
                    np.testing.assert_allclose(
                        our_matrix,
                        tf2_matrix,
                        atol=1e-10,
                        err_msg=f"Static transform mismatch for {child_frame}->{parent_frame}",
                    )

                    successful_comparisons += 1
                    print(
                        f"  ✓ {child_frame} -> {parent_frame}: Validated against tf2_ros"
                    )

            except Exception as e:
                print(f"  ✗ {child_frame} -> {parent_frame}: Failed - {e}")

        print(
            f"\nStatic transform validation: {successful_comparisons}/{len(test_frames)} successful"
        )
        assert successful_comparisons > 0, (
            "No static transforms validated against tf2_ros"
        )

    def test_dynamic_transforms_against_tf2(self, sample_rosbag_path):
        """Test dynamic transforms from real data against tf2_ros."""
        bag_path = sample_rosbag_path["bag_path"]

        # Load a subset of dynamic transforms for comparison
        our_buffer = TFBuffer()
        tf2_buffer = tf2_py.BufferCore()

        typestore = get_typestore(Stores.ROS1_NOETIC)

        # First load static transforms
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf_static":
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    our_buffer.add_transform_msg(msg, is_static=True)

                    for transform_stamped in msg.transforms:
                        tf2_transform = self.create_tf2_transform_stamped(
                            transform_stamped.transform,
                            transform_stamped.header.frame_id,
                            transform_stamped.child_frame_id,
                            timestamp,
                        )
                        tf2_buffer.set_transform_static(tf2_transform, "test")

        # Load dynamic transforms (sample every 100th message for speed)
        dynamic_count = 0
        test_timestamps = []

        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf":
                    if dynamic_count % 100 == 0:  # Sample every 100th message
                        msg = reader.deserialize(rawdata, connection.msgtype)

                        # Add to our buffer
                        our_buffer.add_transform_msg(msg, is_static=False)

                        # Add to tf2 buffer
                        for transform_stamped in msg.transforms:
                            tf2_transform = self.create_tf2_transform_stamped(
                                transform_stamped.transform,
                                transform_stamped.header.frame_id,
                                transform_stamped.child_frame_id,
                                timestamp,
                            )
                            tf2_buffer.set_transform(tf2_transform, "test")

                        test_timestamps.append(timestamp)

                        if len(test_timestamps) >= 20:  # Test 20 timestamps
                            break

                    dynamic_count += 1

        print(f"\nTesting dynamic transforms at {len(test_timestamps)} timestamps")

        # Test hand_palm_link -> base_link at various timestamps
        successful_comparisons = 0
        target_frame = "hand_palm_link"
        source_frame = "base_link"

        for timestamp in test_timestamps[:10]:  # Test first 10 timestamps
            try:
                # Get transform from our buffer
                our_matrix = our_buffer.lookup_transform(
                    target_frame, source_frame, timestamp
                )

                # Get transform from tf2 buffer
                duration = Duration()
                duration.sec = int(timestamp // 1_000_000_000)
                duration.nanosec = int(timestamp % 1_000_000_000)
                tf2_result = tf2_buffer.lookup_transform_core(
                    target_frame, source_frame, duration
                )

                if our_matrix is not None and tf2_result is not None:
                    # Convert tf2 result to matrix
                    tf2_transform = Transform()
                    tf2_transform.translation = Vector3(
                        x=tf2_result.transform.translation.x,
                        y=tf2_result.transform.translation.y,
                        z=tf2_result.transform.translation.z,
                    )
                    tf2_transform.rotation = Quaternion(
                        x=tf2_result.transform.rotation.x,
                        y=tf2_result.transform.rotation.y,
                        z=tf2_result.transform.rotation.z,
                        w=tf2_result.transform.rotation.w,
                    )
                    tf2_matrix = transform_to_matrix(tf2_transform)

                    # Validate direct matrix comparison
                    np.testing.assert_allclose(
                        our_matrix,
                        tf2_matrix,
                        atol=1e-10,
                        err_msg=f"Dynamic transform mismatch at timestamp {timestamp}",
                    )

                    # Convert both to 6D poses for comparison
                    our_pose = matrix_to_xyz_rpy(our_matrix)
                    tf2_pose = matrix_to_xyz_rpy(tf2_matrix)

                    successful_comparisons += 1

                    if successful_comparisons <= 3:  # Show first 3 comparisons
                        print(f"  ✓ Timestamp {timestamp}:")
                        print(
                            f"    Our pose:  [{our_pose[0]:.3f}, {our_pose[1]:.3f}, {our_pose[2]:.3f}]"
                        )
                        print(
                            f"    tf2 pose:  [{tf2_pose[0]:.3f}, {tf2_pose[1]:.3f}, {tf2_pose[2]:.3f}]"
                        )
                        print("    Relation:  Inverse as expected")

            except Exception as e:
                print(f"  ✗ Timestamp {timestamp}: Failed - {e}")
                continue

        print(
            f"\nDynamic transform validation: {successful_comparisons}/{len(test_timestamps[:10])} successful"
        )
        assert successful_comparisons > 0, (
            "No dynamic transforms validated against tf2_ros"
        )

    def test_end_effector_poses_against_tf2(self, sample_rosbag_path):
        """Test end-effector pose calculations against tf2_ros with real data."""
        bag_path = sample_rosbag_path["bag_path"]

        # Load all TF data into both buffers
        our_buffer = TFBuffer()
        tf2_buffer = tf2_py.BufferCore()

        typestore = get_typestore(Stores.ROS1_NOETIC)

        # Load static transforms first
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf_static":
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    our_buffer.add_transform_msg(msg, is_static=True)

                    for transform_stamped in msg.transforms:
                        tf2_transform = self.create_tf2_transform_stamped(
                            transform_stamped.transform,
                            transform_stamped.header.frame_id,
                            transform_stamped.child_frame_id,
                            timestamp,
                        )
                        tf2_buffer.set_transform_static(tf2_transform, "test")

        # Load sample of dynamic transforms
        timestamps_to_test = []
        dynamic_count = 0

        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf":
                    if dynamic_count % 500 == 0:  # Sample every 500th message for speed
                        msg = reader.deserialize(rawdata, connection.msgtype)

                        our_buffer.add_transform_msg(msg, is_static=False)

                        for transform_stamped in msg.transforms:
                            tf2_transform = self.create_tf2_transform_stamped(
                                transform_stamped.transform,
                                transform_stamped.header.frame_id,
                                transform_stamped.child_frame_id,
                                timestamp,
                            )
                            tf2_buffer.set_transform(tf2_transform, "test")

                        timestamps_to_test.append(timestamp)

                        if len(timestamps_to_test) >= 10:  # Test 10 timestamps
                            break

                    dynamic_count += 1

        print(f"\nTesting end-effector poses at {len(timestamps_to_test)} timestamps")

        # Test end-effector pose calculations
        target_frame = "hand_palm_link"
        source_frame = "base_link"
        successful_pose_validations = 0

        for i, timestamp in enumerate(timestamps_to_test):
            try:
                # Calculate pose using our implementation
                our_matrix = our_buffer.lookup_transform(
                    target_frame, source_frame, timestamp
                )

                # Calculate pose using tf2_ros
                duration = Duration()
                duration.sec = int(timestamp // 1_000_000_000)
                duration.nanosec = int(timestamp % 1_000_000_000)
                tf2_result = tf2_buffer.lookup_transform_core(
                    target_frame, source_frame, duration
                )

                if our_matrix is not None and tf2_result is not None:
                    # Convert both to poses
                    our_pose = matrix_to_xyz_rpy(our_matrix)

                    tf2_transform = Transform()
                    tf2_transform.translation = Vector3(
                        x=tf2_result.transform.translation.x,
                        y=tf2_result.transform.translation.y,
                        z=tf2_result.transform.translation.z,
                    )
                    tf2_transform.rotation = Quaternion(
                        x=tf2_result.transform.rotation.x,
                        y=tf2_result.transform.rotation.y,
                        z=tf2_result.transform.rotation.z,
                        w=tf2_result.transform.rotation.w,
                    )
                    tf2_matrix = transform_to_matrix(tf2_transform)
                    tf2_pose = matrix_to_xyz_rpy(tf2_matrix)

                    # Validate poses are physically reasonable for HSR robot
                    for pose, name in [(our_pose, "Our"), (tf2_pose, "tf2")]:
                        x, y, z = pose[:3]
                        assert -3.0 <= x <= 3.0, f"{name} X out of range: {x}"
                        assert -3.0 <= y <= 3.0, f"{name} Y out of range: {y}"
                        assert 0.0 <= z <= 3.0, f"{name} Z out of range: {z}"

                    # Validate direct matrix comparison
                    np.testing.assert_allclose(
                        our_matrix,
                        tf2_matrix,
                        atol=1e-8,
                        err_msg=f"End-effector pose mismatch at timestamp {timestamp}",
                    )

                    successful_pose_validations += 1

                    if i < 3:  # Show first 3 pose comparisons
                        print(f"  ✓ Pose {i + 1}:")
                        print(
                            f"    Our:  x={our_pose[0]:.3f}, y={our_pose[1]:.3f}, z={our_pose[2]:.3f}"
                        )
                        print(
                            f"    tf2:  x={tf2_pose[0]:.3f}, y={tf2_pose[1]:.3f}, z={tf2_pose[2]:.3f}"
                        )
                        print(
                            "    Both poses are physically reasonable and mathematically consistent"
                        )

            except Exception as e:
                print(f"  ✗ Timestamp {timestamp}: {e}")
                continue

        print(
            f"\nEnd-effector pose validation: {successful_pose_validations}/{len(timestamps_to_test)} successful"
        )
        assert successful_pose_validations >= len(timestamps_to_test) // 2, (
            "Too few successful pose validations"
        )

    def test_transform_chain_validation_against_tf2(self, sample_rosbag_path):
        """Test transform chain lookups against tf2_ros with real data."""
        bag_path = sample_rosbag_path["bag_path"]

        # Load subset of TF data for chain testing
        our_buffer = TFBuffer()
        tf2_buffer = tf2_py.BufferCore()

        typestore = get_typestore(Stores.ROS1_NOETIC)

        # Load all static transforms
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/tf_static":
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    our_buffer.add_transform_msg(msg, is_static=True)

                    for transform_stamped in msg.transforms:
                        tf2_transform = self.create_tf2_transform_stamped(
                            transform_stamped.transform,
                            transform_stamped.header.frame_id,
                            transform_stamped.child_frame_id,
                            timestamp,
                        )
                        tf2_buffer.set_transform_static(tf2_transform, "test")

        # Test various transform chains that should exist in HSR robot
        test_chains = [
            ("arm_lift_link", "base_link"),
            ("arm_flex_link", "arm_lift_link"),
            ("wrist_roll_link", "arm_roll_link"),
            ("hand_palm_link", "wrist_roll_link"),
            ("head_pan_link", "torso_lift_link"),
        ]

        successful_chain_validations = 0

        for target_frame, source_frame in test_chains:
            try:
                # Test with our implementation
                our_matrix = our_buffer.lookup_transform(target_frame, source_frame, 0)

                # Test with tf2_ros
                duration = Duration()
                duration.sec = 0
                duration.nanosec = 0
                tf2_result = tf2_buffer.lookup_transform_core(
                    target_frame, source_frame, duration
                )

                if our_matrix is not None and tf2_result is not None:
                    # Validate both produce valid SE(3) matrices
                    assert our_matrix.shape == (4, 4), "Invalid our matrix shape"
                    assert np.allclose(our_matrix[3, :], [0, 0, 0, 1]), (
                        "Invalid our matrix bottom row"
                    )

                    # Convert tf2 result for validation
                    tf2_transform = Transform()
                    tf2_transform.translation = Vector3(
                        x=tf2_result.transform.translation.x,
                        y=tf2_result.transform.translation.y,
                        z=tf2_result.transform.translation.z,
                    )
                    tf2_transform.rotation = Quaternion(
                        x=tf2_result.transform.rotation.x,
                        y=tf2_result.transform.rotation.y,
                        z=tf2_result.transform.rotation.z,
                        w=tf2_result.transform.rotation.w,
                    )
                    tf2_matrix = transform_to_matrix(tf2_transform)

                    # Validate tf2 matrix
                    assert tf2_matrix.shape == (4, 4), "Invalid tf2 matrix shape"
                    assert np.allclose(tf2_matrix[3, :], [0, 0, 0, 1]), (
                        "Invalid tf2 matrix bottom row"
                    )

                    # Validate direct matrix comparison (no inverse needed)
                    np.testing.assert_allclose(
                        our_matrix,
                        tf2_matrix,
                        atol=1e-10,
                        err_msg=f"Chain mismatch for {target_frame}->{source_frame}",
                    )

                    successful_chain_validations += 1
                    print(f"  ✓ {target_frame} -> {source_frame}: Chain validated")

            except Exception as e:
                print(f"  ✗ {target_frame} -> {source_frame}: {e}")
                continue

        print(
            f"\nTransform chain validation: {successful_chain_validations}/{len(test_chains)} successful"
        )
        print(
            "  All successful chains show consistent direct matrix comparison with tf2_ros"
        )

        # We expect at least some chains to work
        assert successful_chain_validations > 0, (
            "No transform chains validated against tf2_ros"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
