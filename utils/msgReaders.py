import rosbag
import numpy as np
import utm
import time
import math
import cv2
import os
import transformation as trans
from transformations import euler_from_quaternion
from std_msgs.msg import Int32, String
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree


def print_bag_topics(bag):
    topics = bag.get_type_and_topic_info()[1].keys()
    print("Topics: ", topics)


def camera_info_msg(bag, topic):
    """read camera info msg from bag"""
    # only read the first one published
    # return camera info in  dictonary, follow this ros cam_info msg definition, http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    print("Topics: ", topic)

    for topic, msg, t in bag.read_messages(topics=topic):
        camera_info = {
            "height": msg.height,
            "width": msg.width,
            "distortion_model": msg.distortion_model,
            "D": msg.D,
            "K": msg.K,
            "R": msg.R,
            "P": msg.P,
        }

        break

    return camera_info


def odom_msg(bag, topic, bag_time=False):
    """read gps msg from bag"""
    # return poses: N * (time, x, y, z, roll, pitch, yaw)
    print("Topics: ", topic)

    poses = []
    velocities = []
    if bag_time:
        print("using bag time!!")
    for topic, msg, t in bag.read_messages(topics=topic):
        if bag_time:
            timestamp = t.to_sec()
        else:
            timestamp = (
                msg.header.stamp.secs + float(msg.header.stamp.nsecs) * 1e-9
            )  # this is the true time, it is different from bag t.to_time()
        euler = euler_from_quaternion(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ]
        )
        poses.append(
            [
                timestamp,
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                euler[0],
                euler[1],
                euler[2],
            ]
        )
        # velocities =

    # convert to numpy
    poses = np.asarray(poses)
    print("poses", poses.shape)

    return poses


def control_cmd_msg(bag, topic, bag_time=False):
    # read INS
    control_cmd_msg = general_msg_reader(bag, topic)

    control_cmds = []

    for i in range(len(control_cmd_msg)):
        t = control_cmd_msg[i][0].to_sec()
        velocity = control_cmd_msg[i][1].linear_speed
        steering_angle = control_cmd_msg[i][1].steering_angle
        steering_rate = control_cmd_msg[i][1].steering_rate
        angular_speed = control_cmd_msg[i][1].angular_speed

        control_cmds.append([t, velocity, steering_angle, steering_rate, angular_speed])

    control_cmds_array = np.array(control_cmds)

    print("control commands: t, v, gamma, gamma_dot, omega")
    print("control commands shape: ", control_cmds_array.shape)

    return control_cmds_array


def pid_msg(bag, topic, bag_time=False):
    # read INS
    pid_msgs = general_msg_reader(bag, topic)

    pid_states = []

    for i in range(len(pid_msgs)):
        t = pid_msgs[i][0].to_sec()
        k_p = pid_msgs[i][1].k_p
        k_i = pid_msgs[i][1].k_i
        k_d = pid_msgs[i][1].k_d
        error = pid_msgs[i][1].error
        error_i = pid_msgs[i][1].error_i
        error_d = pid_msgs[i][1].error_d
        target = pid_msgs[i][1].target
        current = pid_msgs[i][1].current
        output = pid_msgs[i][1].output

        pid_states.append(
            [t, k_p, k_i, k_d, error, error_i, error_d, target, current, output]
        )

    pid_states_array = np.array(pid_states)

    print(
        "pid states: t, k_p, k_i, k_d, error, error_i, error_d, target, current, output"
    )
    print("control commands shape: ", pid_states_array.shape)

    return pid_states_array


def ins_msg(bag, topic, bag_time=False):
    # read INS
    ins_msg = general_msg_reader(bag, topic)

    ins_data = []
    ins_covirance_data = []
    for i in range(len(ins_msg)):
        t = ins_msg[i][0].to_sec()
        v = ins_msg[i][1].velocity
        lat = ins_msg[i][1].latitude
        lon = ins_msg[i][1].longitude
        alt = ins_msg[i][1].altitude
        v_e = ins_msg[i][1].velocity_east
        v_n = ins_msg[i][1].velocity_north
        v_u = ins_msg[i][1].velocity_up
        heading = ins_msg[i][1].heading
        sat_status = ins_msg[i][1].sat_status
        sys_status = ins_msg[i][1].sys_status
        ins_covariance = ins_msg[i][1].position_covariance
        east, north, _, _ = utm.from_latlon(lat, lon)
        ins_data.append(
            [
                t,
                lat,
                lon,
                alt,
                v_e,
                v_n,
                v_u,
                v,
                sat_status,
                sys_status,
                heading,
                east,
                north,
            ]
        )
        ins_covirance_data.append(ins_covariance)
    ins_covirance_data = np.array(ins_covirance_data)
    ins_data = np.array(ins_data)
    print(
        "ins data: t, lat, lon, alt, v_e, v_n, v_u, v, sat_status, sys_status, heading, east, north"
    )
    print("ins_data shape: ", ins_data.shape)

    return ins_data, ins_covirance_data


def general_msg_reader(bag, topic):
    """read any msg into [n*[t, msg]] list"""
    print("Topics: ", topic)
    msgs = []
    for topic, msg, t in bag.read_messages(topics=topic):
        msgs.append([t, msg])
    return msgs


def imu_msg(bag, topics="/imu/imu"):
    """read imu ypr msg from bag"""
    imu_yprs, imu_gyroscope, imu_accelerometer = [], [], []
    timestamps = []
    for topic, msg, t in bag.read_messages(topics="/imu/ypr"):
        imu_yprs.append([msg.x, msg.y, msg.z])

    for topic, msg, t in bag.read_messages(topics=topics):
        imu_gyroscope.append(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )
        imu_accelerometer.append(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        timestamp = (
            msg.header.stamp.secs + float(msg.header.stamp.nsecs) * 1e-9
        )  # this is the true time, it is different from bag t.to_time()

        timestamps.append(timestamp)

    # convert to numpy
    imu_yprs = np.asarray(imu_yprs)
    imu_gyroscope = np.asarray(imu_gyroscope)
    imu_accelerometer = np.asarray(imu_accelerometer)
    timestamps = np.asarray(timestamps)

    print("imu_yprs", imu_yprs.shape)
    print("imu_gyroscope", imu_gyroscope.shape)
    print("imu_accelerometer", imu_accelerometer.shape)
    print("timestamps", timestamps.shape)

    return imu_yprs, imu_gyroscope, imu_accelerometer, timestamps


def navSatFix_msg_to_utm(bag, topic):
    """read gps msg from bag"""
    print("topic: ", topic)
    utm_gps = []
    for topic, msg, t in bag.read_messages(topics=topic):
        timestamp = (
            msg.header.stamp.secs + float(msg.header.stamp.nsecs) * 1e-9
        )  # this is the true time, it is different from bag t.to_time()
        utm_coord = utm.from_latlon(msg.latitude, msg.longitude)
        utm_gps.append([timestamp, utm_coord[0], utm_coord[1]])

    # convert to numpy
    utm_gps = np.asarray(utm_gps)
    print("utm_gps", utm_gps.shape)

    return utm_gps


# Image extraction not working on python 3
try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:
    print(
        "[WARNING] cv_bridge not working, image_msg reader is not imported, \
            make sure you are using python 2, installed ros and has ros python path in your python path"
    )
else:

    def image_msg(bag, topic, save_dir):
        """read and save image msg"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Save directory: ", save_dir)
        print("Topics: ", topic)
        bridge = CvBridge()
        for topic, msg, t in bag.read_messages(topics=topic):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                print(e)
            timestamp = (
                msg.header.stamp.secs + float(msg.header.stamp.nsecs) * 1e-9
            )  # this is the true time, it is different from bag t.to_time()
            filename = save_dir + "/" + "%.9f" % timestamp + ".png"
            print("filename: ", filename)
            cv2.imwrite(filename, cv_image)

    def read_nth_msg(n_idx, bag, topic, timelist):
        msg_time = timelist[n_idx]
        retrived_msg = None
        for topic, msg, t in bag.read_messages(topics=topic, start_time=msg_time):
            retrived_msg = msg
            break
        return retrived_msg

    def create_timelist(bag, topic):
        time_list = []
        row_time_list = []
        i = 0
        for topic, msg, t in bag.read_messages(topics=topic):
            time_list.append([t.to_sec(), i])
            row_time_list.append(t)
            i += 1
        print("%s: %i " % (topic, len(time_list)))
        time_list = np.asarray(time_list)
        return time_list, row_time_list

    def retrive_image(
        n_idx,
        bag,
        timelist_ros_synced,
        topic,
        desired_encoding="passthrough",
        compressed=False,
    ):
        # Change the encoding to desired encoding if the passthrough not working well.
        # Encoding: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        # read the nth msg
        img_msg = read_nth_msg(n_idx, bag, topic, timelist_ros_synced)
        # msg to image
        bridge = CvBridge()
        if compressed:
            img = bridge.compressed_imgmsg_to_cv2(img_msg)
        else:
            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding=desired_encoding)
        return img

    # get synchronized multiple msgs
    def sync_msgs(msgs, dt_threshold=None):
        # sync multiple msgs in msgs list []
        # msg should be a numpy array of size (N, data), timestamps should be the first dimension of the msgs
        # synchronization will based on the first msg in the list
        print("msgs to sync: ", len(msgs))

        if dt_threshold is None:
            # if dt is not set, dt will be the average period of the first msg
            msg_t = msgs[0][:, 0]
            dt_threshold = (msg_t[-1] - msg_t[1]) / len(msg_t)

        print("dt threshold: ", dt_threshold)

        msg1_t = msgs[0][:, 0]

        # timestamp kd of the rest msgs
        timestamps_kd_list = []
        for msg in msgs[1:]:
            timestamps_kd = KDTree(np.asarray(msg[:, 0]).reshape(-1, 1))
            timestamps_kd_list.append(timestamps_kd)

        msgs_idx_synced = []
        for msg1_idx in range(len(msg1_t)):
            msg_idx_list = [msg1_idx]
            dt_valid = True
            for timestamps_kd in timestamps_kd_list:
                dt, msg_idx = timestamps_kd.query([msg1_t[msg1_idx]])
                if abs(dt) > dt_threshold:
                    dt_valid = False
                    break
                msg_idx_list.append(msg_idx)

            if dt_valid:
                msgs_idx_synced.append(msg_idx_list)

        msgs_idx_synced = np.asarray(msgs_idx_synced).T
        print("msgs_idx_synced: ", msgs_idx_synced.shape)

        msgs_synced = []
        for i, msg in enumerate(msgs):
            msg_synced = msg[msgs_idx_synced[i]]
            msgs_synced.append(msg_synced)

        return msgs_synced
