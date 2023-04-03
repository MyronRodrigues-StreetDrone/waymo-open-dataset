import rosbag2_py
from rclpy.serialization import serialize_message
import std_msgs.msg as std_msgs
import numpy as np
from builtin_interfaces.msg import Time
import ros2_numpy as rnpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TFMessage
from pyquaternion import Quaternion
from pathlib import Path
from waymo_open_dataset import v2
import tensorflow as tf
import dask.dataframe as dd
import argparse

import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

class RosbagWriter:
    def __init__(self, topic_msg_type_map: dict, output_bag_path: str, serialization='cdr') -> None:
        self.writer = rosbag2_py.SequentialWriter()
        self.open_bag_for_writing(output_bag_path, serialization)
        for topic, msg_type in topic_msg_type_map.items():
            self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
                name=topic,
                type=msg_type,
                serialization_format=serialization))
    
    def open_bag_for_writing(self, path, serialization_format='cdr'):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format=serialization_format,
            output_serialization_format=serialization_format)

        self.writer.open(storage_options, converter_options)
    
    def write(self, topic: str, ros2_msg: any, timestamp_ns: int):
        self.writer.write(topic, serialize_message(ros2_msg), timestamp_ns)

def time_msg_from_timestamp_us(timestamp_us: int):
    return Time(sec=int(timestamp_us/1e6), nanosec=int((timestamp_us%1e6)*1e3))


def lidar_range_img_to_pcd_xyzi_msg(range_img, lidar_calib, lidar_frame: str, time_msg: Time):
    pc = v2.convert_range_image_to_point_cloud(range_img, lidar_calib, keep_polar_features=True)
    pc = np.rec.fromarrays(tf.transpose(tf.gather(pc, [3,4,5,1], axis=1)),
                           dtype={'names': ('x', 'y', 'z', 'i'),
                                  'formats': (np.float32, np.float32, np.float32, np.float32)})
    return rnpy.array_to_pointcloud2(pc, time_msg, lidar_frame)


def make_transform_msg(tf_matrix, from_frame_id: str, to_frame_id: str, time_msg: Time):
  t = TransformStamped()

  t.header.stamp = time_msg
  t.header.frame_id = from_frame_id
  t.child_frame_id = to_frame_id
  t.transform.translation.x = float(tf_matrix[0, 3])
  t.transform.translation.y = float(tf_matrix[1, 3])
  t.transform.translation.z = float(tf_matrix[2, 3])
  quat = Quaternion(matrix=tf_matrix)
  t.transform.rotation.x = float(quat.x)
  t.transform.rotation.y = float(quat.y)
  t.transform.rotation.z = float(quat.z)
  t.transform.rotation.w = float(quat.w)
  return t

def get_args():
  parser = argparse.ArgumentParser(
                    prog='waymo_od2_to_rosbag',
                    description='Waymo open dataset 2 to rosbag converter')
  parser.add_argument('filename')           # positional argument
  parser.add_argument('-c', '--count')      # option that takes a value
  parser.add_argument('-v', '--verbose',
                      action='store_true')  # on/off flag

def merge_frames(lidar_df, lidar_calib_df, vehicle_pose_df):
  lidar_lcalib_df = v2.merge(lidar_df, lidar_calib_df, right_group=True)
  return v2.merge(vehicle_pose_df, lidar_lcalib_df, right_group=False)

def make_rosbag2(dataframe, bag_out_path,
                 pointcloud_topic='points', lidar_frame = '/lidar',
                 map_frame='mape', vehicle_frame='base_link', lidar_id=1):
  rb_writer = RosbagWriter({pointcloud_topic: 'sensor_msgs/msg/PointCloud2',
                          '/tf': 'tf2_msgs/msg/TFMessage',
                          '/tf_static': 'tf2_msgs/msg/TFMessage'}, bag_out_path)
  written_static_tf = False
  for _, r in dataframe.iterrows():
    # Create component dataclasses for the raw data
    lidar = v2.LiDARComponent.from_dict(r)
    laser_name = int(lidar.key.laser_name)

    if(laser_name == lidar_id):
      timestamp_us = lidar.key.frame_timestamp_micros
      time_ns = int(timestamp_us*1e3)
      time_msg = time_msg_from_timestamp_us(timestamp_us)
      
      lidar_calib = v2.LiDARCalibrationComponent.from_dict(r)
      vehicle_pose_df = v2.VehiclePoseComponent.from_dict(r)
      vehicle_pose = np.array(vehicle_pose_df.world_from_vehicle.transform).reshape((4,4))
      lidar_pose = np.array(lidar_calib.extrinsic.transform).reshape((4,4))
      
      pcd_msg = lidar_range_img_to_pcd_xyzi_msg(lidar.range_image_return1, lidar_calib, lidar_frame, time_msg)
      rb_writer.write(pointcloud_topic, pcd_msg, time_ns)

      dyn_tf = TFMessage()
      dyn_tf.transforms.append(
        make_transform_msg(np.linalg.inv(vehicle_pose), lidar_frame, map_frame, time_msg))
      rb_writer.write('/tf', dyn_tf, time_ns)
      
      if not written_static_tf:
        static_tf = TFMessage()
        static_tf.transforms.append(
          make_transform_msg(np.linalg.inv(lidar_pose), vehicle_frame, lidar_frame, time_msg))
        rb_writer.write('/tf_static', static_tf, time_ns)
        written_static_tf = True

def main():
  lidar_frame = 'base_link'
  vehicle_frame = '/lidar'
  map_frame = 'map'
  pointcloud_topic = '/points'
  lidar_id = 1
  dataset_dir = Path("/media/myron/715851be-98ca-40b5-b47a-8a3c72b5419d/data/waymo_od2/validation")
  bag_out_path = Path("/media/myron/715851be-98ca-40b5-b47a-8a3c72b5419d/data/waymo_od2/rosbags/validation")

  lidar_dir = dataset_dir/'lidar'
  vehicle_pose_dir = dataset_dir/'vehicle_pose'
  lidar_calib_dir = dataset_dir/'lidar_calibration'

  for lpath in lidar_dir.iterdir():
     if lpath.suffix == '.parquet':
        context = lpath.stem
        print('processing: ', context)
        lidar_df = dd.read_parquet(lpath)
        lidar_calib_df = dd.read_parquet(lidar_calib_dir/(context+'.parquet'))
        vehicle_pose_df = dd.read_parquet(vehicle_pose_dir/(context+'.parquet'))
        lidar_lcalib_vpose_df = merge_frames(lidar_df, lidar_calib_df, vehicle_pose_df)
        make_rosbag2(lidar_lcalib_vpose_df, str(bag_out_path/context), pointcloud_topic, lidar_frame,
                     map_frame, vehicle_frame, lidar_id)
        print('done processing: ', context)


if __name__ == '__main__':
  main()