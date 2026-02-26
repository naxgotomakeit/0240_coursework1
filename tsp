#!/usr/bin/env python3

# ==============================================================================
# MASTER'S NOTE: 
# This script has been upgraded with Heuristic TSP and Obstacle Avoidance Penalty.
# It integrates a planning phase before the flight phase and includes comprehensive 
# mission logging.
# ==============================================================================

import argparse
from time import sleep
import time
import yaml
import numpy as np

# 导入 TSP 求解器
from python_tsp.heuristics import solve_tsp_local_search
from as2_python_api.drone_interface import DroneInterface
import rclpy

TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s

# ==========================================================
# 🧠 大脑模块：算法与数学逻辑 (Algorithm & Math Logic)
# ==========================================================

def extract_bounds(obstacles_dict):
    """提取障碍物的 3D 边界框，并加上 0.3m 的安全膨胀"""
    bounds = []
    if not obstacles_dict:
        return bounds
    for obs_id, obs in obstacles_dict.items():
        half_w, half_d, half_h = obs['w'] / 2.0, obs['d'] / 2.0, obs['h'] / 2.0
        # 加上 0.3m 的安全边界防止擦碰
        min_bound = np.array([obs['x'] - half_w, obs['y'] - half_d, obs['z'] - half_h]) - 0.3
        max_bound = np.array([obs['x'] + half_w, obs['y'] + half_d, obs['z'] + half_h]) + 0.3
        bounds.append((min_bound, max_bound))
    return bounds

def check_collision(p1, p2, obstacle_bounds, num_samples=50):
    """矢量化检测两点之间的连线是否穿过任何障碍物"""
    if not obstacle_bounds:
        return False
    t = np.linspace(0, 1, num_samples)[:, np.newaxis] 
    segment_points = p1 + t * (p2 - p1)
    for min_b, max_b in obstacle_bounds:
        inside_x = (segment_points[:, 0] >= min_b[0]) & (segment_points[:, 0] <= max_b[0])
        inside_y = (segment_points[:, 1] >= min_b[1]) & (segment_points[:, 1] <= max_b[1])
        inside_z = (segment_points[:, 2] >= min_b[2]) & (segment_points[:, 2] <= max_b[2])
        if np.any(inside_x & inside_y & inside_z):
            return True 
    return False

def build_cost_matrix(scenario_data):
    """构建 N x N 的代价矩阵"""
    points = []
    start_pose = scenario_data['drone_start_pose']
    points.append([start_pose['x'], start_pose['y'], start_pose['z']])
    
    vp_poses = scenario_data['viewpoint_poses']
    vp_ids = sorted(vp_poses.keys()) 
    
    for vp_id in vp_ids:
        vp = vp_poses[vp_id]
        points.append([vp['x'], vp['y'], vp['z']])
        
    points_array = np.array(points)
    N = len(points_array)
    obstacle_bounds = extract_bounds(scenario_data.get('obstacles', {}))
    
    cost_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j: continue
            dist = np.linalg.norm(points_array[i] - points_array[j])
            # 如果路径穿过障碍物，施加巨大的惩罚
            if check_collision(points_array[i], points_array[j], obstacle_bounds):
                cost_matrix[i][j] = dist * 100.0  
            else:
                cost_matrix[i][j] = dist
                
    return cost_matrix, vp_ids, points_array

# ==========================================================
# 🚁 物理执行模块：无人机控制逻辑 (Drone Control Logic)
# ==========================================================

def drone_start(drone_interface: DroneInterface) -> bool:
    print('[LOG] 🛫 Start mission sequence...')
    success = drone_interface.arm()
    success = success and drone_interface.offboard()
    success = success and drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'[LOG] Take Off status: {"Success" if success else "Failed"}')
    return success

def drone_run(drone_interface: DroneInterface, scenario: dict, metrics: dict) -> bool:
    print('\n[LOG] === 🧠 PHASE 1: INTELLIGENT PATH PLANNING ===')
    
    # 1. 算法规划阶段
    start_algo_time = time.time()
    matrix, original_vp_ids, points_array = build_cost_matrix(scenario)
    route_indices, total_cost = solve_tsp_local_search(matrix)
    algo_time = time.time() - start_algo_time
    
    # 2. 翻译路线并计算纯物理距离（不含惩罚）
    optimized_vpids = []
    planned_distance = 0.0
    
    for i in range(len(route_indices) - 1):
        idx_current = route_indices[i]
        idx_next = route_indices[i+1]
        planned_distance += np.linalg.norm(points_array[idx_current] - points_array[idx_next])

    for idx in route_indices:
        if idx != 0:
            optimized_vpids.append(original_vp_ids[idx - 1])

    # 记录指标
    metrics['algo_time'] = algo_time
    metrics['planned_distance'] = planned_distance
    metrics['visit_order'] = optimized_vpids

    print(f'[LOG] Algorithm Computation Time: {algo_time:.4f} seconds')
    print(f'[LOG] Theoretical Distance (w/o penalty): {planned_distance:.2f} meters')
    print(f'[LOG] Optimal Visit Order: {" -> ".join(map(str, optimized_vpids))}')
    
    print('\n[LOG] === 🚁 PHASE 2: MISSION EXECUTION ===')
    
    # 3. 物理执行阶段
    for vpid in optimized_vpids:
        vp = scenario["viewpoint_poses"][vpid]
        print(f'[LOG] >>> Flying to Viewpoint {vpid} (Pose: x={vp["x"]:.2f}, y={vp["y"]:.2f}, yaw={vp["w"]:.2f})')
        goal = [vp["x"], vp["y"], vp["z"]]
        
        # 飞向目标点
        success = drone_interface.go_to.go_to_point_with_yaw(goal, angle=vp["w"], speed=SPEED)
        
        if not success:
            print(f'[ERROR] ❌ Failed to reach Viewpoint {vpid}!')
            return False
            
        print(f'[LOG] ✅ Reached Viewpoint {vpid}. (Simulating Camera Trigger...)')
        sleep(SLEEP_TIME)

    return True

def drone_end(drone_interface: DroneInterface) -> bool:
    print('\n[LOG] 🛬 Ending mission...')
    success = drone_interface.land(speed=LAND_SPEED)
    success = success and drone_interface.manual()
    print(f'[LOG] Land status: {"Success" if success else "Failed"}')
    return success

def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# ==========================================================
# 📊 入口与日志报告系统 (Main & Reporting)
# ==========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intelligent TSP Drone Mission')
    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace', type=str, default='drone0', help='ID of the drone')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose output')
    parser.add_argument('-s', '--use_sim_time', action='store_true', default=True, help='Use sim time')

    args = parser.parse_args()
    
    print(f"==================================================")
    print(f"🚀 INITIALIZING MISSION SYSTEM")
    print(f"Scenario: {args.scenario}")
    print(f"Drone ID: {args.namespace}")
    print(f"==================================================")
    
    scenario = read_scenario(args.scenario)
    rclpy.init()

    uav = DroneInterface(drone_id=args.namespace, use_sim_time=args.use_sim_time, verbose=args.verbose)
    
    # 用于收集报告数据的字典
    mission_metrics = {
        'algo_time': 0.0,
        'planned_distance': 0.0,
        'visit_order': [],
        'mission_success': False,
        'total_time': 0.0
    }

    start_total_time = time.time()
    
    success = drone_start(uav)
    try:
        if success:
            flight_success = drone_run(uav, scenario, mission_metrics)
            mission_metrics['mission_success'] = flight_success
            
    except KeyboardInterrupt:
        print("\n[WARNING] Mission interrupted by user!")
        
    finally:
        drone_end(uav)
        
    mission_metrics['total_time'] = time.time() - start_total_time

    uav.shutdown()
    rclpy.shutdown()
    
    # --- 最终交付给客户（老师）的硬核报告 ---
    print("\n==================================================")
    print("📈 MISSION POST-FLIGHT REPORT")
    print("==================================================")
    print(f"Status:          {'✅ SUCCESS' if mission_metrics['mission_success'] else '❌ FAILED'}")
    print(f"Algorithm Time:  {mission_metrics['algo_time']:.4f} seconds")
    print(f"Total Time:      {mission_metrics['total_time']:.2f} seconds (inc. flight & delays)")
    print(f"Flight Distance: {mission_metrics['planned_distance']:.2f} meters (Theoretical optimal)")
    print(f"Path Visited:    START -> {' -> '.join(map(str, mission_metrics['visit_order']))}")
    print("==================================================")
    print('Clean exit')
    exit(0)
