"""
轨迹优化核心算法验证模块

该模块包含以下核心功能：
1. 轨迹数据加载
2. 基于欧拉积分的轨迹推演
3. 轨迹优化求解
4. 结果可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_trajectory(file_path):
    """加载CSV格式的轨迹数据"""
    df = pd.read_csv(file_path)
    return {
        'x': df['x'].values,
        'y': df['y'].values,
        'heading': df['heading'].values,
        'velocity': df['velocity'].values,
        'acceleration': df['acceleration'].values,
        'dt': df['timestamp'].values[1] - df['timestamp'].values[0]
    }

def euler_integration_predict(x0, controls, dt, L=3.0):
    """基于加速度和转向角的欧拉积分推演函数"""
    N = len(controls)
    states = np.zeros((N + 1, 4))
    states[0] = x0
    
    for k in range(N):
        x_k, y_k, psi_k, v_k = states[k]
        a_k, delta_k = controls[k]
        
        x_next = x_k + v_k * np.cos(psi_k) * dt
        y_next = y_k + v_k * np.sin(psi_k) * dt
        psi_next = psi_k + (v_k / L) * np.tan(delta_k) * dt
        v_next = max(0, v_k + a_k * dt)
        
        states[k + 1] = [x_next, y_next, psi_next, v_next]
    
    return states

def solve_tracking_optimization(traj_data, x0, N, dt, L=3.0):
    """求解轨迹跟踪优化问题"""
    def objective_function(controls):
        controls = controls.reshape(N, 2)
        predicted_states = euler_integration_predict(x0, controls, dt, L)
        
        Q = np.diag([1000.0, 1000.0, 0.0, 0.0])
        total_cost = 0
        
        min_len = min(len(predicted_states), len(traj_data['x']))
        for k in range(min_len):
            pred_state = predicted_states[k]
            ref_state = np.array([traj_data['x'][k], traj_data['y'][k], 
                                 traj_data['heading'][k], traj_data['velocity'][k]])
            error = pred_state - ref_state
            cost = error @ Q @ error
            total_cost += cost
        
        return total_cost
    
    def constraint_function(controls):
        controls = controls.reshape(N, 2)
        constraints = []
        
        for k in range(N):
            # 加速度约束：-2 <= a <= 2
            constraints.append(controls[k, 0] + 4.0)  # a + 2 >= 0
            constraints.append(4.0 - controls[k, 0])  # 2 - a >= 0
            
            # 转向角约束：-0.5 <= delta <= 0.5
            constraints.append(controls[k, 1] + 2.5)  # delta + 0.5 >= 0
            constraints.append(2.5 - controls[k, 1])  # 0.5 - delta >= 0
        
        return np.array(constraints)
    
    # 初始控制序列
    initial_controls = np.zeros((N, 2))
    for k in range(N):
        if k < len(traj_data['acceleration']):
            initial_controls[k, 0] = traj_data['acceleration'][k]
        if k < len(traj_data['heading']) - 1:
            dpsi = traj_data['heading'][k+1] - traj_data['heading'][k]
            v_k = traj_data['velocity'][k]
            initial_controls[k, 1] = np.arctan(dpsi * L / (v_k * dt)) if v_k > 0.1 else 0
    
    # 求解优化问题
    result = minimize(
        objective_function,
        initial_controls.flatten(),
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraint_function},
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    optimal_controls = result.x.reshape(N, 2)
    optimal_states = euler_integration_predict(x0, optimal_controls, dt, L)
    return optimal_states, optimal_controls

def plot_results(traj_data, predicted_states):
    """绘制轨迹对比图并保存"""
    import os
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(traj_data['x'], traj_data['y'], 'b-', label='Original Trajectory', linewidth=2)
    plt.plot(predicted_states[:, 0], predicted_states[:, 1], 'r--', label='Optimized Trajectory', linewidth=2)
    plt.plot(traj_data['x'], traj_data['y'], 'bo', markersize=4, alpha=0.7, label='Original Points')
    plt.plot(predicted_states[:, 0], predicted_states[:, 1], 'ro', markersize=4, alpha=0.7, label='Optimized Points')
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Trajectory Comparison')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    
    save_path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trajectory comparison saved to: {save_path}")

def main():
    """主函数：验证轨迹优化算法"""
    # 加载轨迹数据
    trajectory_file = 'trajectory_dataset/trajectory_0003.csv'
    traj_data = load_trajectory(trajectory_file)
    
    # 设置优化参数
    N = min(30, len(traj_data['x']) - 1)
    x0 = np.array([traj_data['x'][0], traj_data['y'][0], 
                   traj_data['heading'][0], traj_data['velocity'][0]])
    
    print("Starting trajectory optimization...")
    
    # 求解优化问题
    optimal_states, optimal_controls = solve_tracking_optimization(
        traj_data, x0, N, traj_data['dt'])
    
    print("Optimization completed!")
    
    # 绘制结果
    plot_results(traj_data, optimal_states)

if __name__ == '__main__':
    main()
