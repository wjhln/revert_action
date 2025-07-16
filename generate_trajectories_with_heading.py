import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def generate_trajectory_dataset(num_samples=1000, save_path='trajectory_dataset'):
    """生成带有航向角的轨迹数据集"""
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    all_trajectories = []
    all_headings = []
    all_v0 = []
    all_a0 = []
    trajectory_types = []
    
    for i in range(num_samples):
        # 生成换道轨迹
        t = torch.linspace(0, 5, 18)
        x = t
        y = 1.5 * torch.tanh((t - 2.5) * 1.5)
        
        # 计算航向角（使用差分近似）
        dx = torch.diff(x, prepend=x[0].unsqueeze(0))
        dy = torch.diff(y, prepend=y[0].unsqueeze(0))
        heading = torch.atan2(dy, dx)
        
        traj = torch.stack([x, y], dim=-1)
        
        # 添加随机缩放和旋转
        scale = 0.8 + 0.4 * torch.rand(1)
        angle = torch.rand(1) * 2 * np.pi
        
        # 应用变换
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        
        traj = traj * scale
        traj = torch.matmul(traj, rotation_matrix.T)
        heading = heading + angle  # 更新航向角
        
        # 将起始点移动到原点
        start_point = traj[0]
        traj = traj - start_point
        
        all_trajectories.append(traj)
        all_headings.append(heading)
        
        # 生成初始状态
        v0 = 2.0 + torch.rand(1) * 2.0  # 2-4 m/s
        a0 = torch.zeros(1)
        
        all_v0.append(v0)
        all_a0.append(a0)
        trajectory_types.append('lane_change')
    
    # 保存为CSV格式
    for idx, (traj, heading, v0, a0) in enumerate(zip(all_trajectories, all_headings, all_v0, all_a0)):
        # 计算实际的速度和加速度
        dt = 5.0 / 17  # 时间间隔
        
        # 计算速度（位置的一阶导数）
        velocity_x = torch.diff(traj[:, 0], prepend=traj[0, 0].unsqueeze(0)) / dt
        velocity_y = torch.diff(traj[:, 1], prepend=traj[0, 1].unsqueeze(0)) / dt
        velocity = torch.sqrt(velocity_x**2 + velocity_y**2)
        
        # 计算加速度（速度的一阶导数）
        acceleration_x = torch.diff(velocity_x, prepend=velocity_x[0].unsqueeze(0)) / dt
        acceleration_y = torch.diff(velocity_y, prepend=velocity_y[0].unsqueeze(0)) / dt
        acceleration = torch.sqrt(acceleration_x**2 + acceleration_y**2)
        
        # 添加一些随机变化使速度更真实
        velocity_variation = 0.1 * torch.randn_like(velocity)
        velocity = torch.clamp(velocity + velocity_variation, min=0.5, max=8.0)
        
        data = {
            'timestamp': np.linspace(0, 5, 18),
            'x': traj[:, 0].numpy(),
            'y': traj[:, 1].numpy(),
            'heading': heading.numpy(),
            'velocity': velocity.numpy(),
            'acceleration': acceleration.numpy()
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_path, f'trajectory_{idx:04d}.csv'), index=False)
    
    return all_trajectories, all_headings, all_v0, all_a0, trajectory_types

def visualize_dataset(trajectories, headings, types, num_samples=8):
    """可视化带有航向角的轨迹"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(trajectories))):
        ax = axes[i]
        traj = trajectories[i].numpy()
        heading = headings[i].numpy()
        
        # 绘制轨迹
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        
        # 每隔几个点绘制航向箭头
        step = 3
        for j in range(0, len(traj), step):
            ax.arrow(traj[j, 0], traj[j, 1],
                    0.2 * np.cos(heading[j]), 0.2 * np.sin(heading[j]),
                    head_width=0.1, head_length=0.15, fc='r', ec='r')
        
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=8, label='End')
        
        ax.set_title(f'{types[i]}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_with_heading.png', dpi=150, bbox_inches='tight')

def visualize_velocity_acceleration(trajectories, headings, types, num_samples=4):
    """可视化速度和加速度变化"""
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        heading = headings[i]
        
        # 计算速度和加速度
        dt = 5.0 / 17
        velocity_x = torch.diff(traj[:, 0], prepend=traj[0, 0].unsqueeze(0)) / dt
        velocity_y = torch.diff(traj[:, 1], prepend=traj[0, 1].unsqueeze(0)) / dt
        velocity = torch.sqrt(velocity_x**2 + velocity_y**2)
        
        acceleration_x = torch.diff(velocity_x, prepend=velocity_x[0].unsqueeze(0)) / dt
        acceleration_y = torch.diff(velocity_y, prepend=velocity_y[0].unsqueeze(0)) / dt
        acceleration = torch.sqrt(acceleration_x**2 + acceleration_y**2)
        
        # 添加随机变化
        velocity_variation = 0.1 * torch.randn_like(velocity)
        velocity = torch.clamp(velocity + velocity_variation, min=0.5, max=8.0)
        
        time = np.linspace(0, 5, 18)
        
        # 绘制速度
        axes[0, i].plot(time, velocity.numpy(), 'b-', linewidth=2)
        axes[0, i].set_title(f'{types[i]} - 速度')
        axes[0, i].set_ylabel('速度 (m/s)')
        axes[0, i].grid(True, alpha=0.3)
        
        # 绘制加速度
        axes[1, i].plot(time, acceleration.numpy(), 'r-', linewidth=2)
        axes[1, i].set_title(f'{types[i]} - 加速度')
        axes[1, i].set_xlabel('时间 (s)')
        axes[1, i].set_ylabel('加速度 (m/s²)')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_acceleration.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    # 生成数据集
    trajectories, headings, v0, a0, types = generate_trajectory_dataset(num_samples=15)
    
    # 可视化轨迹
    visualize_dataset(trajectories, headings, types)
    
    # 可视化速度和加速度
    visualize_velocity_acceleration(trajectories, headings, types)
    
    print("数据集生成完成！")
    print("文件保存在: trajectory_dataset/")
    print("轨迹图保存为: trajectory_with_heading.png")
    print("速度加速度图保存为: velocity_acceleration.png")
