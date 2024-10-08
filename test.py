import matplotlib.pyplot as plt
import numpy as np

def create_gauge_chart():
    # 设置仪表盘的分区和角度范围
    labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good']
    colors = ['green', 'lightgreen', 'yellow', 'red', 'darkred']
    start_angle = 180
    end_angle = 0
    
    fig, ax = plt.subplots()
    ax.axis('equal')  # 保证饼图是一个圆形

    # 创建分区（半圆）
    wedges, _ = ax.pie(
        [1, 1, 1, 1, 6],
        startangle=start_angle,
        colors=colors,
        counterclock=False,
        wedgeprops={'width': 0.4, 'edgecolor': 'none'}
    )
    
    # 隐藏下半部分的饼图，使其成为半圆
    ax.add_artist(plt.Rectangle((-1, -1), 2, 1, color='white', zorder=3))
    
    # 设置指针角度，和仪表盘的总角度对应
    needle_angle = 135
    needle_length = 0.5
    needle_x = needle_length * np.cos(np.radians(needle_angle))
    needle_y = needle_length * np.sin(np.radians(needle_angle))
    
    ax.arrow(0, 0, needle_x, needle_y, head_width=0.05, head_length=0.1, fc='black', ec='black', linewidth=4)

    # 设置中心的空心圆
    ax.add_artist(plt.Circle((0, 0), 0.1, color='black'))

    # 设置标签
    for i, label in enumerate(labels):
        angle = (start_angle - end_angle) * (i + 0.5) / len(labels) + end_angle
        angle_rad = np.radians(angle)
        ax.text(
            0.75 * np.cos(angle_rad),
            0.75 * np.sin(angle_rad),
            label,
            ha='center', va='center',
            fontsize=10,
            color='black'
        )
    
    plt.show()

create_gauge_chart()