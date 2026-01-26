#!/usr/bin/env python3
"""
测试ZMQ推理服务器的简单客户端
"""
import zmq
import pickle
import numpy as np

def test_server():
    # 连接到服务器
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    print("Connected to ACT inference server on port 5555")
    
    # 测试1: 发送RESET命令
    print("\n--- Test 1: RESET ---")
    socket.send(pickle.dumps('RESET'))
    response = pickle.loads(socket.recv())
    print(f"Response: {response}")
    
    # 测试2: 发送推理请求
    print("\n--- Test 2: Inference ---")
    # 创建模拟数据 (qpos: 17维)
    # base_pos(3) + base_vel(6) + joint_pos(8)
    base_pos = np.random.randn(3).astype(np.float32)
    base_vel = np.random.randn(6).astype(np.float32)
    joint_pos = np.random.randn(8).astype(np.float32)
    qpos = np.concatenate([base_pos, base_vel, joint_pos])
    
    images = {
        'rgb_main': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'rgb_left': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'rgb_under': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    
    data = {
        'qpos': qpos,
        'images': images
    }
    
    print(f"Sending qpos shape: {qpos.shape}")
    print(f"Sending images: {list(images.keys())}")
    
    socket.send(pickle.dumps(data))
    response = pickle.loads(socket.recv())
    
    print(f"Response status: {response.get('status', 'N/A')}")
    if 'action' in response:
        action = response['action']
        print(f"Action shape: {action.shape} (expected: (16,))")
        print(f"  - Joint positions (7): {action[:7]}")
        print(f"  - Gripper position (1): {action[7:8]}")
        print(f"  - Joint velocities (7): {action[8:15]}")
        print(f"  - Gripper velocity (1): {action[15:]}")
    elif 'message' in response:
        print(f"Error message: {response['message']}")
    
    # 测试3: 再次RESET
    print("\n--- Test 3: RESET again ---")
    socket.send(pickle.dumps('RESET'))
    response = pickle.loads(socket.recv())
    print(f"Response: {response}")
    
    print("\n✓ All tests completed!")
    socket.close()
    context.term()

if __name__ == '__main__':
    test_server()
