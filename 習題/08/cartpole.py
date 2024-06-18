# 參考範例並原創
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

steps = 0
max_steps = 0

for _ in range(1000):
    env.render()

    position, velocity, angle, angular_velocity = observation

    if angle > 0:
        if angular_velocity > 0: action = 1
        else: action = 0 
    else:
        if angular_velocity < 0: action = 0 
        else: action = 1 

    if position > 0.1: action = 0
    elif position < -0.1: action = 1

    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        if steps > max_steps: max_steps = steps
        print(f'Died after {steps} steps')
        steps = 0
        observation, info = env.reset()

env.close()
print(f'Maximum steps survived: {max_steps}')