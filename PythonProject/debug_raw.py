import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

env.reset()

# Step a bit so the game is visible
for _ in range(500):
    env.step(0)
    frame = env.render()

    plt.imshow(frame)
    plt.title("RAW FRAME FROM render()")
    plt.axis("off")
    plt.show()

# 🔥 GET PIXELS FROM render(), NOT obs

print("frame shape:", frame.shape)
print("frame dtype:", frame.dtype)
print("min pixel:", frame.min(), "max pixel:", frame.max())

env.close()
