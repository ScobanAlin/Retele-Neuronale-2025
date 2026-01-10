
import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_frame

env = gym.make("FlappyBird-v0", render_mode="rgb_array")
env.reset()

for _ in range(100):
    env.step(0)
    frame = env.render()
    processed = preprocess_frame(frame)

    plt.imshow(processed, cmap="gray")
    plt.title("Outline-only Preprocessing")
    plt.axis("off")
    plt.show()




env.close()