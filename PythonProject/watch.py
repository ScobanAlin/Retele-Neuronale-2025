import gymnasium as gym
import torch
import cv2
import numpy as np
import os
import time

# Import your classes. 
# NOTE: Ensure your test.py has the training loop inside `if __name__ == "__main__":`
# otherwise importing it will start training again!
from test import Agent, FlappyBirdWrapper

def watch_agent():
    print("Initializing Environment...")
    # 1. Use "rgb_array" so the Agent gets the pixel data it needs
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = FlappyBirdWrapper(env)
    
    agent = Agent(env)
    
    # 2. Load the trained model
    if os.path.exists("flappy_checkpoint.pth"):
        agent.load("flappy_checkpoint.pth")
        print("Model loaded successfully.")
    else:
        print("Error: No checkpoint found! Train the agent first.")
        return

    # 3. Turn off Randomness (Pure Skill)
    agent.epsilon = 0.0
    print(f"Agent Ready! Epsilon: {agent.epsilon}")
    
    # 4. Run the game loop
    episodes = 10
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"Starting Episode {ep+1}...")
        
        while True:
            # Select best action
            action = agent.select_action(state)
            
            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # === VISUALIZATION ===
            # We want to see what the agent sees.
            # 'next_state' is a stack of 4 frames (4, 84, 84)
            # We grab the last frame (most current) to show on screen
            latest_frame = next_state[-1] 
            
            # Resize it from 84x84 to 400x400 so we can see it clearly
            big_frame = cv2.resize(latest_frame, (400, 400), interpolation=cv2.INTER_NEAREST)
            
            # Show the window
            cv2.imshow("Agent Vision (What the AI Sees)", big_frame)
            
            # Wait 30ms to simulate 30 FPS (otherwise it runs too fast)
            if cv2.waitKey(30) == ord('q'): 
                print("Quitting...")
                env.close()
                return

            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Episode {ep+1} Finished. Score: {total_reward} (Steps: {steps})")
                time.sleep(1) # Pause briefly between deaths
                break
                
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    watch_agent()