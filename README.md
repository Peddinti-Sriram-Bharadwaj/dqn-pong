# 🏓 DQN Pong Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Status](https://img.shields.io/badge/Status-Solved-success)

An implementation of a **Deep Q-Network (DQN)** capable of solving *Atari Pong* with superhuman performance. The agent learns from raw pixels using the Arcade Learning Environment (ALE) and achieves a perfect or near-perfect score (Reward > +19.0) against the built-in AI.

## 📖 Overview

The agent (Green Paddle) controls the right side of the Pong game. The inputs are raw 84x84 grayscale pixel frames, processed via a Convolutional Neural Network (CNN). Through deep reinforcement learning, the agent learns optimal gameplay strategies without any human intervention.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/dqn-pong.git
cd dqn-pong
```

### 2. Install Dependencies

This project uses Gymnasium v1.0+ and PyTorch.

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
gymnasium[atari,accept-rom-license]>=1.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

**Note:** If you encounter ROM errors, you may need to accept the AutoROM license:

```bash
AutoROM --accept-license
```

---

## 🕹️ Usage

### 1. Watch the Trained Agent

See the AI in action using the pre-trained model included in this repo.

```bash
python watch_pong.py -m models/PongNoFrameskip-v4-best_19.dat
```

**Controls:** Press `Q` to quit the window.

### 2. Train from Scratch

To train the agent yourself (Warning: Takes ~30-40 mins on RTX 3090, or ~2-3 hours on M-series Mac).

```bash
python dqn_pong.py --env PongNoFrameskip-v4
```

**Optional Arguments:**
- `--dev cuda` - Use NVIDIA GPU
- `--dev cpu` - Use CPU (default for Mac if MPS is not explicitly supported)

### 3. Play as Human

Think you can beat the built-in AI? Try the human control script.

```bash
python play_human.py
```

**Controls:** 
- `W` - Move paddle up
- `S` - Move paddle down

---

## 🧠 Technical Details

This implementation follows the classic DQN (Deep Q-Network) architecture introduced by DeepMind (Mnih et al., 2015), featuring:

### Key Components

- **Experience Replay Buffer:** Stores 10,000+ past transitions to break correlation between consecutive samples
- **Target Network:** A frozen copy of the main network to stabilize the Q-value targets (Bellman equation)
- **Frame Stacking:** The input is a stack of 4 consecutive frames, allowing the agent to perceive motion and velocity
- **Reward Clipping:** Rewards are clipped to {-1, 0, 1} to stabilize gradients

### Network Architecture

```
Input (84x84x4 stacked frames)
    ↓
Conv2D(32 filters, 8x8 kernel, stride=4) + ReLU
    ↓
Conv2D(64 filters, 4x4 kernel, stride=2) + ReLU
    ↓
Conv2D(64 filters, 3x3 kernel, stride=1) + ReLU
    ↓
Flatten → Fully Connected(512) + ReLU
    ↓
Fully Connected(6) → Q-values for 6 actions
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Discount Factor (γ) | 0.99 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.01 |
| Epsilon Decay | 10,000 frames |
| Batch Size | 32 |
| Replay Buffer Size | 10,000 |
| Target Network Update | 1,000 steps |

### Performance

- **Solved Condition:** Mean reward of +19.0 over 100 episodes
- **Training Time:** ~800,000 frames to convergence
- **Final Average Reward:** +19.5
- **Peak Performance:** +21.0

---

## 📂 File Structure

```
dqn-pong/
├── dqn_pong.py          # Main training loop and hyperparameter configuration
├── watch_pong.py        # Visualization script using OpenCV
├── play_human.py        # Human player interface
├── requirements.txt     # Project dependencies
├── lib/
│   ├── dqn_model.py     # CNN architecture (PyTorch)
│   └── wrappers.py      # Gymnasium wrappers for frame preprocessing
└── models/
    └── *.dat            # Saved model checkpoints
```

---

## 🔧 Troubleshooting

**ROM not found error:**
```bash
pip install gymnasium[accept-rom-license]
```

**CUDA out of memory:**

Reduce batch size in `dqn_pong.py`:
```python
BATCH_SIZE = 16  # Default is 32
```

**Training too slow on CPU:**

Consider using Google Colab with GPU runtime (free tier available) or reduce the replay buffer size.

---

## 📊 Training Results

The agent typically solves Pong within 800 episodes. Training progress can be monitored through the console output, which displays:
- Current episode number
- Mean reward over last 100 episodes
- Epsilon value (exploration rate)
- Loss values

Models are automatically saved when the agent achieves new best performance.

---

## 📜 Credits

Based on the Deep Q-Learning algorithm from the paper:

**"Playing Atari with Deep Reinforcement Learning"**  
Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013)  
DeepMind Technologies

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- DeepMind for the original DQN paper
- OpenAI Gymnasium team for the Atari environments
- PyTorch team for the deep learning framework
