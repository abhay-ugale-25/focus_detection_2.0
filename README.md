# Focus-CV: Temporal Intelligence Proctoring Engine

An advanced, headless computer vision pipeline that classifies student cognitive states (Focused, Distracted, Drowsy) in real-time. 

Originally built as a static rule-based script, Focus-CV has been entirely re-architected to utilize a Long Short-Term Memory (LSTM) neural network. By analyzing 30-frame rolling windows of facial metrics, the system understands sustained behavioral patterns rather than reacting to single-frame anomalies, providing a robust, highly stable proctoring kill-switch.

## 🧠 Core Architecture

* **Deep Temporal Memory:** Replaces brittle frame-by-frame counters with a PyTorch LSTM (Input: 3, Hidden: 64, Layers: 2). The model processes a continuous sliding window to recognize the flow of time and behavior.
* **Scale-Invariant Feature Engineering:** * **EAR (Eye Aspect Ratio):** Tracks blinks and prolonged eye closures.
  * **Gaze Ratio:** Tracks left/right iris orientation.
  * **Bounded Head Pitch Ratio:** Calculates `Upper Face Distance / Total Face Height`. This mathematical constraint ensures the pitch metric remains strictly between 0.0 and 1.0, allowing users to lean in, sit back, or shift side-to-side without triggering false distance anomalies.
* **Signal Smoothing (Mode Filter):** Implements a rolling voting buffer on the live inference output to debounce transition boundaries and eliminate micro-flickering.
* **Privacy-First Headless Backend:** Vision inference runs completely locally as a client (`live_inference.py`). State changes are broadcasted via JSON payloads to a local FastAPI microservice (`api_server.py`), strictly avoiding the privacy/security vulnerabilities of live-hosted web UIs like Streamlit.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, scikit-learn
* **Computer Vision:** MediaPipe FaceLandmarker, OpenCV
* **Backend:** FastAPI, Uvicorn, Requests
* **Data Processing:** Pandas, NumPy

## ⚙️ Installation

It is highly recommended to run this pipeline within a dedicated Anaconda environment to prevent dependency conflicts between PyTorch and MediaPipe.

```bash
# 1. Create and activate the environment
conda create -n focus_lstm python=3.10 -y
conda activate focus_lstm

# 2. Install PyTorch (CPU version shown, add CUDA toolkit if utilizing NVIDIA GPUs)
conda install pytorch torchvision torchaudio -c pytorch -y

# 3. Install Data & Backend dependencies
conda install pandas numpy scikit-learn -y
conda install fastapi uvicorn requests -c conda-forge -y

# 4. Install Computer Vision wheels
pip install opencv-python mediapipe
```

## 🚀 Usage Guide
### Phase 1: Data Collection & Training (If retraining is required)
- Run the data collection script to record raw features while acting out the three cognitive states. Press 0 for Focused, 1 for Distracted, and 2 for Drowsy to tag the incoming data.

- The `data_prep.py` module will parse the resulting CSV using Stratified Sequence Chunking and Sequence-Mode Labeling to construct an overlapping 30-frame tensor dataset while preventing class imbalance and temporal leakage.

- Run ```python train.py``` to train the LSTM and save the weights to focus_lstm_model.pth.

### Phase 2: Live Inference & Proctoring
The system operates as a decoupled microservice. You must run the server and the client simultaneously.

##### 1. Start the Kill-Switch Server:
* In your first terminal, spin up the FastAPI listener:
```python
api_server.py
```
_The server will run on port 8000 and listen for POST /update_state payloads._

##### 2. Launch the Vision Worker:
* In your second terminal, start the inference engine:
```python
live_inference.py
```
* Sit upright and press 'b' to lock in your baseline metrics.
* The system will maintain a rolling buffer of your facial features, predict your state, apply the mode filter, and fire an HTTP alert to the backend exactly when a sustained state change occurs. Press 'q' to quit.
> Built with OpenCV, MediaPipe, and PyTorch.
