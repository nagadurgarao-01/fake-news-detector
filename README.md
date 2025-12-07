# Hybrid Fake News Detection System

A comprehensive fake news detection system that combines **BERT** (for semantic text analysis) and **Graph Neural Networks (GNN)** (for entity relationship modeling). This project includes a complete ML training pipeline, a Flask inference server, and a Chrome extension for real-time verification.

## 🚀 Features

*   **Hybrid Model**: Combines BERT embeddings with a Graph Convolutional Network (GCN) to analyze both content semantics and knowledge graph context.
*   **Knowledge Graph**: Automatically builds a graph of named entities from news articles to capture relationships between key players.
*   **Community Detection**: Uses Louvain community detection to identify clusters of related entities.
*   **Real-time Inference**: A Flask-based API server that loads the trained model for fast predictions.
*   **Chrome Extension**: A browser extension that scans news websites and highlights headlines in **Green (Real)** or **Red (Fake)** based on the model's prediction.
*   **GPU Optimization**: Optimized for NVIDIA GPUs (specifically tuned for RTX 3050).

## 🛠️ Prerequisites

*   Python 3.8+
*   NVIDIA GPU with CUDA (recommended for training)
*   Google Chrome (for the extension)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nagadurgarao-01/fake-news-detector.git
    cd fake-news-detector
    ```

2.  **Install Dependencies:**
    Run the provided batch script to install Python requirements and download necessary SpaCy models.
    ```powershell
    ./install_dependencies.bat
    ```
    *Or manually:*
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## 🏋️‍♂️ Training the Model

To train the model from scratch using the provided datasets (Fake.csv, True.csv, etc.):

1.  Make sure your data files are in the `data/` directory.
2.  Run the setup script:
    ```powershell
    ./run_setup.bat
    ```
    *This will executes `main.py`, which handles data loading, graph building, training, and saving the model artifacts to the `models/` folder.*

## 🔌 Running the Server

Once training is complete (or if you have pre-trained models), start the inference server:

```powershell
./run_server.bat
```
*Alternatively:*
```bash
python server.py
```
The server runs on `http://127.0.0.1:5000`.

## 🌐 Installing the Chrome Extension

1.  Open Chrome and navigate to `chrome://extensions`.
2.  Enable **Developer mode** (toggle in the top right).
3.  Click **Load unpacked**.
4.  Select the `chrome_extension` folder located in this project directory.
5.  Browse any news website! Headlines will be automatically analyzed and highlighted.

## 📂 Project Structure

*   `main.py`: Master pipeline script for training and evaluation.
*   `server.py`: Flask application for real-time inference.
*   `config.yaml`: Configuration for model hyperparameters and paths.
*   `scripts/`: Helper modules for data processing, graph building, and modeling.
    *   `hybrid_model.py`: PyTorch model definitions (BERT + GNN).
    *   `get_bert_embeddings.py`: BERT feature extraction.
    *   `extract_relationships.py`: Knowledge graph construction.
*   `chrome_extension/`: Source code for the browser extension.
*   `data/`: Directory for input datasets (CSV/TSV).
*   `models/`: Directory where trained models and graph artifacts are saved.
*   `logs/`: Execution logs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
