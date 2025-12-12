# 🧠 DNN Training Pipeline

**A complete, production-ready machine learning pipeline for training Deep Neural Networks from scratch using pure NumPy**

Train and compare 9 different DNN models across 3 datasets (MNIST, Fashion-MNIST, CIFAR-10) with a single command, then interact with them through a standalone web interface. Perfect for understanding deep learning fundamentals and production ML pipeline design.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Why This Project?

### Two Components, One Purpose
This project has **two completely separate parts**:

1. **🧪 Training Pipeline** (`run_pipeline.py`) - For training models
2. **🎨 Interactive UI** (`launch_ui.py`) - For testing and visualization

**Why separate?**
- Training needs computational resources and time
- UI needs to be responsive and always available
- Different teams can work on different components
- You can train on GPU servers and deploy UI anywhere

### What You Get
- ✅ **No black boxes**: See exactly how forward/backward propagation works
- ✅ **Production patterns**: Learn how real ML pipelines are structured
- ✅ **Interactive testing**: Web interface to play with trained models
- ✅ **Modular design**: Each component is independent and reusable
- ✅ **Comprehensive experiments**: Train 9 models automatically and compare results
- ✅ **Real-world data**: Work with actual datasets (MNIST, Fashion-MNIST, CIFAR-10)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Clone and enter
git clone https://github.com/yourusername/dnn-training-pipeline.git
cd dnn-training-pipeline

# Install (using uv for fast dependency management)
uv venv 
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv sync
```

### 2. Train All Models (5 minutes)
```bash
# This downloads datasets and trains all 9 models
python run_pipeline.py --mode all
```

### 3. Launch Interactive UI
```bash
# In a separate terminal (after training completes)
python launch_ui.py
```

### 4. Open Your Browser
Go to `http://localhost:7860` and start testing models!

---

## 📁 Project Structure

```
dnn-training-pipeline/
│
├── 📁 config/
│   └── config.yaml                 # Training configuration
│
├── 📁 src/                         # Training pipeline source code
│   ├── data/                       # Data loading & preprocessing
│   ├── models/                     # DNN implementation (pure NumPy)
│   ├── training/                   # Training orchestration
│   ├── evaluation/                 # Metrics & visualization
│   └── utils/                      # Logging utilities
│
├── 📄 run_pipeline.py              # 🧪 TRAINING CLI (no UI)
├── 📄 launch_ui.py                 # 🎨 STANDALONE UI launcher
│
├── 📁 data/                        # Auto-created: Downloaded datasets
├── 📁 models/                      # Auto-created: Trained models (.pkl)
├── 📁 results/                     # Auto-created: Training visualizations
├── 📁 logs/                        # Auto-created: Training logs
│
├── requirements.txt
└── README.md
```

---

## 🧪 Part 1: Training Pipeline (`run_pipeline.py`)

**Purpose:** Train models and save them to disk. No UI components.

### Available Commands

```bash
# Train ALL 9 models (3 architectures × 3 datasets)
python run_pipeline.py --mode all

# Train ONE specific model
python run_pipeline.py --mode train --dataset mnist --arch simple_dnn

# Just download and preprocess data
python run_pipeline.py --mode data

# Evaluate a trained model
python run_pipeline.py --mode evaluate --dataset mnist --arch simple_dnn

# Compare all trained models
python run_pipeline.py --mode compare
```

### What Happens During Training?
1. **Downloads datasets** (cached locally)
2. **Preprocesses**: Normalizes, flattens, splits train/val/test
3. **Trains**: Mini-batch SGD with progress bars
4. **Saves**: Model weights, training curves, metrics
5. **Compares**: Generates comparison plots

### Output Directory Structure
```
models/
├── simple_dnn_mnist.pkl
├── medium_dnn_mnist.pkl
├── deep_dnn_mnist.pkl
├── simple_dnn_fashion.pkl
└── ... (9 total)

results/
├── simple_dnn_mnist_training.png     # Loss/acc curves
├── simple_dnn_mnist_results.json     # Metrics
├── comparison.png                    # All models comparison
└── ...

logs/
└── dnn_pipeline_20250101_120000.log  # Complete training log
```

### Expected Performance
| Dataset | Simple DNN | Medium DNN | Deep DNN | Training Time |
|---------|------------|------------|----------|---------------|
| MNIST | ~95% | ~97% | ~97.5% | ~30s |
| Fashion-MNIST | ~85% | ~87% | ~88% | ~30s |
| CIFAR-10* | ~45% | ~48% | ~50% | ~45s |

*CIFAR-10 is converted to grayscale, explaining lower accuracy

---

## 🎨 Part 2: Interactive UI (`launch_ui.py`)

**Purpose:** Web interface to test trained models. Completely separate from training.

### Features
- 📸 Upload images or draw directly
- 🎯 Real-time predictions with confidence scores
- 📊 Probability distribution visualizations
- 🔄 Compare all models on same image
- 🖼️ Sample images for quick testing
- 🌐 Public URL sharing capability

### Launch Options
```bash
# Basic launch (local only)
python launch_ui.py

# Create public shareable URL (expires in 72h)
python launch_ui.py --share

# Run on different port
python launch_ui.py --port 8080

# Use custom models directory
python launch_ui.py --models-dir ./custom_models
```

### UI Screenshots
```
┌─────────────────────────────────────────────────────┐
│                DNN Model Inference                  │
├─────────────────────────────────────────────────────┤
│ [Dataset: MNIST]     [Upload Image]                │
│                                                    │
│  ┌─────────────────┐     ┌──────────────────┐     │
│  │                 │     │ Prediction: 7    │     │
│  │     Image       │     │ Confidence: 98%  │     │
│  │                 │     │                  │     │
│  └─────────────────┘     └──────────────────┘     │
│                                                    │
│  ┌────────────────────────────────────────────┐    │
│  │ Probability Distribution                   │    │
│  │ ██████ 0: 2%   ████████████ 7: 98%        │    │
│  │ ████ 1: 1%    ...                         │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Quick Test Buttons
The UI includes one-click test buttons:
- **MNIST**: Sample handwritten digits
- **Fashion**: Sample clothing items
- **CIFAR-10**: Sample objects (car, plane, etc.)

### Model Comparison Table
Upload one image and see how ALL trained models perform:

| Dataset | Model | Prediction | Confidence |
|---------|-------|------------|------------|
| MNIST | simple_dnn_mnist | 7 | 98.2% |
| Fashion | medium_dnn_fashion | T-shirt | 87.5% |
| CIFAR-10 | deep_dnn_cifar10 | Airplane | 52.3% |

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize training:

```yaml
# Example: Add a new architecture
architectures:
  experimental_dnn:
    layers: [null, 512, 256, 128, 10]  # null = auto-filled
    activations: ['relu', 'relu', 'relu', 'softmax']
    learning_rate: 0.0005
    l2_lambda: 0.001
    epochs: 30
    batch_size: 128
```

### Supported Architectures
- `simple_dnn`: 1 hidden layer (64 units)
- `medium_dnn`: 2 hidden layers (128, 64 units)
- `deep_dnn`: 3 hidden layers (256, 128, 64 units)

### Supported Datasets
- **MNIST**: 28×28 grayscale handwritten digits (0-9)
- **Fashion-MNIST**: 28×28 grayscale fashion items
- **CIFAR-10**: 32×32 RGB images (converted to grayscale)

---

## 📚 Learning Path

### Week 1: Fundamentals
```bash
# Day 1: Run complete pipeline
python run_pipeline.py --mode all

# Day 2: Examine code structure
# Look at src/models/dnn.py to understand forward/backward

# Day 3: Train single models
python run_pipeline.py --mode train --dataset mnist --arch simple_dnn
python run_pipeline.py --mode train --dataset cifar10 --arch simple_dnn

# Day 4: Compare results
# Why does CIFAR-10 perform worse? (hint: spatial structure)

# Day 5: Use UI to test understanding
python launch_ui.py
# Draw digits, test predictions
```

### Week 2: Extensions
1. Add new activation functions (leaky ReLU, tanh)
2. Implement momentum SGD
3. Add dropout regularization
4. Create custom dataset loader

### Week 3: Productionization
1. Add unit tests
2. Create Docker container
3. Set up CI/CD pipeline
4. Deploy UI to cloud

---

## 🧪 Advanced Usage

### Using as a Library
```python
from src.models.dnn import DNN
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

# Load data
loader = DataLoader()
train_data, test_data, classes = loader.load_dataset('mnist')

# Preprocess
preprocessor = DataPreprocessor()
data_dict = preprocessor.prepare_dataset(*train_data, *test_data, 'mnist')

# Build and train custom model
model = DNN(
    layer_sizes=[784, 128, 64, 10],
    activations=['relu', 'relu', 'softmax'],
    learning_rate=0.001,
    name='my_custom_model'
)

# Train (simplified)
for epoch in range(20):
    model.train_epoch(data_dict['X_train'], data_dict['y_train'])
    
# Save
model.save('models/my_custom_model.pkl')
```

### Adding New Dataset
1. Add to `config.yaml`:
```yaml
datasets:
  my_dataset:
    urls:
      train: https://example.com/train.npy
      test: https://example.com/test.npy
    classes: ['class1', 'class2', 'class3']
```

2. Update `src/data/loader.py` to handle new format

### Monitoring Training
```bash
# Watch log file in real-time
tail -f logs/dnn_pipeline_*.log

# Check GPU memory (if available)
nvidia-smi

# Monitor system resources
htop  # or top on Linux, Activity Monitor on Mac
```

---

## 🔬 Key Insights You'll Discover

### 1. **Why CNNs Beat DNNs on Images**
```python
# DNN sees this as different:
image1 = [[1, 0],    # Object in top-left
          [0, 0]]
          
image2 = [[0, 0],    # Same object, bottom-right
          [0, 1]]
          
# Same pixels to DNN, different spatial arrangement!
```

### 2. **Diminishing Returns with Depth**
```
Depth vs Accuracy on MNIST:
1 layer: 95.2% (50K params)
2 layers: 96.5% (150K params)  ← +1.3% for 3× params
3 layers: 97.5% (300K params)  ← +1.0% for 6× params
```

### 3. **Parameter Efficiency**
- **Simple DNN**: 50K params, 95% accuracy
- **Deep DNN**: 300K params, 97.5% accuracy
- **6× params for 2.5% improvement** - often not worth it!

### 4. **Dataset Complexity Gradient**
- MNIST: Simple patterns, high accuracy
- Fashion-MNIST: More variation, lower accuracy
- CIFAR-10: Complex spatial+color, poor DNN performance

---

## ❓ FAQ

### Q: Can I run training and UI simultaneously?
**A:** Yes! They're completely separate. Train on one machine, run UI on another.

### Q: How do I deploy the UI to the cloud?
**A:** Several options:
```bash
# Option 1: ngrok (quickest)
ngrok http 7860

# Option 2: PythonAnywhere (free tier)
# Upload launch_ui.py and models/, run as web app

# Option 3: Hugging Face Spaces (recommended)
# Create new Space, upload code, runs free
```

### Q: Why separate training from UI?
**A:** Production best practice:
- Training: Heavy compute, batch processing
- Inference: Lightweight, real-time responses
- Can scale independently
- Different update cycles

### Q: Can I use my own images?
**A:** Yes! The UI accepts:
- Uploaded images (PNG, JPG, etc.)
- Drawn images (canvas)
- Webcam input
- Clipboard paste

### Q: What if training fails?
**A:** Check:
```bash
# 1. Internet connection for downloads
ping storage.googleapis.com

# 2. Disk space
df -h

# 3. Memory
free -h

# 4. Logs for error details
cat logs/*.log | grep -i error
```

### Q: How to save/load custom models?
```python
# Save
model.save('path/to/model.pkl')

# Load (in UI or another script)
from src.models.dnn import DNN
loaded = DNN.load('path/to/model.pkl')
```

---

## 🚀 Production Workflow Example

### Team A: Data Scientists
```bash
# Develop new model architecture
# Edit config.yaml, test locally
python run_pipeline.py --mode train --dataset fashion --arch experimental

# When satisfied, push to git
git add config.yaml
git commit -m "Add experimental architecture"
git push
```

### Team B: ML Engineers
```bash
# Pull latest, train at scale
git pull
python run_pipeline.py --mode all  # Trains all models

# Deploy best models
cp models/best_model.pkl /production/models/
```

### Team C: Frontend/DevOps
```bash
# Launch UI for stakeholders
python launch_ui.py --share

# Get public URL: https://xxxx.gradio.live
# Share with PM, designers, clients
```

### End Users
- Open browser to provided URL
- Upload test images
- Get instant predictions
- No coding required!

---

## 📈 Expected Results Timeline

### First 5 Minutes
```bash
# Terminal 1: Training
$ python run_pipeline.py --mode all
✓ Downloaded MNIST (60,000 images)
✓ Downloaded Fashion-MNIST (60,000 images) 
✓ Downloaded CIFAR-10 (60,000 images)
✓ Training simple_dnn_mnist... 95.2% accuracy
✓ Training medium_dnn_mnist... 96.5% accuracy
...
✓ All 9 models trained! (4m 32s)

# Terminal 2: UI (after training)
$ python launch_ui.py
🚀 Launching Gradio UI on port 7860...
🌐 Local URL: http://localhost:7860
```

### First 30 Minutes
- Test all 9 models through UI
- Understand performance differences
- Identify DNN limitations
- Plan CNN implementation

### First Week
- Modify architectures in config
- Add custom datasets
- Deploy UI to cloud
- Share with team for feedback

---

## 🤝 Contributing

### Areas Needing Help
1. **New Features**
   - CNN implementation
   - Advanced optimizers (Adam, RMSprop)
   - Data augmentation
   - Model explainability visuals

2. **Documentation**
   - Tutorial videos
   - Interactive notebooks
   - API documentation
   - Translation to other languages

3. **Infrastructure**
   - Docker support
   - CI/CD pipeline
   - Unit test coverage
   - Performance benchmarking

### How to Contribute
```bash
# 1. Fork repository
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/dnn-training-pipeline

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and test
python run_pipeline.py --mode train --dataset mnist --arch simple_dnn
python launch_ui.py

# 5. Commit and push
git add .
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 6. Create Pull Request
```

---

## 📚 Resources

### Learn More
- [Neural Networks from Scratch](https://nnfs.io/) - Book
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks) - Videos
- [CS231n Convolutional Neural Networks](http://cs231n.stanford.edu/) - Course
- [Deep Learning Book](https://www.deeplearningbook.org/) - Textbook

### Related Projects
- [Micrograd](https://github.com/karpathy/micrograd) - Tiny autograd engine
- [NumPy CNN](https://github.com/parasdahal/deepnet) - CNN from scratch
- [ML from Scratch](https://github.com/eriklindernoren/ML-From-Scratch) - Many algorithms

### Tools Used
- **NumPy**: Numerical computations
- **Matplotlib**: Visualizations
- **Gradio**: Web interface
- **tqdm**: Progress bars
- **PyYAML**: Configuration

---

## 🎯 Project Goals Achieved

### Educational Value ✅
- Understand every layer of neural networks
- See backpropagation in pure NumPy
- Learn production ML pipeline design
- Experience full ML workflow

### Practical Utility ✅
- Train models with one command
- Interactive testing without coding
- Easy configuration changes
- Reproducible experiments

### Extensibility ✅
- Add new datasets easily
- Create custom architectures
- Separate training from inference
- Modular, testable code

---

## 📞 Support

### Getting Help
1. **Check FAQ section** above
2. **Examine logs**: `logs/*.log`
3. **Search issues**: GitHub Issues tab
4. **Create new issue** with:
   - Command you ran
   - Full error message
   - `logs/*.log` snippet
   - System info

### Community
- **Discussions**: GitHub Discussions tab
- **Contributing**: See CONTRIBUTING.md
- **Star history**: Watch project growth
- **Releases**: Get notified of updates

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

### Citation
If you use this in research or teaching:
```bibtex
@software{dnn_training_pipeline,
  author = {Your Name},
  title = {DNN Training Pipeline with Interactive UI},
  year = {2025},
  url = {https://github.com/yourusername/dnn-training-pipeline}
}
```

---

## 🌟 What's Next?

After mastering this pipeline:

1. **Implement CNNs** - Add convolutional layers for spatial data
2. **Add Transfer Learning** - Use pre-trained models
3. **Create REST API** - Replace Gradio with FastAPI
4. **Add Monitoring** - Track model performance over time
5. **Deploy to Cloud** - AWS/GCP/Azure production deployment
6. **Build Mobile App** - TensorFlow Lite for iOS/Android

---

**Happy Learning and Coding! 🚀**

*Built with ❤️ by the SAIR for open-source community*
```
