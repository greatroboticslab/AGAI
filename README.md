# $\color{green}{\text{Plant Diagnostic System}}$ <img width="300" height="300" alt="agailogofix" src="https://github.com/user-attachments/assets/160162c3-f736-45db-8b6e-daa10c0e5378" />



An AI-powered strawberry disease detection and diagnostic system built on three complementary models: a **ResNet-50** image classifier for disease identification, an **RF-DETR Small** object detector for plant-part grounding, and **MiniGPT-v2** for generating detailed, visually grounded diagnostic reports. The system serves a Gradio web interface where users upload or capture images of strawberry plants and receive structured diagnoses with actionable treatment recommendations sourced from university extension publications.

> Research prototype developed at Middle Tennessee State University. Not a substitute for professional agronomic advice.

---

## How It Works

The system runs a three-stage pipeline on every uploaded image:

1. **ResNet-50 Classification** -- A fine-tuned ResNet-50 classifies the image into one of seven conditions: healthy, drought, overwatering, root rot, frost injury, gray mold, or white mold. Test-time augmentation and temperature scaling produce calibrated confidence scores.

2. **RF-DETR Part Detection** -- An RF-DETR Small object detector identifies which plant parts are visible in the image (flower, fruit, leaf, root, soil, stem). Per-class confidence thresholds are tuned to each part's detection characteristics. The detected parts constrain MiniGPT's output so it only describes what is actually visible.

3. **MiniGPT-v2 Grounded Explanation** -- A LoRA-adapted MiniGPT-v2 (LLaMA-2-7B backbone) receives the ResNet diagnosis as ground truth and the RF-DETR part detections as visibility constraints. It generates a structured diagnostic report covering the diagnosis, visible symptoms on detected parts, and treatment recommendations. Parts that are not detected are excluded from the response, reducing hallucination.

```
Image Upload --> ResNet-50 Classification --> RF-DETR Part Detection --> MiniGPT-v2 Grounded Report
```

A knowledge graph RAG system supports follow-up Q&A and provides part-specific treatment advice when RF-DETR detects particular plant structures.

---

## Features

- **Three-stage AI pipeline** with classification, detection, and vision-language explanation
- **Seven disease classes**: healthy, drought, overwatering, root rot, frost injury, gray mold, white mold
- **Plant-part grounding**: RF-DETR constrains MiniGPT to only describe visible parts, with optional bounding box overlay on the uploaded image
- **Knowledge graph RAG**: disease-specific follow-up Q&A with part-specific treatment recommendations grounded in university extension publications
- **Hallucination detection**: cross-checks model output against known disease indicators to flag and regenerate confused responses
- **Web search integration**: optional SERP API enrichment for supplementary context
- **Dark-themed Gradio interface** with chat, settings, and an interactive disease knowledge graph visualization

---

## Requirements

- Python 3.9+ (the demo runs in a `minigptv` Conda environment; RF-DETR training uses a separate `rfdetr` environment with Python 3.11)
- CUDA GPU with 16 GB+ VRAM (RTX 3090 or better recommended)
- PyTorch with CUDA support
- Model weights (not included in the repository):
  - LLaMA-2-7B-chat: `llama_weights/Llama-2-7b-chat-hf/`
  - MiniGPT-v2 fine-tuned checkpoint: `output/minigpt_strawberry/checkpoint_best.pth`
  - ResNet-50 classifier: `plant_diagnostic/models/resnet_strawberry.pth`
  - RF-DETR Small fine-tuned checkpoint: `rfdetr_parts/output/checkpoint_best_total.pth`
- (Optional) SERP API key in a `.env` file for web search features

---

## Setup

```bash
# Clone the repository
git clone https://github.com/greatroboticslab/AGAI.git
cd AGAI

# Create the Conda environment for the demo
conda env create -f environment.yml
conda activate minigptv

# Install PyTorch with CUDA support (adjust cu121 to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install transformers==4.41.1 bitsandbytes gradio pillow numpy pandas plotly networkx python-dotenv timm serpapi
```

For RF-DETR training, a separate environment is needed:

```bash
conda create -n rfdetr python=3.11 -y
conda activate rfdetr
pip install rfdetr torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Environment Variables

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SERP_API_KEY=your_key_here   # optional, for web search features
```

Or place `SERP_API_KEY=your_key_here` in a `.env` file at the project root (gitignored).

---

## Running the Demo

```bash
python demo_v5.py
```

This starts a Gradio web interface. Open the printed URL in your browser, upload a strawberry plant image, and the system will return a structured diagnosis. Use the settings panel to toggle bounding box visualization, adjust generation temperature, or explore the knowledge graph tab.

---

## Training

### MiniGPT-v2 Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29607 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m torch.distributed.run --nproc_per_node=2 train.py \
  --cfg-path train_configs/minigptv2_strawberry_v2.yaml
```

### RF-DETR Part Detector

Prepare the COCO-format dataset and launch training (in the `rfdetr` Conda environment):

```bash
conda activate rfdetr
python rfdetr_parts/prepare_dataset.py
python rfdetr_parts/train_rfdetr.py
```

Training hyperparameters are in `rfdetr_parts/rfdetr_config.yaml`. The trained checkpoint is saved to `rfdetr_parts/output/`.

---

## Project Structure

```
AGAI/
├── demo_v5.py                  # Gradio web interface (main entry point)
├── resnet_classifier.py        # ResNet-50 inference and TTA
├── train.py                    # MiniGPT-v2 training script
├── eval_only.py                # Non-interactive evaluation
├── eval_holdout.py             # Holdout set evaluation
├── test_grounding.py           # End-to-end grounding test orchestrator
├── ui_components.py            # Gradio UI HTML/CSS constants
├── dark_theme.css              # UI theme stylesheet
├── AGAILogo.png                # Project logo
├── environment.yml             # Conda environment definition
│
├── grounding/                  # RF-DETR grounding pipeline
│   ├── config.py               # Paths, part lists, thresholds
│   ├── detector.py             # Subprocess wrapper for RF-DETR inference
│   ├── analyzer.py             # Text analysis (negation, advice, calyx remap)
│   ├── inference.py            # MiniGPT loading and prompt construction
│   └── report.py               # Structured terminal output
│
├── rfdetr_parts/               # RF-DETR Small training and inference
│   ├── rfdetr_config.yaml      # Training hyperparameters
│   ├── prepare_dataset.py      # COCO dataset preparation and splitting
│   ├── train_rfdetr.py         # Training script
│   ├── detect_images.py        # Inference with per-class thresholds
│   └── eval_thresholds.py      # Confidence threshold evaluation
│
├── knowledge_graph/            # Disease knowledge RAG system
│   ├── disease_knowledge_base.json  # Sourced treatments, part-specific advice
│   ├── rag_retriever.py        # Context retrieval, hallucination checks
│   ├── qa_retriever.py         # Follow-up Q&A with pattern matching
│   └── README.md
│
├── minigpt4/                   # Core MiniGPT-v2 framework
│   ├── models/                 # Model architectures (MiniGPT-v2, EVA-ViT, Q-Former)
│   ├── datasets/               # Dataset builders and loaders
│   ├── processors/             # Image and text processors
│   ├── runners/                # Training runners
│   ├── tasks/                  # Training task definitions
│   ├── conversation/           # Conversation management
│   └── common/                 # Utilities, config, registry
│
├── plant_diagnostic/           # ResNet-50 training data and models
│   ├── data/                   # Training images (7 classes)
│   ├── models/                 # Trained ResNet checkpoint
│   ├── scripts/                # Training and utility scripts
│   └── src/                    # Source modules
│
├── hallucination_evaluation/   # Response quality evaluation suite
├── evaluation/                 # Model evaluation scripts
├── configs/                    # MiniGPT model config YAMLs
├── eval_configs/               # Inference configuration
├── train_configs/              # Training configuration
├── figs/                       # Architecture diagrams and paper figures
├── documentation/              # Session notes
├── annotationjsons/            # Label Studio annotation exports
├── RESEARCH_PAPER.md           # Research paper (Markdown)
└── RESEARCH_PAPER.docx         # Research paper (Word)
```

Large directories excluded from the repository via `.gitignore`:

- `llama_weights/` -- LLaMA-2-7B-chat model weights
- `output/` -- MiniGPT training checkpoints
- `checkpoints/` -- Stage-2 checkpoints
- `rfdetr_parts/dataset/` and `rfdetr_parts/output/` -- RF-DETR training data and checkpoints

---

## Knowledge Graph and RAG

The `knowledge_graph/disease_knowledge_base.json` file contains detailed, citation-backed information for all seven disease classes, including:

- Symptoms, causes, visual indicators, and severity ratings
- Treatment protocols web-crawled and curated from UC IPM, Penn State Extension, NC State Extension, Cornell Extension, University of Minnesota Extension, and others
- **Part-specific treatments** keyed to RF-DETR detections (e.g., fruit-specific advice for gray mold when fruit is detected in the image)
- Recovery timelines and prevention strategies

The `DiseaseRAG` retriever provides context injection and hallucination checking, while `DiseaseQA` handles follow-up questions with pattern-matched routing to the appropriate knowledge section.

---

## Architecture Diagrams (outdated)

### Inference Dataflow
![Inference Dataflow](https://github.com/user-attachments/assets/b6f433ad-d25a-4b06-b0c5-e00addd43984)

### End-to-End Inference Sequence
![End-to-End Inference Sequence](https://github.com/user-attachments/assets/7bf297ba-d818-48be-a803-d94d6994c62c)

### Training Pipeline
![Training Pipeline](https://github.com/user-attachments/assets/fc1dfc01-c445-4c3f-9398-709c4845fcfc)

---

## Demo Videos (outdated)

### Project Overview
[![Plant Diagnostic Project Demo](https://img.youtube.com/vi/-QEf8KkALK4/0.jpg)](https://youtu.be/-QEf8KkALK4?si=mVatePAGOcpFYOXw)

### Current Demo (demo_v5)
[![Demo v5](https://img.youtube.com/vi/eduSbkigjLY/0.jpg)](https://youtu.be/eduSbkigjLY?si=24JHh-qNNbvROLdD)

### Webcam and Unknown Scenario
[![Webcam Demo](https://img.youtube.com/vi/PXN6_6oj7_M/0.jpg)](https://youtu.be/PXN6_6oj7_M)

---

## Citation

```bibtex
@software{plant_diagnostic_system,
  title  = {Plant Diagnostic System: AI-Powered Strawberry Disease Detection},
  author = {William Starks and Gus Marcum},
  year   = {2026},
  url    = {https://github.com/greatroboticslab/AGAI}
}
```

---

## License and Credits

This project builds on:

- [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) -- vision-language model framework
- [LLaMA-2](https://ai.meta.com/llama/) -- language model backbone (Meta AI)
- [RF-DETR](https://github.com/roboflow/rf-detr) -- real-time detection transformer (Roboflow)
- [ResNet](https://arxiv.org/abs/1512.03385) -- image classification architecture

Treatment recommendations in the knowledge base are web-crawled and sourced from university cooperative extension publications (UC IPM, Cornell, Penn State, NC State, UMN, Virginia Tech, Utah State, UGA). See individual disease entries in `knowledge_graph/disease_knowledge_base.json` for full citations.

The models were trained on a curated corpus of publicly available web-scale data and indexed digital repositories.

Please respect upstream licenses and dataset terms of use.
