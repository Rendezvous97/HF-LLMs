# 🤗 HuggingFace Transformers Course - Learning Journey

This repository contains my implementations and exercises from the HuggingFace Transformers course, covering fundamental concepts to advanced fine-tuning techniques in Natural Language Processing.

## 📚 Course Overview

This comprehensive course takes you through the entire journey of working with Transformer models, from basic pipeline usage to advanced training techniques. Each notebook builds upon previous concepts, providing hands-on experience with the HuggingFace ecosystem.

## 🗂️ Repository Structure

### Core Learning Modules

#### **Chapter 1: Transformers 101** (`CH1_transformers_101.ipynb`)
- **Objective**: Introduction to the Transformers library and pipeline functionality
- **Key Concepts**:
  - Pipeline function basics (sentiment analysis, text generation, etc.)
  - Multi-modal pipelines (text, image, audio)
  - Text generation parameters (greedy decoding, beam search, sampling)
  - Named Entity Recognition (NER) with `grouped_entities`
  - Question answering and summarization
- **Highlights**: 
  - Comparison of different text generation methods
  - Working with various pipeline tasks across modalities

#### **Chapter 2: Inside the Pipeline** (`CH2_inside_pipeline.ipynb`)
- **Objective**: Understanding the three-step process behind pipelines
- **Key Concepts**:
  - Preprocessing with tokenizers
  - Model processing and hidden states
  - Postprocessing with softmax and label mapping
- **Architecture**: Tokenizer → Model → Post-processing
- **Practical Skills**: Manual pipeline recreation, understanding model outputs

#### **Chapter 3: Inside Models** (`CH3_inside_models.ipynb`)
- **Objective**: Deep dive into Transformer model architecture and usage
- **Key Concepts**:
  - AutoModel vs specific model classes (BertModel, etc.)
  - Model loading, saving, and Hub integration
  - Text encoding and special tokens
  - Padding and truncation strategies
  - Model heads and their purposes
- **File Operations**: Working with `config.json` and model weights

#### **Chapter 4: Inside Tokenizers** (`CH4_inside_tokenizers.ipynb`)
- **Objective**: Comprehensive understanding of tokenization methods
- **Key Concepts**:
  - Word-based vs Character-based vs Subword tokenization
  - BPE, WordPiece, and Unigram algorithms
  - Encoding and decoding processes
  - Tokenizer loading and saving
- **Algorithms Covered**: Byte-level BPE (GPT-2), WordPiece (BERT), SentencePiece/Unigram

#### **Chapter 5: Multiple Sequences in Tokenizers** (`CH5_multiple_sequences_in_tokenizers.ipynb`)
- **Objective**: Handling batch processing and sequence pairs
- **Key Concepts**:
  - Batch tokenization and padding strategies
  - Attention masks and their importance
  - Tensor creation and data type handling
  - Sequence pair processing

#### **Chapter 6: Using Complete Pipeline** (`CH6_using_complete_pipeline.ipynb`)
- **Objective**: End-to-end pipeline implementation and customization
- **Advanced Usage**: Custom pipeline creation and optimization

#### **Chapter 7: Optimized Inference & Deployment** (`CH7_Optimized_Inference_Deployment.ipynb`)
- **Objective**: Production-ready model deployment strategies
- **Focus Areas**: Performance optimization and scalable inference

#### **Chapter 8: Fine-tuning Data Processing** (`CH8_Fine_tuning_data_processing.ipynb`)
- **Objective**: Data preparation for model fine-tuning
- **Key Skills**: Dataset preprocessing, formatting, and validation

#### **Chapter 9: Fine-Tuning with Trainer API** (`CH9_Fine_Tuning_w_Trainer_API.ipynb`)
- **Objective**: High-level fine-tuning using HuggingFace Trainer
- **Features**: Simplified training loop, evaluation metrics, and model checkpointing

#### **Chapter 10: Training with PyTorch** (`CH10_Training_w_Pytorch.ipynb`)
- **Objective**: Low-level training implementation with raw PyTorch
- **Advanced Topics**: Custom training loops, optimization strategies

### 🧪 Experimental Notebooks

#### **GLUE Benchmark Tasks** (`EXP_GLUE_Tasks.ipynb`)
- **Purpose**: Comprehensive overview of GLUE benchmark tasks
- **Tasks Covered**:
  - **Single-sentence**: CoLA (grammatical acceptability), SST-2 (sentiment)
  - **Sentence pairs**: MRPC (paraphrase), QQP (question similarity), STS-B (semantic similarity)
  - **NLI Tasks**: MNLI, QNLI, RTE, WNLI
- **Utility Function**: `preprocess_glue()` for standardized data preprocessing

#### **DistilGPT2 Fine-tuning Experiments**
- `EXP_finetune_distilGPT_causal_lm.ipynb`: Causal language modeling fine-tuning
- `EXP_finetune_instruct_distilGPT_causal_lm.ipynb`: Instruction-following fine-tuning

## 🚀 Getting Started

### Prerequisites

```bash
# Create conda environment
conda create -n transformers-course python=3.12
conda activate transformers-course

# Install required packages
pip install transformers datasets torch torchvision torchaudio
pip install accelerate evaluate wandb
pip install jupyter notebook ipywidgets
```

### Environment Setup

1. **HuggingFace Authentication**:
   ```python
   from huggingface_hub import login
   login()  # Enter your HF token
   ```

2. **Environment Variables** (optional):
   ```bash
   # Create .env file
   echo "HUGGINGFACE_TOKEN=your_token_here" > .env
   ```

### Usage Instructions

1. **Start with fundamentals**: Begin with CH1-CH4 for core concepts
2. **Progress sequentially**: Each chapter builds upon previous knowledge
3. **Experiment**: Use the EXP notebooks to test specific techniques
4. **Practice**: Modify examples with your own data

## 📖 Key Learning Outcomes

- **Pipeline Usage**: Efficient use of pre-trained models for various NLP tasks
- **Model Architecture**: Understanding of Transformer internals and components
- **Tokenization**: Different tokenization strategies and their trade-offs
- **Fine-tuning**: Both high-level (Trainer API) and low-level (PyTorch) approaches
- **Data Processing**: Efficient dataset preparation and preprocessing
- **Deployment**: Production-ready model optimization and inference
- **Evaluation**: Working with standard benchmarks like GLUE

## 🛠️ Technical Highlights

### Text Generation Methods
- **Greedy Decoding**: Fast, deterministic (single output)
- **Beam Search**: High-quality, diverse outputs (`num_beams` parameter)
- **Sampling**: Creative, varied text (`do_sample=True`, `temperature`)

### Model Types Covered
- **Base Models**: BERT, DistilBERT, GPT-2, T5
- **Task-Specific**: Sequence classification, token classification, question answering
- **Multimodal**: Vision Transformer (ViT), Whisper (speech)

### Advanced Features
- **Attention Mechanisms**: Understanding attention masks and their importance
- **Special Tokens**: CLS, SEP, MASK tokens and their usage
- **Model Heads**: Task-specific layers for different applications

## 🎯 Best Practices Learned

1. **Always use appropriate tokenizer**: Match tokenizer to model checkpoint
2. **Handle padding correctly**: Use attention masks for variable-length sequences
3. **Validate data types**: Ensure proper tensor types (LongTensor for embeddings)
4. **Monitor training**: Use wandb for experiment tracking
5. **Version control models**: Save checkpoints and configurations

## 📄 License

This project is for educational purposes, following the HuggingFace course materials.

## 🙏 Acknowledgments

- **HuggingFace Team**: For the excellent course materials and library
- **Transformers Community**: For continuous improvements and support
- **Research Community**: For the foundational papers and models

---

*Happy Learning! 🚀*
