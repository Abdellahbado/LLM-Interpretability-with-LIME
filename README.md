# LIME for Decoder Language Models

This project demonstrates how to use LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of decoder-only language models (like Gemma, Qwen) for classification tasks.

It reframes the classification task as a next-token prediction problem, allowing LIME to analyze which parts of the input text influence the model's choice between different class-representative tokens.

## Features

*   Uses Hugging Face `transformers` for model loading.
*   Supports GPU acceleration via MPS (macOS) or CUDA (if available).
*   Provides a flexible command-line interface (`explain.py`) for running explanations.
*   Outputs explanations as HTML files.

## Requirements

*   Python 3.x
*   PyTorch
*   Transformers
*   LIME
*   NumPy
*   SciPy

Install dependencies (preferably in a virtual environment):
```bash
pip install torch transformers lime numpy scipy
```

## Usage

Run explanations from the command line:

```bash
python explain.py --text "Your text to explain here." --model_name "Qwen/Qwen3-0.6B" 
```


**Arguments:**

*   `--text`: (Required) The input text to explain.
*   `--model_name`: The Hugging Face model name (default: `gpt2`).
*   `--class_names`: List of class names (default: `["Negative", "Positive"]`).
*   `--num_features`: Number of features in the LIME explanation (default: 6).
*   `--num_samples`: Number of samples LIME uses (default: 1000).
*   `--output_dir`: Directory to save the HTML explanation (default: `./lime_explanations`).

The explanation will be saved as an HTML file in the specified output directory.
