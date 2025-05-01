import numpy as np
from lime.lime_text import LimeTextExplainer
import argparse
import os

from model_utils import load_model_and_tokenizer
from lime_predictor import DecoderLimePredictor

def explain_text(model_name: str, text_to_explain: str, class_names: list[str], num_features: int, num_samples: int, output_dir: str):
    """
    Loads a model, explains a given text using LIME, and saves the explanation.

    Args:
        model_name: Name of the Hugging Face model to use.
        text_to_explain: The input text string to explain.
        class_names: List of class names for the explanation (e.g., ["Negative", "Positive"])
        num_features: Number of features (words) to show in the explanation.
        num_samples: Number of samples LIME should generate.
        output_dir: Directory to save the explanation HTML file.
    """
    model, tokenizer, device = load_model_and_tokenizer(model_name)

    predictor = DecoderLimePredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        class_names=class_names
    )

    explainer = LimeTextExplainer(class_names=class_names)

    print(f"\nGenerating LIME explanation for: '{text_to_explain}'")
    exp = explainer.explain_instance(
        text_to_explain,
        predictor.predict_proba, 
        num_features=num_features,
        num_samples=num_samples
    )

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "lime_explanation.html")

    exp.save_to_file(output_file)
    print(f"Explanation saved to {output_file}")

    print("\nTop features contributing to prediction:")
    print(exp.as_list())

    pred = predictor.predict_proba([text_to_explain])
    print(f"\nModel prediction probabilities for '{text_to_explain}':")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {pred[0, i]:.4f}")
    predicted_class_idx = np.argmax(pred)
    print(f"Predicted class: {class_names[predicted_class_idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain text predictions using LIME with decoder models.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the Hugging Face Causal LM model.")
    parser.add_argument("--text", type=str, required=True, help="The text to explain.")
    parser.add_argument("--class_names", nargs='+', default=["Negative", "Positive"], help="List of class names.")
    parser.add_argument("--num_features", type=int, default=6, help="Number of features for LIME explanation.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for LIME.")
    parser.add_argument("--output_dir", type=str, default="./lime_explanations", help="Directory to save the explanation HTML file.")

    args = parser.parse_args()

    explain_text(
        model_name=args.model_name,
        text_to_explain=args.text,
        class_names=args.class_names,
        num_features=args.num_features,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
