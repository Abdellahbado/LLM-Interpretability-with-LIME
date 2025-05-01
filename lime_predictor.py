import torch
import numpy as np
from scipy.special import softmax
from transformers import PreTrainedModel, PreTrainedTokenizer

class DecoderLimePredictor:
    """
    Wraps a decoder-only transformer model to provide a prediction function
    suitable for LIME, framing classification as next-token prediction.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device, class_names: list[str], prompt_template: str = "{text} Sentiment:"):
        """
        Initializes the predictor.

        Args:
            model: The loaded Hugging Face causal LM model.
            tokenizer: The loaded Hugging Face tokenizer.
            device: The torch device (e.g., 'cuda' or 'cpu').
            class_names: A list of class names (e.g., ["Negative", "Positive"]).
                         The predictor will try to find single tokens for these.
            prompt_template: A format string for the prompt. Must contain '{text}'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names
        self.prompt_template = prompt_template
        self._determine_target_token_ids()

    def _determine_target_token_ids(self):
        """
        Determines the token IDs for the target class names.
        Attempts common variations (with/without leading space) if needed.
        Raises ValueError if single tokens cannot be found.
        """
        self.target_token_ids = []
        potential_prefixes = ['', ' ']

        for name in self.class_names:
            found_token = False
            for prefix in potential_prefixes:
                token_str = prefix + name
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    self.target_token_ids.append(token_ids[0])
                    print(f"Using token ID {token_ids[0]} for class '{name}' (encoded from '{token_str}')")
                    found_token = True
                    break
            if not found_token:
                 if name.lower() == "positive": alt_name = " good"
                 elif name.lower() == "negative": alt_name = " bad"
                 else: alt_name = None

                 if alt_name:
                     token_ids = self.tokenizer.encode(alt_name, add_special_tokens=False)
                     if len(token_ids) == 1:
                         self.target_token_ids.append(token_ids[0])
                         print(f"Warning: Could not find single token for '{name}'. Using alternative token ID {token_ids[0]} (encoded from '{alt_name}')")
                         found_token = True

            if not found_token:
                raise ValueError(
                    f"Could not find a single token representation for class name '{name}'. "
                    f"Tried with prefixes {potential_prefixes} and potential alternatives. "
                    f"Check the tokenizer vocabulary for model '{self.model.config._name_or_path}'."
                )

        if len(self.target_token_ids) != len(self.class_names):
             raise RuntimeError("Mismatch between number of class names and found token IDs.")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Generates class probabilities for a batch of texts.

        Args:
            texts: A list of input strings.

        Returns:
            A numpy array of shape (n_texts, n_classes) with probabilities.
        """
        prompts = [self.prompt_template.format(text=text) for text in texts]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        sequence_lengths = torch.ne(inputs['attention_mask'], 0).sum(-1) - 1
        last_token_logits = logits[torch.arange(logits.shape[0], device=self.device), sequence_lengths, :]

        class_logits = last_token_logits[:, self.target_token_ids]

        class_logits_cpu = class_logits.cpu()

        probs = softmax(class_logits_cpu.numpy(), axis=1)

        return probs
