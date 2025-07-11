# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.

"""
    classifiers_utils.py
    This file contains the utility functions for loading the classifier of vulnerabilities
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
script_dir = os.path.dirname(__file__)
from cyberbattle.utils.file_utils import load_yaml # noqa: E402

class CustomTransformerClassifier(nn.Module):
    def __init__(self, num_labels, model_class, freeze_layers=True, num_trainable_layers=2):
        super(CustomTransformerClassifier, self).__init__()
        self.model = model_class
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels) # additional linear layer for mapping to the number of classes

        if freeze_layers:
            self.freeze_parameters(num_trainable_layers) # freeze all layers except the last 'num_trainable_layers' otherwise too much data may be needed

    def freeze_parameters(self, num_trainable_layers):
        # Remove the requirements of gradients for all layers except the last 'num_trainable_layers'
        layers = list(self.model.modules())
        for layer in layers[:-num_trainable_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        try:
            pooled_output = outputs.pooler_output
        except AttributeError:
            pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

class VulnerabilityClassifier:
    def __init__(self, hf_token, logger, verbose=2, threshold=0.2, max_len=256, model_dir=os.path.join(script_dir, '..', 'models','classifier')):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_file = os.path.join(self.model_dir, 'pytorch_model.bin')
        general_config = load_yaml(os.path.join(script_dir, '..', '..', 'config.yaml'))
        self.vulnerability_classifier_path = general_config['vulnerability_classifier_path']
        self.model_identifier = general_config['vulnerability_classifier_base_identifier']
        self.logger = logger
        self.verbose = verbose

        # Check if model files are already downloaded
        if not self.check_if_model_exists():
            raise ValueError("Use the setup_files.sh file to download the default classifier or modify the config.yaml file resetting the path of the classifier of your choice.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_identifier, token=hf_token)
        self.model = CustomTransformerClassifier(12, AutoModel.from_pretrained(self.model_identifier, token=hf_token))
        logger.info(f"Loading model from {self.model_file}")

        self.model.load_state_dict(torch.load(self.model_file, map_location='cpu'))
        self.model.eval()
        self.threshold = threshold
        self.max_len = max_len

        # Define inclusion and exclusion rules enforced on the model output to fix potential issues with defined rules
        # Inclusion rule A -> B are rules such that once you have a class A for a sample (you always have to have the B class(es))
        self.inclusion_rules = [
            ('execution', ['lateral move', 'privilege escalation', 'discovery', 'reconnaissance', 'collection', 'exfiltration', 'DOS', 'persistence', 'defense evasion']),
            ('collection', ['exfiltration']),
        ]
        # Exclusion rule A are rules such that if you have a class A for a sample, you cannot have any other class(es)
        self.exclusion_labels = ['No Class']

    def check_if_model_exists(self):
        return os.path.exists(os.path.join(script_dir, "..", "models", self.vulnerability_classifier_path))


    def map_strings(self, index):
        labels = ['discovery', 'reconnaissance', 'collection', 'exfiltration', 'execution', 'privilege escalation', 'lateral move', 'credential access', 'DOS', 'persistence', 'defense evasion', None]
        return labels[index]

    def enforce_inclusion_rules(self, predicted_labels):
        label_prob_map = {label: prob for label, prob in predicted_labels}
        for label_present, required_labels in self.inclusion_rules:
            if label_present in label_prob_map:
                for required_label in required_labels:
                    if required_label not in label_prob_map:
                        label_prob_map[required_label] = label_prob_map[label_present]
        return label_prob_map

    def enforce_exclusion_rules(self, predicted_labels):
        if any(label in self.exclusion_labels for label in predicted_labels):
            return [(label, predicted_labels[label]) for label in predicted_labels if label in self.exclusion_labels]
        return predicted_labels

    def predict(self, text):
        # Encode the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_len, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs)
        probs = torch.softmax(logits, dim=1).squeeze()

        # Gather labels and probabilities where the probability is greater than threshold
        high_prob_labels = [(self.map_strings(idx), prob.item()) for idx, prob in enumerate(probs) if float(prob.item()) > self.threshold]
        predicted_labels = [label for label, _ in high_prob_labels]
        high_prob_labels = [(label, prob) for label, prob in high_prob_labels if label in predicted_labels]
        # Enforce inclusion and exclusion rules
        sanitized_labels = self.enforce_inclusion_rules(high_prob_labels)
        sanitized_labels = self.enforce_exclusion_rules(sanitized_labels)

        return sanitized_labels
