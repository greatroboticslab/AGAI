"""
Configuration for Hallucination Evaluation
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_graph" / "disease_knowledge_base.json"
TRAINING_DATA_PATH = PROJECT_ROOT / "plant_diagnostic" / "datasets" / "stage2_train_7class_fixed.json"
HOLDOUT_DIR = PROJECT_ROOT / "plant_diagnostic" / "data" / "holdout"
TRAIN_DIR = PROJECT_ROOT / "plant_diagnostic" / "data" / "train_aug"
OUTPUT_DIR = Path(__file__).parent / "results"

# Model configs
EVAL_CONFIG_PATH = PROJECT_ROOT / "eval_configs" / "minigptv2_eval.yaml"
RESNET_PATH = PROJECT_ROOT / "plant_diagnostic" / "models" / "resnet_strawberry.pth"

# Disease classes
DISEASE_CLASSES = [
    "drought",
    "overwatering", 
    "root_rot",
    "frost_injury",
    "gray_mold",
    "white_mold",
    "healthy"
]

# Label normalization mapping
LABEL_ALIASES = {
    "overwatered": "overwatering",
    "frost": "frost_injury",
    "root rot": "root_rot",
    "gray mold": "gray_mold",
    "grey mold": "gray_mold",
    "white mold": "white_mold",
}

# Unique visual indicators per disease (for wrong-disease hallucination detection)
# IMPORTANT: Only include DEFINITE indicators that:
#   1. Are truly unique to ONE disease
#   2. Would NOT appear in care advice, comparisons, or negations
#   3. Describe actual visual symptoms, not conditions
UNIQUE_INDICATORS = {
    "drought": [
        # Specific visual symptoms only seen in drought
        "crispy brown edges", "papery leaves", "soil pulling away from pot"
    ],
    "overwatering": [
        # Specific visual symptoms - removed generic terms like "waterlogged"
        "mushy crown", "algae on soil", "green moss on soil", "yellowing lower leaves"
    ],
    "root_rot": [
        # Very specific pathogen-related terms
        "red stele", "phytophthora", "pythium", "root sloughing", "black mushy roots"
    ],
    "frost_injury": [
        # Specific frost damage patterns
        "black eye symptom", "translucent tissue", "glassy appearance", "ice crystal damage"
    ],
    "gray_mold": [
        # Very specific to Botrytis cinerea
        "gray fuzzy growth", "dusty gray spores", "tan mummified fruit", "botrytis cinerea"
    ],
    "white_mold": [
        # Very specific to Sclerotinia
        "black sclerotia", "cottony white mycelium", "sclerotinia", "hard black bodies"
    ],
    "healthy": [
        # These shouldn't trigger false positives - only flag if describing disease
        # Leave empty to avoid false positives when describing healthy characteristics
    ]
}

# Plant region keywords (for visibility hallucination detection)
PLANT_REGIONS = {
    "roots": ["root", "roots", "root system", "root zone", "root ball"],
    "crown": ["crown", "crown base", "plant crown"],
    "leaves": ["leaf", "leaves", "foliage", "leaflet", "leaflets"],
    "stems": ["stem", "stems", "petiole", "petioles", "runner", "runners"],
    "fruit": ["fruit", "berry", "berries", "strawberry", "strawberries"],
    "flowers": ["flower", "flowers", "blossom", "blossoms", "bloom", "blooms"],
    "soil": ["soil", "ground", "media", "growing media", "dirt"]
}

# Keywords that indicate claims about plant parts
CLAIM_KEYWORDS = [
    "shows", "appears", "visible", "can see", "observe", "notice",
    "evident", "present", "displaying", "exhibits", "has", "have",
    "affected", "damaged", "infected", "diseased"
]

# Evaluation settings
DEFAULT_NUM_SAMPLES = 50  # Number of images to evaluate
DEFAULT_TEMPERATURE = 0.2
MAX_NEW_TOKENS = 500

