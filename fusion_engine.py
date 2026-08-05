import numpy as np

# Canonical shared classes
SHARED_CLASSES = ["Anger", "Disgust/Fear", "Joy/Happiness", "Sadness", "Surprise", "Neutral"]

# Mapping dictionary for Vision (ResNet-18 output classes)
VISION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 1,
    'happy': 2,
    'sad': 3,
    'surprise': 4,
    'neutral': 5
}

# Mapping dictionary for Text (michelleli99 classifier classes)
TEXT_MAP = {
    'anger': 0,
    'fear': 1,
    'joy': 2,
    'love': 2,  # Map 'love' to Joy bucket
    'sadness': 3,
    'surprise': 4
}

class EmotionFusionEngine:
    def __init__(self, incongruity_threshold=0.45):
        """
        incongruity_threshold: Distance threshold above which a contradiction is flagged.
        """
        self.threshold = incongruity_threshold

    def align_vision_probs(self, raw_vision_dict):
        """Converts raw vision class probabilities to 6-class shared vector."""
        vec = np.zeros(6, dtype=np.float32)
        for cls_name, prob in raw_vision_dict.items():
            if cls_name in VISION_MAP:
                idx = VISION_MAP[cls_name]
                vec[idx] += prob
        # Normalize vector
        total = np.sum(vec)
        return vec / total if total > 0 else vec

    def align_text_probs(self, raw_text_dict):
        """Converts raw text class probabilities to 6-class shared vector."""
        vec = np.zeros(6, dtype=np.float32)
        for item in raw_text_dict:
            cls_name = item['label'].lower()
            prob = item['score']
            if cls_name in TEXT_MAP:
                idx = TEXT_MAP[cls_name]
                vec[idx] += prob
        
        # If no strong emotional signal in text, attribute remaining mass to Neutral
        sum_prob = np.sum(vec)
        if sum_prob < 1.0:
            vec[5] = 1.0 - sum_prob
            
        total = np.sum(vec)
        return vec / total if total > 0 else vec

    def analyze(self, vision_raw, text_raw):
        """
        Evaluates facial vs spoken emotion vectors, calculates incongruity,
        and determines final verdict.
        """
        v_vec = self.align_vision_probs(vision_raw)
        t_vec = self.align_text_probs(text_raw)

        top_v_idx = np.argmax(v_vec)
        top_t_idx = np.argmax(t_vec)

        top_v_class = SHARED_CLASSES[top_v_idx]
        top_t_class = SHARED_CLASSES[top_t_idx]

        # 1. Compute Jensen-Shannon / Cosine Discrepancy
        # Distance metric: 1 - Cosine Similarity
        dot_prod = np.dot(v_vec, t_vec)
        norm_product = (np.linalg.norm(v_vec) * np.linalg.norm(t_vec))
        cosine_sim = dot_prod / (norm_product + 1e-8)
        discrepancy_score = 1.0 - cosine_sim

        # 2. Contradiction Evaluation Rules
        is_contradiction = False
        verdict = ""
        badge_color = "GREEN"  # GREEN = Agreement, RED = Contradiction, YELLOW = Subtle

        # Ignore contradiction if either modality is Neutral
        if top_v_class == "Neutral" or top_t_class == "Neutral":
            is_contradiction = False
            # Prioritize the non-neutral signal for final verdict
            primary_idx = top_t_idx if top_v_class == "Neutral" else top_v_idx
            verdict = f"{SHARED_CLASSES[primary_idx]} (Mild)"
            badge_color = "GREEN"

        elif top_v_idx == top_t_idx:
            # Complete agreement
            is_contradiction = False
            fused_vec = 0.5 * v_vec + 0.5 * t_vec
            verdict = f"Aligned: {SHARED_CLASSES[np.argmax(fused_vec)]}"
            badge_color = "GREEN"

        else:
            # Check for valence opposition (Sarcasm / Masked Emotion)
            # Joy vs Sadness/Anger is a hard contradiction
            is_happy_face = (top_v_class == "Joy/Happiness")
            is_negative_text = (top_t_class in ["Sadness", "Anger", "Disgust/Fear"])

            is_negative_face = (top_v_class in ["Sadness", "Anger", "Disgust/Fear"])
            is_happy_text = (top_t_class == "Joy/Happiness")

            if (is_happy_face and is_negative_text) or (is_negative_face and is_happy_text):
                is_contradiction = True
                verdict = f"Sarcasm / Masked {top_t_class}"
                badge_color = "RED"
            elif discrepancy_score > self.threshold:
                is_contradiction = True
                verdict = f"Incongruent ({top_v_class} Face vs {top_t_class} Text)"
                badge_color = "RED"
            else:
                is_contradiction = False
                verdict = f"Mixed: {top_v_class} / {top_t_class}"
                badge_color = "YELLOW"

        return {
            "vision_top": top_v_class,
            "vision_conf": float(v_vec[top_v_idx]),
            "text_top": top_t_class,
            "text_conf": float(t_vec[top_t_idx]),
            "discrepancy_score": float(discrepancy_score),
            "is_contradiction": is_contradiction,
            "final_verdict": verdict,
            "badge_color": badge_color
        }

# Quick Verification Test
if __name__ == "__main__":
    engine = EmotionFusionEngine()

    # Mock Case 1: Sarcasm (Smiling face, but negative text)
    mock_vision = {'happy': 0.88, 'neutral': 0.10, 'sad': 0.02}
    mock_text = [{'label': 'sadness', 'score': 0.94}, {'label': 'anger', 'score': 0.06}]

    result = engine.analyze(mock_vision, mock_text)
    print("--- MOCK SARCASM TEST RESULT ---")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")