class EntityExtractor:
    def extract_entities(self, text: str) -> dict:
        # Simple rule-based extraction for testing
        tech_terms = ["AWS", "Azure", "Kubernetes", "Docker", "DevOps", "Terraform"]
        entities = []
        
        for term in tech_terms:
            if term in text:
                entities.append({
                    "name": term,
                    "type": "TECHNOLOGY",
                    "confidence": 0.9
                })
        
        return {
            "entities": entities,
            "total_entities": len(entities),
            "method": "rule-based",
            "key_phrases": tech_terms
        }