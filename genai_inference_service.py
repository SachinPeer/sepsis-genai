"""
GenAI Inference Service - Azure OpenAI Integration
Stage 2 of the Multimodal Sepsis Early Warning System

This service handles communication with Azure OpenAI (GPT-4o) for
narrative-based sepsis risk prediction.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GenAIInferenceService:
    """
    Azure OpenAI inference service for sepsis prediction.
    Uses GPT-4o for Chain-of-Thought reasoning on clinical narratives.
    """
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        
        # Load the system prompt
        self.system_prompt = self._load_system_prompt()
        
        # Initialize client (lazy loading)
        self._client = None
        
    def _load_system_prompt(self) -> str:
        """Load the clinical system prompt from file."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), 
            "knowledge", "Guide bank", "prompt.md"
        )
        try:
            with open(prompt_path, 'r') as f:
                content = f.read()
                # Extract just the prompt content (skip metadata)
                if "## SYSTEM ROLE" in content:
                    return content
                return content
        except FileNotFoundError:
            logger.warning(f"Prompt file not found at {prompt_path}, using default")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback system prompt if file not found."""
        return """You are a Board-Certified ICU Intensivist and Medical Data Scientist. 
Your specialty is detecting "Silent Sepsis"—the period where physiological compensation hides organ failure.
Your goal is to predict septic shock 6 hours before a blood pressure crash.

Analyze the patient narrative and return ONLY a JSON object with this structure:
{
  "prediction": {
    "risk_score_0_100": integer,
    "priority": "Critical" | "High" | "Standard",
    "sepsis_probability_6h": "High" | "Moderate" | "Low",
    "clinical_rationale": "Brief explanation"
  },
  "clinical_metrics": {
    "qSOFA_score": integer,
    "SIRS_met": boolean,
    "trend_velocity": "Improving" | "Stable" | "Deteriorating",
    "organ_stress_indicators": []
  },
  "logic_gate": {
    "discordance_detected": boolean,
    "primary_driver": "What triggered this score?",
    "missing_parameters": []
  }
}"""
    
    @property
    def client(self):
        """Lazy-load the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                
                if not self.api_key:
                    raise ValueError("AZURE_OPENAI_API_KEY not set in environment")
                if not self.endpoint:
                    raise ValueError("AZURE_OPENAI_ENDPOINT not set in environment")
                
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint
                )
                logger.info("Azure OpenAI client initialized successfully")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                raise
        return self._client
    
    def predict(self, patient_narrative: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Send patient narrative to Azure OpenAI for sepsis risk prediction.
        
        Args:
            patient_narrative: The serialized clinical narrative from preprocessor
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            Parsed JSON response from the LLM
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{{PATIENT_NARRATIVE_SUMMARY}}\n\n{patient_narrative}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Enforce JSON output
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Add metadata
            result["_metadata"] = {
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "finish_reason": response.choices[0].finish_reason
            }
            
            logger.info(f"GenAI prediction completed. Risk score: {result.get('prediction', {}).get('risk_score_0_100', 'N/A')}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return self._create_error_response("JSON parse error", str(e))
        except Exception as e:
            logger.error(f"GenAI inference failed: {e}")
            return self._create_error_response("Inference error", str(e))
    
    def predict_batch(self, narratives: list, temperature: float = 0.1) -> list:
        """
        Process multiple patient narratives in sequence.
        (For true batch processing, use Azure Bedrock Batch API)
        
        Args:
            narratives: List of patient narrative strings
            temperature: LLM temperature
            
        Returns:
            List of prediction results
        """
        results = []
        for i, narrative in enumerate(narratives):
            logger.info(f"Processing batch item {i+1}/{len(narratives)}")
            result = self.predict(narrative, temperature)
            result["_batch_index"] = i
            results.append(result)
        return results
    
    def _create_error_response(self, error_type: str, details: str) -> Dict[str, Any]:
        """Create a structured error response."""
        return {
            "prediction": {
                "risk_score_0_100": 50,  # Default to medium risk on error
                "priority": "High",  # Err on side of caution
                "sepsis_probability_6h": "Moderate",
                "clinical_rationale": f"[SYSTEM ERROR: {error_type}] Manual review required."
            },
            "clinical_metrics": {
                "qSOFA_score": -1,
                "SIRS_met": None,
                "trend_velocity": "Unknown",
                "organ_stress_indicators": []
            },
            "logic_gate": {
                "discordance_detected": False,
                "primary_driver": f"Error: {error_type}",
                "missing_parameters": ["All - system error"],
                "error_details": details
            },
            "_metadata": {
                "error": True,
                "error_type": error_type
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the Azure OpenAI service is accessible."""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Respond with: OK"}],
                max_tokens=10
            )
            return {
                "status": "healthy",
                "model": self.model,
                "endpoint": self.endpoint[:50] + "..." if self.endpoint else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance for reuse
_inference_service: Optional[GenAIInferenceService] = None

def get_genai_service() -> GenAIInferenceService:
    """Get or create the GenAI inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = GenAIInferenceService()
    return _inference_service


# --- CLI Testing ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    service = GenAIInferenceService()
    
    # Test narrative
    test_narrative = """
    IDENTIFICATION: Patient Age 68
    --- NUMERIC TRENDS ---
    CRITICAL TRENDS: The patient shows signs of tachycardia (118 bpm, rising), hypotension (85 mmHg, dropping).
    Heart Rate is 118 bpm. Lactate is 3.2 mmol/L, which is critically elevated.
    Other telemetry: WBC: 15.2, Creatinine: 2.1, Temp: 38.8...
    
    --- CLINICIAN NOTES ---
    Patient is complains of chills. Altered mental status noted by nursing staff.
    Received 2L fluid bolus with minimal response. Skin appears mottled.
    """
    
    print("Testing GenAI Inference Service...")
    print("-" * 50)
    
    # Check health first
    health = service.health_check()
    print(f"Health Check: {health}")
    
    if health["status"] == "healthy":
        result = service.predict(test_narrative)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
