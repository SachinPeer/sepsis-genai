"""
GenAI Inference Service - Multi-Provider LLM Integration
Stage 2 of the Multimodal Sepsis Early Warning System

This service handles communication with LLM providers for
narrative-based sepsis risk prediction.

Supported Providers:
  - Azure OpenAI (GPT-4o) - Default
  - AWS Bedrock (Claude 3.5 Sonnet)

Set LLM_PROVIDER environment variable to switch:
  - "azure" (default)
  - "bedrock"
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT BASE CLASS FOR LLM PROVIDERS
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def predict(self, system_prompt: str, user_message: str, temperature: float) -> Dict[str, Any]:
        """Send a prediction request to the LLM."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is accessible."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


# =============================================================================
# AZURE OPENAI PROVIDER
# =============================================================================

class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI GPT-4o provider."""
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "azure_openai"
    
    @property
    def model_name(self) -> str:
        return self.model
    
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
    
    def predict(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Send prediction request to Azure OpenAI."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        result["_metadata"] = {
            "provider": self.provider_name,
            "model": self.model,
            "tokens_used": response.usage.total_tokens if response.usage else None,
            "finish_reason": response.choices[0].finish_reason
        }
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Azure OpenAI is accessible."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Respond with: OK"}],
                max_tokens=10
            )
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model,
                "endpoint": self.endpoint[:50] + "..." if self.endpoint else None
            }
        except Exception as e:
            return {"status": "unhealthy", "provider": self.provider_name, "error": str(e)}


# =============================================================================
# AWS BEDROCK CLAUDE PROVIDER
# =============================================================================

class BedrockClaudeProvider(BaseLLMProvider):
    """AWS Bedrock Claude 3.5 Sonnet provider."""
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        # Use inference profile ID for on-demand invocation (required for Claude 3.5+)
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "aws_bedrock"
    
    @property
    def model_name(self) -> str:
        return self.model_id
    
    @property
    def client(self):
        """Lazy-load the Bedrock client."""
        if self._client is None:
            try:
                import boto3
                
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.region
                )
                logger.info(f"AWS Bedrock client initialized (region: {self.region})")
            except ImportError:
                logger.error("boto3 package not installed. Run: pip install boto3")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        return self._client
    
    def predict(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> Dict[str, Any]:
        """Send prediction request to AWS Bedrock Claude."""
        
        # Claude message format
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        content = response_body["content"][0]["text"]
        
        # Parse JSON from Claude's response
        # Claude may wrap JSON in markdown code blocks, so we handle that
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        result["_metadata"] = {
            "provider": self.provider_name,
            "model": self.model_id,
            "tokens_used": response_body.get("usage", {}).get("input_tokens", 0) + 
                          response_body.get("usage", {}).get("output_tokens", 0),
            "finish_reason": response_body.get("stop_reason", "unknown")
        }
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Bedrock is accessible."""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Respond with: OK"}]
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.model_id,
                "region": self.region
            }
        except Exception as e:
            return {"status": "unhealthy", "provider": self.provider_name, "error": str(e)}


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

def get_llm_provider(provider_name: str = None) -> BaseLLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider_name: "azure" or "bedrock". Defaults to LLM_PROVIDER env var or "azure"
    
    Returns:
        Configured LLM provider instance
    """
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "azure").lower()
    
    providers = {
        "azure": AzureOpenAIProvider,
        "azure_openai": AzureOpenAIProvider,
        "bedrock": BedrockClaudeProvider,
        "aws_bedrock": BedrockClaudeProvider,
        "claude": BedrockClaudeProvider,
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown LLM provider: {provider_name}. Supported: {list(providers.keys())}")
    
    logger.info(f"Creating LLM provider: {provider_name}")
    return providers[provider_name]()


# =============================================================================
# MAIN INFERENCE SERVICE (Backward Compatible)
# =============================================================================

class GenAIInferenceService:
    """
    Multi-provider LLM inference service for sepsis prediction.
    Supports Azure OpenAI (GPT-4o) and AWS Bedrock (Claude 3.5).
    
    Set LLM_PROVIDER env var to switch: "azure" (default) or "bedrock"
    """
    
    def __init__(self, provider: str = None):
        """
        Initialize the inference service.
        
        Args:
            provider: Optional provider override ("azure" or "bedrock")
        """
        self._provider: Optional[BaseLLMProvider] = None
        self._provider_name = provider
        
        # Load the system prompt
        self.system_prompt = self._load_system_prompt()
        
        logger.info(f"GenAI Inference Service initialized (provider: {provider or os.getenv('LLM_PROVIDER', 'azure')})")
    
    @property
    def provider(self) -> BaseLLMProvider:
        """Lazy-load the LLM provider."""
        if self._provider is None:
            self._provider = get_llm_provider(self._provider_name)
        return self._provider
    
    # Backward compatibility properties
    @property
    def model(self) -> str:
        return self.provider.model_name
    
    @property
    def endpoint(self) -> str:
        if hasattr(self.provider, 'endpoint'):
            return self.provider.endpoint
        return f"bedrock:{self.provider.region}" if hasattr(self.provider, 'region') else "unknown"
        
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
    
    def predict(self, patient_narrative: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Send patient narrative to LLM for sepsis risk prediction.
        
        Args:
            patient_narrative: The serialized clinical narrative from preprocessor
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            Parsed JSON response from the LLM
        """
        try:
            user_message = f"{{PATIENT_NARRATIVE_SUMMARY}}\n\n{patient_narrative}"
            
            result = self.provider.predict(
                system_prompt=self.system_prompt,
                user_message=user_message,
                temperature=temperature
            )
            
            logger.info(f"GenAI prediction completed. Provider: {self.provider.provider_name}, Risk score: {result.get('prediction', {}).get('risk_score_0_100', 'N/A')}")
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
        """Check if the LLM service is accessible."""
        return self.provider.health_check()
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider."""
        return {
            "provider": self.provider.provider_name,
            "model": self.provider.model_name
        }


# Singleton instances for reuse (one per provider)
_inference_services: Dict[str, GenAIInferenceService] = {}

def get_genai_service(provider: str = None) -> GenAIInferenceService:
    """
    Get or create the GenAI inference service singleton.
    
    Args:
        provider: Optional provider override ("azure" or "bedrock")
    
    Returns:
        GenAIInferenceService instance
    """
    global _inference_services
    
    # Determine provider key
    provider_key = provider or os.getenv("LLM_PROVIDER", "azure")
    
    if provider_key not in _inference_services:
        _inference_services[provider_key] = GenAIInferenceService(provider=provider)
    
    return _inference_services[provider_key]


def switch_provider(new_provider: str) -> GenAIInferenceService:
    """
    Switch to a different LLM provider.
    
    Args:
        new_provider: "azure" or "bedrock"
    
    Returns:
        New GenAIInferenceService instance for the provider
    """
    logger.info(f"Switching LLM provider to: {new_provider}")
    return get_genai_service(provider=new_provider)


# --- CLI Testing ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GenAI Inference Service")
    parser.add_argument("--provider", choices=["azure", "bedrock"], default=None,
                        help="LLM provider to use (default: from LLM_PROVIDER env var or 'azure')")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison between Azure and Bedrock")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
    
    if args.compare:
        print("=" * 60)
        print("MODEL COMPARISON: Azure OpenAI vs AWS Bedrock Claude")
        print("=" * 60)
        
        for provider in ["azure", "bedrock"]:
            print(f"\n--- Testing {provider.upper()} ---")
            try:
                service = GenAIInferenceService(provider=provider)
                health = service.health_check()
                print(f"Health: {health}")
                
                if health["status"] == "healthy":
                    import time
                    start = time.time()
                    result = service.predict(test_narrative)
                    elapsed = time.time() - start
                    
                    print(f"Latency: {elapsed:.2f}s")
                    print(f"Risk Score: {result.get('prediction', {}).get('risk_score_0_100', 'N/A')}")
                    print(f"Priority: {result.get('prediction', {}).get('priority', 'N/A')}")
                    print(f"Tokens: {result.get('_metadata', {}).get('tokens_used', 'N/A')}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        service = GenAIInferenceService(provider=args.provider)
        
        print("Testing GenAI Inference Service...")
        print(f"Provider: {service.provider.provider_name}")
        print(f"Model: {service.provider.model_name}")
        print("-" * 50)
        
        # Check health first
        health = service.health_check()
        print(f"Health Check: {health}")
        
        if health["status"] == "healthy":
            result = service.predict(test_narrative)
            print("\nPrediction Result:")
            print(json.dumps(result, indent=2))
