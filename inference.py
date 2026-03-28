import os
import sys
import json
from app.baseline import run_baseline

def main():
    print("==================================================")
    print("TrustDeskEnv - Hackathon Inference Script")
    print("==================================================")
    
    # 1. Read required environment variables from the Hackathon Evaluation specs
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")
    
    print(f"[Config] MODEL_NAME: {model_name}")
    print(f"[Config] API_BASE_URL: {api_base_url or 'Default OpenAI Base URL'}")
    print(f"[Config] HF_TOKEN: {'***' if hf_token else 'Not Set'}")
    print("--------------------------------------------------")
    
    if not hf_token and not os.getenv("OPENAI_API_KEY"):
        print("Warning: Neither HF_TOKEN nor OPENAI_API_KEY environment variable is set.")
        print("The baseline agent requires an API key for the OpenAI client initialization.")
    
    # 2. Run the baseline agent
    try:
        # We pass hf_token as api_key and api_base_url as base_url
        result = run_baseline(
            model=model_name,
            api_key=hf_token,
            base_url=api_base_url,
            verbose=True
        )
    except Exception as e:
        print(f"\n[Error] Inference failed with exception: {e}")
        sys.exit(1)
        
    # 3. Validation format checks
    if "error" in result:
        print(f"\n[Error] Baseline internally failed: {result['error']}")
        sys.exit(1)
        
    # 4. Success output
    summary = {k: v for k, v in result.items() if k != "details"}
    print("\n==================================================")
    print("Task Execution Complete. Final Summary:")
    print("==================================================")
    print(json.dumps(summary, indent=2))
    
    average_score = result.get("average_score", 0.0)
    print(f"\n[Final Submission Result] Average Score: {average_score:.3f}")
    
    return 0

if __name__ == "__main__":
    main()
