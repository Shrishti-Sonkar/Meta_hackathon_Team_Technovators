#!/usr/bin/env python3
"""
TrustDeskEnv Hackathon Inference Script

Requirement: Must be in root directory and use OpenAI Client for all LLM calls
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.baseline import run_baseline


def main():
    # Read required environment variables
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")
    
    # Check for API key
    api_key = hf_token or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("[Error] Neither HF_TOKEN nor OPENAI_API_KEY environment variable is set.\n")
        sys.exit(1)
    
    # Run baseline agent (uses OpenAI Client internally)
    result = None
    try:
        result = run_baseline(
            model=model_name,
            api_key=api_key,
            base_url=api_base_url,
            verbose=True  # Keep verbose to ensure output is generated
        )
    except Exception as e:
        sys.stderr.write(f"[Error] Exception during inference: {e}\n")
        sys.exit(1)
    
    # Check for errors in result
    if not result:
        sys.stderr.write("[Error] run_baseline returned None\n")
        sys.exit(1)
    
    if "error" in result:
        sys.stderr.write(f"[Error] {result['error']}\n")
        sys.exit(1)
    
    # Output structured results for validator (MUST be to stdout with flush)
    details = result.get("details", [])
    
    if not details:
        sys.stderr.write("[Error] No task details returned\n")
        sys.exit(1)
    
    # Print structured output blocks
    for task_result in details:
        task_id = task_result.get("task_id", "unknown")
        steps_taken = task_result.get("steps", 0)
        score = task_result.get("grader_score", 0.0)
        history = task_result.get("history", [])
        
        # REQUIRED: Print START block
        print(f"[START] task={task_id}", flush=True)
        sys.stdout.flush()
        
        # REQUIRED: Print STEP blocks for each step
        for step_entry in history:
            step_num = step_entry.get("step", 0)
            reward = step_entry.get("reward", 0.0)
            print(f"[STEP] step={step_num} reward={reward}", flush=True)
            sys.stdout.flush()
        
        # REQUIRED: Print END block
        print(f"[END] task={task_id} score={score} steps={steps_taken}", flush=True)
        sys.stdout.flush()
    
    # Summary info
    average_score = result.get("average_score", 0.0)
    completion_rate = result.get("efficiency", {}).get("completion_rate", "0/0")
    print(f"# Average Score: {average_score:.3f} | Completion: {completion_rate}", flush=True)
    sys.stdout.flush()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
