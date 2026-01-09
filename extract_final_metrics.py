#!/usr/bin/env python3
"""
Extract final test metrics from wandb logs and save to JSON.
This script polls wandb API to get the final test accuracy after training completes.
"""

import argparse
import json
import os
import time
import sys

def extract_metrics_from_wandb(run_name, project_name, max_wait_hours=48):
    """Extract final test metrics from wandb run."""
    try:
        import wandb
        api = wandb.Api()

        # Get the run
        print(f"Fetching run: {project_name}/{run_name}")

        # Try to find the run
        max_attempts = int(max_wait_hours * 12)  # Check every 5 minutes
        for attempt in range(max_attempts):
            try:
                runs = api.runs(project_name, filters={"display_name": run_name})
                if runs:
                    run = runs[0]
                    break
            except Exception as e:
                if attempt == 0:
                    print(f"Run not found yet, will retry... ({e})")

            if attempt < max_attempts - 1:
                time.sleep(300)  # Wait 5 minutes
        else:
            print(f"ERROR: Run {run_name} not found after {max_wait_hours} hours")
            return None

        # Wait for run to complete
        print(f"Run found. Status: {run.state}")
        while run.state in ['running', 'pending']:
            print(f"Run still {run.state}, waiting...")
            time.sleep(300)  # Check every 5 minutes
            run.update()

        # Get final metrics
        history = run.scan_history()

        # Find test metrics (they have "test" prefix in the key)
        test_metrics = {}
        train_metrics = {}

        for row in history:
            # Get test accuracy
            if 'test/exact_accuracy' in row:
                test_metrics['exact_accuracy'] = row['test/exact_accuracy']
            if 'test/accuracy' in row:
                test_metrics['accuracy'] = row['test/accuracy']

            # Also capture train metrics for reference
            if 'train/exact_accuracy' in row:
                train_metrics['exact_accuracy'] = row['train/exact_accuracy']
            if 'train/accuracy' in row:
                train_metrics['accuracy'] = row['train/accuracy']

        result = {
            'run_name': run_name,
            'run_id': run.id,
            'state': run.state,
            'test_metrics': test_metrics,
            'train_metrics': train_metrics,
            'config': dict(run.config),
        }

        return result

    except ImportError:
        print("ERROR: wandb not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"ERROR extracting metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_metrics_from_checkpoint(checkpoint_path):
    """Extract metrics from checkpoint directory if available."""
    import torch

    metrics_file = os.path.join(checkpoint_path, "final_metrics.pt")
    if os.path.exists(metrics_file):
        return torch.load(metrics_file)
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract final metrics from training run")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the wandb run")
    parser.add_argument("--project_name", type=str, default="TinyRecursiveModels", help="Wandb project name")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--checkpoint_path", type=str, help="Optional checkpoint path to extract from")
    parser.add_argument("--max_wait_hours", type=int, default=48, help="Max hours to wait for run to complete")

    args = parser.parse_args()

    # Try checkpoint first if provided
    metrics = None
    if args.checkpoint_path:
        print(f"Checking checkpoint: {args.checkpoint_path}")
        metrics = extract_metrics_from_checkpoint(args.checkpoint_path)

    # Fall back to wandb
    if metrics is None:
        print("Extracting from wandb...")
        metrics = extract_metrics_from_wandb(args.run_name, args.project_name, args.max_wait_hours)

    if metrics is None:
        print("ERROR: Could not extract metrics")
        sys.exit(1)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {args.output_file}")
    print(f"Test exact accuracy: {metrics.get('test_metrics', {}).get('exact_accuracy', 'N/A')}")
    print(f"Test accuracy: {metrics.get('test_metrics', {}).get('accuracy', 'N/A')}")


if __name__ == "__main__":
    main()
