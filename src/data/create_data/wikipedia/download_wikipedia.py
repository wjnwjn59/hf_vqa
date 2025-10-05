import os
from datasets import load_dataset
import argparse
from pathlib import Path

def download_wikipedia_dataset(output_dir: str = None, cache_dir: str = None):
    """
    Download English Wikipedia dataset from Hugging Face
    
    Args:
        output_dir (str): Directory to save the dataset (default is current directory)
        cache_dir (str): Cache directory for Hugging Face datasets
    """
    
    # Set up output path
    if output_dir is None:
        output_dir = "src/data/create_data/wikipedia"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading English Wikipedia dataset...")
    print(f"Dataset will be saved to: {output_path}")
    
    try:
        # Load English Wikipedia dataset
        dataset = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.en",  # English version from November 1, 2023
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print(f"Dataset downloaded successfully!")
        print(f"Number of samples in train: {len(dataset['train'])}")
        
        # Save dataset to local storage
        dataset_path = output_path / "wikipedia_en_20231101"
        dataset.save_to_disk(str(dataset_path))
        
        print(f"Dataset saved to: {dataset_path}")
        
        # Print dataset information
        print("\n=== Dataset Information ===")
        print(f"Dataset structure: {dataset}")
        print(f"Column names: {dataset['train'].column_names}")
        
        # Display first example
        print("\n=== First Example ===")
        example = dataset['train'][0]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"{key}: {value[:200]}...")
            else:
                print(f"{key}: {value}")
                
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Download English Wikipedia dataset from Hugging Face")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./src/data/create_data/wikipedia",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Cache directory for Hugging Face datasets"
    )
    
    args = parser.parse_args()
    
    print("=== Starting Wikipedia Dataset Download ===")
    dataset = download_wikipedia_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    print("=== Completed! ===")

if __name__ == "__main__":
    main()