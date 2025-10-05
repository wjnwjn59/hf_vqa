import os
import re
from datasets import load_from_disk, Dataset
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json
from tqdm import tqdm


class WikipediaIntroExtractor:
    """Class to extract introductions from Wikipedia articles"""
    
    # Keywords to filter out articles about history and politics
    EXCLUDED_CATEGORIES = {
        'history', 'histories', 'historical',
        'politics', 'political', 'politicians', 'politician',
        'wars', 'war', 'battles', 'battle',
        'elections', 'election', 'electoral',
        'government', 'governments', 'governmental',
        'military', 'army', 'armies',
        'empires', 'empire', 'imperial',
        'revolutions', 'revolution', 'revolutionary',
        'conflicts', 'conflict',
        'diplomacy', 'diplomatic',
        'presidents', 'president', 'presidential',
        'prime ministers', 'prime minister',
        'kings', 'king', 'queens', 'queen', 'monarchy',
        'dynasties', 'dynasty'
    }
    
    def __init__(self, min_words: int = 1024):
        """
        Args:
            min_words (int): Minimum number of words for the introduction
        """
        self.min_words = min_words
    
    def extract_categories(self, text: str) -> List[str]:
        """
        Extract categories from the end of a Wikipedia article
        
        Args:
            text (str): Article content
            
        Returns:
            List[str]: List of categories
        """
        # Get the last ~30 lines of text
        lines = text.strip().split('\n')
        last_lines = lines[-30:] if len(lines) > 30 else lines
        
        categories = []
        # The last lines are usually categories, one per line
        for line in reversed(last_lines):
            line = line.strip()
            # If line is empty or too long, we might have passed the categories section
            if not line:
                continue
            if len(line) > 100:  # Categories are usually short
                break
            # Skip lines with special characters or long sentences
            if any(char in line for char in ['==', '===', '{{', '}}', '[[', ']]', '|']):
                continue
            if line.endswith('.') or line.endswith(','):
                continue
            
            categories.append(line.lower())
        
        return categories
    
    def should_exclude_article(self, categories: List[str]) -> bool:
        """
        Check if an article should be excluded (about history or politics)
        
        Args:
            categories (List[str]): List of article categories
            
        Returns:
            bool: True if should exclude, False if keep
        """
        for category in categories:
            category_lower = category.lower()
            # Check if category contains any excluded keyword
            for excluded in self.EXCLUDED_CATEGORIES:
                if excluded in category_lower:
                    return True
        return False
    
    def remove_reference_sections(self, text: str) -> str:
        """
        Remove reference sections like "See also", "References", "Further reading", etc.
        
        Args:
            text (str): Article content
            
        Returns:
            str: Text with reference sections removed
        """
        # List of section titles to remove (and everything after them)
        reference_sections = [
            'see also',
            'references',
            'further reading',
            'external links',
            'bibliography',
            'sources',
            'notes',
            'citations',
            'footnotes',
            'works cited'
        ]
        
        # Find the earliest occurrence of any reference section
        min_position = len(text)
        
        for section in reference_sections:
            # Look for patterns like "\n== Section Name ==" or "\n=== Section Name ==="
            patterns = [
                rf'\n==\s*{re.escape(section)}\s*==',
                rf'\n===\s*{re.escape(section)}\s*===',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.start() < min_position:
                    min_position = match.start()
        
        # If we found any reference section, cut the text there
        if min_position < len(text):
            text = text[:min_position].strip()
        
        return text
    
    def extract_introduction(self, text: str) -> Optional[str]:
        """
        Extract introduction from a Wikipedia article
        Introduction is the text before the first section (before ==)
        
        Args:
            text (str): Article content
            
        Returns:
            Optional[str]: Introduction text or None if invalid
        """
        # First, remove reference sections from the entire text
        text = self.remove_reference_sections(text)
        
        # Find the position of the first section (== marker)
        section_pattern = r'\n==\s*[^=]'
        match = re.search(section_pattern, text)
        
        if match:
            intro = text[:match.start()].strip()
        else:
            # If no section found, use entire text
            intro = text.strip()
        
        # Count words
        word_count = len(intro.split())
        
        # If intro is too short and entire article is also shorter than min_words
        if word_count < self.min_words:
            total_word_count = len(text.split())
            if total_word_count < self.min_words:
                # Use entire article (already cleaned of references)
                intro = text.strip()
                word_count = total_word_count
            else:
                # Intro is short but article is long -> skip
                return None
        
        return intro
    
    def process_dataset(
        self, 
        dataset_path: str,
        output_path: str,
        max_samples: Optional[int] = None,
        save_format: str = 'json'
    ):
        """
        Process Wikipedia dataset and extract introductions
        
        Args:
            dataset_path (str): Path to the downloaded dataset
            output_path (str): Path to save results
            max_samples (Optional[int]): Maximum number of samples to extract (None = all)
            save_format (str): Format to save ('json' or 'jsonl')
        """
        print(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        
        print(f"Dataset loaded: {len(dataset['train'])} articles")
        
        extracted_articles = []
        total_processed = 0
        total_excluded_by_category = 0
        total_excluded_by_length = 0
        total_extracted = 0
        
        print(f"Processing articles...")
        if max_samples:
            print(f"Will stop after extracting {max_samples} valid samples")
        
        # Process until we have enough samples or run out of articles
        for idx in tqdm(range(len(dataset['train'])), desc="Extracting introductions"):
            # Stop if we've reached the desired number of samples
            if max_samples and total_extracted >= max_samples:
                break
                
            article = dataset['train'][idx]
            total_processed += 1
            
            text = article.get('text', '')
            title = article.get('title', f'Article_{idx}')
            article_id = article.get('id', str(idx))
            
            if not text:
                continue
            
            # Extract categories
            categories = self.extract_categories(text)
            
            # Check if should exclude
            if self.should_exclude_article(categories):
                total_excluded_by_category += 1
                continue
            
            # Extract introduction
            intro = self.extract_introduction(text)
            
            if intro is None:
                total_excluded_by_length += 1
                continue
            
            # Save article with only required fields
            extracted_articles.append({
                'id': article_id,
                'title': title,
                'text': intro,
                'categories': categories
            })
            total_extracted += 1
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'json':
            output_file = output_path.with_suffix('.json')
            print(f"\nSaving results to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_articles, f, ensure_ascii=False, indent=2)
        elif save_format == 'jsonl':
            output_file = output_path.with_suffix('.jsonl')
            print(f"\nSaving results to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in extracted_articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {save_format}")
        
        # Print statistics
        print("\n=== Statistics ===")
        print(f"Total articles processed: {total_processed}")
        print(f"Articles excluded by category (history/politics): {total_excluded_by_category}")
        print(f"Articles excluded by length: {total_excluded_by_length}")
        print(f"Articles successfully extracted: {total_extracted}")
        print(f"Results saved to: {output_file}")
        
        # Display example
        if extracted_articles:
            print("\n=== First Article Example ===")
            example = extracted_articles[0]
            print(f"ID: {example['id']}")
            print(f"Title: {example['title']}")
            print(f"Word count: {len(example['text'].split())}")
            print(f"Categories: {example['categories'][:5]}")  # Show first 5 categories
            print(f"Text preview (first 200 chars): {example['text'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Extract introductions from Wikipedia dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./src/data/create_data/wikipedia/wikipedia_en_20231101",
        help="Path to the downloaded Wikipedia dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./src/data/create_data/wikipedia/wikipedia_processed",
        help="Path to save results"
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=1024,
        help="Minimum number of words for the introduction"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to extract (None = all)"
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=['json', 'jsonl'],
        default='json',
        help="Format to save results"
    )
    
    args = parser.parse_args()
    
    print("=== Starting Wikipedia Introduction Extraction ===")
    
    extractor = WikipediaIntroExtractor(min_words=args.min_words)
    extractor.process_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        save_format=args.save_format
    )
    
    print("\n=== Completed! ===")


if __name__ == "__main__":
    main()
