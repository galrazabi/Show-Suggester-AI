# Show-Suggester-AI

This repository contains a Python program that recommends TV shows based on user input and descriptions. It utilizes the OpenAI API for text embeddings and chat completions.

## Features

- Reads TV show data from a CSV file.
- Creates embeddings vectors for each TV show description.
- Allows users to input their favorite TV shows and generates recommendations.
- Provides unit tests for core functionalities.

## Data Source

The TV show data used in this project is sourced from Kaggle.

## Usage

1. Clone the repository
2. Install dependencies
3. Set up environment variables:
   
  Create a `.env` file and add your OpenAI API key:
  OPENAI_API_KEY=your-api-key-here
  
4. Run the script:
  python main.py

Follow the prompts to get recommendations!

## Testing

To run tests:
python test_show_suggester.py












