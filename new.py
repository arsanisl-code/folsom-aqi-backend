import os

# Configuration
PROJECT_DIR = 'C:\folsom_aqi_final\backend'  # Change this to your backend folder path
OUTPUT_FILE = 'master_code_context.txt'
EXCLUDE_DIRS = ['venv', '__pycache__', '.git']

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk(PROJECT_DIR):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith('.py') and file != os.path.basename(__file__):
                file_path = os.path.join(root, file)
                
                # Write a clear separator for the AI
                outfile.write(f"\n\n{'='*50}\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write(f"{'='*50}\n\n")
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

print(f"Success! All code combined into {OUTPUT_FILE}")