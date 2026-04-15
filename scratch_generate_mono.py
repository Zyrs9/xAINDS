import os

output_file = 'project_mono_file.txt'
exclude_dirs = {'.git', '__pycache__', 'nvp_baseline'}
exclude_exts = {'.pkl', '.csv', '.pdf'}

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if any(file.endswith(ext) for ext in exclude_exts):
                continue
            if file == output_file or file == 'scratch_generate_mono.py':
                continue
            
            filepath = os.path.join(root, file)
            filepath_clean = filepath.replace('.\\', '')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
            except Exception as e:
                continue
                
            outfile.write('='*80 + '\n')
            outfile.write(f'File: {filepath_clean}\n')
            outfile.write('='*80 + '\n\n')
            outfile.write(content + '\n\n')

print("Mono file generated successfully.")
