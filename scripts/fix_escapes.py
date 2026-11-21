with open('scripts/rebuild_notebooks_strict.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Make all code_cell strings raw triple quotes
text = text.replace('code_cell("""', 'code_cell(r"""')

# Fix any broken lines from earlier replacements
text = text.replace('print("\\\\nTahap 2:', 'print("\\n\\nTahap 2:')
text = text.replace('print(f"\\\\nPelatihan selesai', 'print(f"\\nPelatihan selesai')
text = text.replace('print("\\\\nLaporan Klasifikasi Rinci:")', 'print("\\nLaporan Klasifikasi Rinci:")')
text = text.replace('print("\\\\nTop-3 sampel', 'print("\\nTop-3 sampel')

with open('scripts/rebuild_notebooks_strict.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated rebuild_notebooks_strict.py with raw string literals.")
