# generate_seeds.py
with open('seeds.txt', 'w') as f:
    for i in range(50,70):
        f.write(f"{i}\n")

print("seeds.txt generated successfully!")
