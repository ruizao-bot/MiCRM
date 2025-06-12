with open('seeds.txt', 'w') as f:
    for i in range(100):
        f.write(f"{i}\n")

print("seeds.txt generated successfully!")