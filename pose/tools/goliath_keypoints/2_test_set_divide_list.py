import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def split_into_chunks(data, num_chunks):
    chunk_size = len(data) // num_chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    # If there are leftover elements, add them to the last chunk
    if len(data) % num_chunks != 0:
        chunks[-1].extend(data[chunk_size * num_chunks:])
    return chunks

def save_chunks(chunks, root_dir, prefix):
    chunk_dir = os.path.join(root_dir, 'chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(chunk_dir, f'{prefix}_{i}.txt')
        with open(chunk_file, 'w') as file:
            file.writelines(chunk)

# Read files
ROOT_DIR = '/mnt/home/rawalk/drive/mmpose/data/goliath/test'
num_chunks = 35
images = read_file(os.path.join(ROOT_DIR, 'images.txt'))
keypoints = read_file(os.path.join(ROOT_DIR, 'keypoints.txt'))

# Split data into chunks
image_chunks = split_into_chunks(images, num_chunks)
keypoint_chunks = split_into_chunks(keypoints, num_chunks)

# Save chunks
save_chunks(image_chunks, ROOT_DIR, 'images')
save_chunks(keypoint_chunks, ROOT_DIR, 'keypoints')
