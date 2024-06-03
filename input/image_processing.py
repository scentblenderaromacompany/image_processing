import os
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import pyheif
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPPORTED_IMAGE_TYPES = ['.heic', '.jpg', '.jpeg', '.png', '.tiff']
FONT_PATH = '/home/robertmcasper/image_processing/fonts/GreatVibes-Regular.ttf'
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TARGET_SIZE = (1024, 768)

def initialize_metadata_file(metadata_path):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['ProductID', 'FileName', 'Type', 'Width', 'Height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def update_metadata_file(metadata_path, product_id, file_name, image_type, width, height):
    with open(metadata_path, 'a', newline='') as csvfile:
        fieldnames = ['ProductID', 'FileName', 'Type', 'Width', 'Height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'ProductID': product_id, 'FileName': file_name, 'Type': image_type, 'Width': width, 'Height': height})

def enhance_image(image):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(image, -1, sharpen_kernel)
    return enhanced_image

def rotate_image(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def add_watermark(image, text="Eternal Elegance Emporium", font_path=FONT_PATH):
    try:
        font = ImageFont.truetype(font_path, 16)
        logging.info(f"Using font from {font_path}")
    except IOError:
        logging.error("Font file not found or cannot be opened. Using default font.")
        try:
            font = ImageFont.truetype(DEFAULT_FONT_PATH, 16)
            logging.info("Using default font.")
        except IOError:
            logging.error("Default font file not found. Using default PIL font.")
            font = ImageFont.load_default()

    watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark, 'RGBA')
    width, height = image.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = width - text_width - 10, height - text_height - 10
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))
    return Image.alpha_composite(image.convert('RGBA'), watermark)

def convert_to_supported_format(file_path):
    try:
        image = Image.open(file_path)
        new_file_path = os.path.splitext(file_path)[0] + ".png"
        image.save(new_file_path)
        return new_file_path
    except UnidentifiedImageError as e:
        logging.error(f"Failed to convert {file_path}: {e}")
        return None

def process_image(file_path, output_dir, product_id, image_index, image_type, font_path):
    try:
        if image_type == '.heic':
            heif_file = pyheif.read(file_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            image = Image.open(file_path)

        image = rotate_image(image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        enhanced_image = enhance_image(image_cv)
        resized_image = cv2.resize(enhanced_image, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image_pil = Image.fromarray(resized_image_rgb)
        resized_image_pil = add_watermark(resized_image_pil, font_path=font_path)
        product_dir = os.path.join(output_dir, f"Product_{product_id:05d}")
        os.makedirs(product_dir, exist_ok=True)
        output_file_name = f"Product_{product_id:05d}_Image_{image_index:02d}.png"
        output_file_path = os.path.join(product_dir, output_file_name)
        resized_image_pil.save(output_file_path, quality=95, optimize=True)
        metadata_path = os.path.join(output_dir, "metadata", "processed_images.csv")
        update_metadata_file(metadata_path, product_id, output_file_name, image_type, resized_image_pil.width, resized_image_pil.height)
        logging.info(f"Processed {file_path} as {output_file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return False

def create_summary_file(output_dir, product_count):
    summary_path = os.path.join(output_dir, "metadata", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"Summary of Image Processing\n")
        summary_file.write(f"Date and Time: {current_time}\n")
        summary_file.write(f"Total Products Processed: {product_count}\n")
    logging.info(f"Summary file created at {summary_path}")

def process_directory(product_dir, output_dir, product_id, font_path):
    image_index = 1
    processed_count = 0

    for file in sorted(os.listdir(product_dir)):
        file_ext = os.path.splitext(file.lower())[1]
        if file_ext not in SUPPORTED_IMAGE_TYPES:
            file_path = convert_to_supported_format(os.path.join(product_dir, file))
            if file_path is None:
                continue
            file_ext = '.png'
        else:
            file_path = os.path.join(product_dir, file)

        if process_image(file_path, output_dir, product_id, image_index, file_ext, font_path):
            image_index += 1

    if image_index > 1:
        processed_count += 1

    return processed_count

def main():
    input_dir = '/home/robertmcasper/image_processing/input'
    output_dir = '/home/robertmcasper/image_processing/output'
    metadata_path = os.path.join(output_dir, "metadata", "processed_images.csv")
    font_path = '/home/robertmcasper/image_processing/fonts/GreatVibes-Regular.ttf'

    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    if not os.path.exists(output_dir):
        logging.error(f"Output directory does not exist: {output_dir}")
        return

    initialize_metadata_file(metadata_path)
    product_id = 1
    processed_product_count = 0

    dirs = [os.path.join(root, d) for root, dnames, _ in os.walk(input_dir) for d in sorted(dnames)]
    
    for d in dirs:
        try:
            processed_count = process_directory(d, output_dir, product_id, font_path)
            processed_product_count += processed_count
            product_id += 1
        except Exception as exc:
            logging.error(f"Directory processing generated an exception: {exc}")

    create_summary_file(output_dir, processed_product_count)

if __name__ == "__main__":
    main()
