import os
import json
import boto3
import openai
import logging
from datetime import datetime
from jinja2 import Template
from ebaysdk.trading import Connection as Trading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# AWS Rekognition client setup
rekognition_client = boto3.client('rekognition', region_name='us-west-2')

# OpenAI client setup
openai.api_key = 'your-openai-api-key'

# eBay API setup
ebay_api = Trading(config_file='ebay.yaml')  # Ensure you have your eBay API credentials in the ebay.yaml file

# Predefined category to SKU mapping
CATEGORY_SKU_MAPPING = {
    'fine jewelry': '4196',
    'engagement & wedding': '1643',
    'fine earrings': '10986',
    'fine necklaces & pendants': '164329',
    'fine rings': '164343',
    'fine bracelets': '164315',
    'fashion jewelry': '10968',
    'fashion bracelets': '50637',
    'fashion earrings': '50647',
    'fashion necklaces & pendants': '155101',
    'fashion rings': '67681',
    'vintage & antique jewelry': '48579',
    'vintage & antique bracelets': '10183',
    'vintage & antique earrings': '10192',
    'vintage & antique necklaces & pendants': '10120',
    'vintage & antique rings': '10196',
}

def analyze_images_with_rekognition(image_paths):
    keywords = []
    for image_path in image_paths:
        with open(image_path, 'rb') as image_file:
            response = rekognition_client.detect_labels(
                Image={'Bytes': image_file.read()},
                MaxLabels=10
            )
            for label in response['Labels']:
                keywords.append(label['Name'])
    return keywords

def get_sku_from_keywords(keywords):
    for keyword in keywords:
        category = keyword.lower()
        if category in CATEGORY_SKU_MAPPING:
            return CATEGORY_SKU_MAPPING[category]
    return 'DEFAULT-SKU'  # Default SKU if no category match is found

def generate_text_data(keywords, sku):
    prompt = (
        f"Generate an eBay listing title, description, and item specifics for a product with the following keywords: "
        f"{', '.join(keywords)} and SKU: {sku}. "
        "The title should be no more than 80 characters. "
        "Provide a detailed description and relevant item specifics."
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()

def fill_html_template(title, description, specifics, template_path='listing_template.html'):
    with open(template_path, 'r') as file:
        template = Template(file.read())
    return template.render(title=title, description=description, specifics=specifics)

def process_folder(folder_path, output_dir):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if len(image_files) < 2:
        logging.error(f"Not enough images in {folder_path} to analyze.")
        return

    image_paths = [os.path.join(folder_path, image_files[i]) for i in range(2)]
    keywords = analyze_images_with_rekognition(image_paths)
    sku = get_sku_from_keywords(keywords)
    text_data = generate_text_data(keywords, sku)

    # Assume text_data contains title, description, and specifics in a structured format
    text_data_json = json.loads(text_data)
    title = text_data_json['title']
    description = text_data_json['description']
    specifics = text_data_json['specifics']

    html_content = fill_html_template(title, description, specifics)

    output_file_path = os.path.join(output_dir, f"{sku}_listing.html")
    with open(output_file_path, 'w') as output_file:
        output_file.write(html_content)
    
    logging.info(f"Processed {folder_path} and saved data to {output_file_path}")

    upload_to_ebay(title, description, specifics, sku, output_file_path)

def upload_to_ebay(title, description, specifics, sku, html_path):
    with open(html_path, 'r') as html_file:
        description_html = html_file.read()
    
    item = {
        "Title": title,
        "Description": description_html,
        "PrimaryCategory": {"CategoryID": specifics['category_id']},
        "StartPrice": "100.00",  # You can customize this
        "ConditionID": "1000",  # New
        "CategoryMappingAllowed": "true",
        "Country": "US",
        "Currency": "USD",
        "DispatchTimeMax": "3",
        "ListingDuration": "GTC",
        "ListingType": "FixedPriceItem",
        "PaymentMethods": "PayPal",
        "PayPalEmailAddress": "you@example.com",
        "PictureDetails": {"PictureURL": ["http://example.com/picture1.jpg"]},  # Add your image URLs
        "PostalCode": "95125",
        "Quantity": "1",
        "ReturnPolicy": {
            "ReturnsAcceptedOption": "ReturnsAccepted",
            "RefundOption": "MoneyBack",
            "ReturnsWithinOption": "Days_30",
            "ShippingCostPaidByOption": "Buyer"
        },
        "ShippingDetails": {
            "ShippingType": "Flat",
            "ShippingServiceOptions": [
                {
                    "ShippingServicePriority": "1",
                    "ShippingService": "USPSMedia",
                    "ShippingServiceCost": "2.50"
                }
            ]
        },
        "Site": "US",
        "SKU": sku,
        "ItemSpecifics": {
            "NameValueList": [
                {"Name": "Brand", "Value": specifics.get('brand', '')},
                {"Name": "Style", "Value": specifics.get('style', '')},
                {"Name": "Metal", "Value": specifics.get('metal', '')},
                # Add more specifics as needed
            ]
        }
    }

    response = ebay_api.execute('AddItem', {'Item': item})
    if response.reply.Ack == 'Success':
        logging.info(f"Successfully listed item {sku} on eBay.")
    else:
        logging.error(f"Failed to list item {sku} on eBay. Response: {response.reply}")

def main():
    input_dir = '/home/robertmcasper/image_processing/image_processing/output'
    output_dir = '/home/robertmcasper/image_processing/listings'

    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = [os.path.join(input_dir, d) for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]

    for folder_path in dirs:
        try:
            process_folder(folder_path, output_dir)
        except Exception as exc:
            logging.error(f"Error processing folder {folder_path}: {exc}")

if __name__ == "__main__":
    main()
