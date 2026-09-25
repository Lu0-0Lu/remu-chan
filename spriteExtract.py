from psd_tools import PSDImage
from PIL import Image
import os

os.makedirs("sprites", exist_ok=True)
print("Loading PSD... This might take a moment!")
psd = PSDImage.open('sprites.psd')

# 1. FORCE VISIBILITY: Turn on the "Eye Icon" for every single folder and layer
for layer in psd.descendants():
    layer.visible = True

# 2. Get the global bounding box to crop empty space
merged = psd.composite()
global_bbox = merged.getbbox()

# 3. Extract, Align, and Crop
for layer in psd.descendants():
    try:
        # Now that it's forced visible, it will successfully grab the face pixels
        layer_img = layer.composite()
        
        if layer_img:
            full_canvas = Image.new("RGBA", psd.size)
            full_canvas.paste(layer_img, layer.offset)
            
            cropped_layer = full_canvas.crop(global_bbox)
            
            safe_name = "".join([c if c.isalnum() else "_" for c in layer.name]).lower()
            cropped_layer.save(f"sprites/{safe_name}.png")
            print(f"Exported: {safe_name}.png")
    except Exception as e:
        pass

print("Extraction Complete!")