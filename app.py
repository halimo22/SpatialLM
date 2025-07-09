import torch
from transformers import AutoTokenizer
from peft import PeftModel
from spatiallm.pcd.pcd_loader import Compose, load_o3d_pcd, cleanup_pcd, get_points_and_colors
from spatiallm import Layout
import numpy as np
import open3d as o3d
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from spatiallm.model.spatiallm_llama import  SpatialLMLlamaForCausalLM
from transformers import TextIteratorStreamer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import logging
from threading import Thread


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_DIR = "models/spatial_lm_lora"  # Base model directory (without LoRA)
LORA_ADAPTER_DIR = "models/spatial_lm_lora/Adapter"  # Your LoRA adapter checkpoint
TEMPLATE_PATH = "code_template.txt"  # path to class template
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
base_model_for_merge = SpatialLMLlamaForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    
logger.info("Step 5: Loading LoRA adapter and merging with base model...")
peft_model = PeftModel.from_pretrained(
        base_model_for_merge,
        LORA_ADAPTER_DIR,
        device_map="auto"
    )
    
model = peft_model.merge_and_unload()
# Load class template
class_template = None
if os.path.exists(TEMPLATE_PATH):
    with open(TEMPLATE_PATH) as f:
        class_template = f.read().strip()
else:
    logger.warning(f"Template file not found at {TEMPLATE_PATH}")

app = FastAPI(title="SpatialLM API", version="1.0.0")



def preprocess_point_cloud(points, colors, grid_size, num_bins):
    transform = Compose(
        [
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=num_bins,
            ),
        ]
    )
    point_cloud = transform(
        {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
    )
    coord = point_cloud["grid_coord"]
    xyz = point_cloud["coord"]
    rgb = point_cloud["color"]
    point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
    return torch.as_tensor(np.stack([point_cloud], axis=0))


def generate_layout(
    model,
    point_cloud,
    tokenizer,
    top_k=10,
    top_p=0.95,
    temperature=0.6,
    num_beams=1,
    max_new_tokens=4096,
    prompt=None,
):  
    if(prompt is None):
        prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {class_template}"
    else:
        prompt = f"<|point_start|><|point_pad|><|point_end|> {prompt}"

    conversation = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(model, 'clear_cache'):
        model.clear_cache()

    generate_kwargs = dict(
        {"input_ids": input_ids, "point_clouds": point_cloud},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    print("Generating layout...\n")
    generate_texts = []
    for text in streamer:
        generate_texts.append(text)
        print(text, end="", flush=True)
    t.join()
    print("\nDone!")

    layout_str = "".join(generate_texts)

    return layout_str

    



@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": DEVICE}

@app.post("/predict")
async def predict_custom(
    file: UploadFile = File(..., description="PLY file containing point cloud data"),
    prompt: str = Form(None, description="Custom prompt for the model (optional)")
):
    """
    Process PLY file with optional custom prompt.
    The point cloud tokens will be automatically added to the prompt.
    """
    if not file.filename.endswith('.ply'):
        raise HTTPException(status_code=400, detail="File must be a PLY file")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        logger.info(f"Processing file: {file.filename}")
        
        # Load and process point cloud
        point_cloud = load_o3d_pcd(tmp_file_path)
        point_cloud = cleanup_pcd(point_cloud)
        points, colors = get_points_and_colors(point_cloud)
        logger.info(f"Point cloud loaded with {len(points)} points")
        
        min_extent = np.min(points, axis=0)
        
        grid_size = Layout.get_grid_size()
        num_bins = Layout.get_num_bins()
        point_cloud_np = preprocess_point_cloud(points, colors, grid_size, num_bins)
        
        logger.info(f"Point cloud numpy shape: {point_cloud_np.shape}")
        
        # Generate layout using the exact working pattern
        grid_size = Layout.get_grid_size()
        num_bins = Layout.get_num_bins()
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

        # generate the layout
        layout = generate_layout(
            model=model,
            point_cloud=input_pcd,
            tokenizer=tokenizer,
            prompt=prompt
        )
        

        logger.info(f"Layout {layout}")
        
        return layout
        
    except Exception as e:
        logger.error(f"Error processing point cloud: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing point cloud: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# === Error handlers ===
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)