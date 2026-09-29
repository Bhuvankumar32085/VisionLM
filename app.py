"""Streamlit Web UI for Custom Vision-Language Model (VLM) Visual QA."""

import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple
from PIL import Image
import streamlit as st
import torch

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing import load_and_preprocess_image
from inference.generate import generate_answer
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from training.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device, get_gpu_memory_mb, get_hardware_info

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Custom VLM | Multimodal Visual QA",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS Styling (Modern Dark Glassmorphism)
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Gradient Header */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .tech-badge {
        background: rgba(139, 92, 246, 0.15);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Card Containers */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
    }

    /* Response Output Box */
    .response-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #8b5cf6;
        border-radius: 12px;
        padding: 1.25rem;
        color: #f1f5f9;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-top: 1rem;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.3);
    }

    /* Stat Box */
    .stat-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Model Loading & Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_vlm_pipeline(
    checkpoint_path: str,
    config_path: str = "configs/config.yaml",
) -> Tuple[VisionLanguageModel, VisionEncoder, LanguageModel, torch.device]:
    """Loads and caches the model architecture, image processor, and tokenizer."""
    config = load_config(config_path) if os.path.exists(config_path) else {}
    device = get_device()

    vision_cfg = config.get("vision", {})
    lang_cfg = config.get("language", {})
    proj_cfg = config.get("projector", {})

    image_processor = VisionEncoder.get_image_processor(vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"))
    tokenizer = LanguageModel.get_tokenizer(lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"))

    model = VisionLanguageModel(
        vision_model_name=vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"),
        language_model_name=lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        use_cls_token=bool(vision_cfg.get("use_cls_token", False)),
        projector_hidden_dim=int(proj_cfg.get("hidden_dim", 1536)),
        projector_activation=str(proj_cfg.get("activation", "gelu")),
        projector_dropout=float(proj_cfg.get("dropout", 0.0)),
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path=checkpoint_path, model=model)

    model.eval()
    return model, image_processor, tokenizer, device


def get_available_checkpoints() -> List[str]:
    """Scans the checkpoints directory for saved .pt files."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return []

    # Priority checkpoints
    priority = ["checkpoint_best.pt", "checkpoint_final.pt"]
    all_files = [f.name for f in ckpt_dir.glob("*.pt")]

    ordered = []
    for p in priority:
        if p in all_files:
            ordered.append(p)

    for f in sorted(all_files):
        if f not in ordered:
            ordered.append(f)

    return [str(ckpt_dir / f) for f in ordered]


# ==============================================================================
# UI Sidebar - Controls & Configuration
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
    st.markdown("### ⚙️ Model & Generation")

    # 1. Checkpoint Selection
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        st.warning("⚠️ No checkpoints found in `checkpoints/`. Using initial initialized weights.")
        selected_checkpoint = "checkpoints/checkpoint_best.pt"
    else:
        # Default to checkpoint_best.pt if available
        default_idx = 0
        selected_checkpoint = st.selectbox(
            "Select Checkpoint",
            options=checkpoints,
            index=default_idx,
            help="Trained weights for the MLP Projector.",
        )

    # Checkpoint details badge
    if os.path.exists(selected_checkpoint):
        file_size_mb = os.path.getsize(selected_checkpoint) / (1024 * 1024)
        st.caption(f"💾 Checkpoint Size: `{file_size_mb:.1f} MB`")

    st.markdown("---")

    # 2. Generation Hyperparameters
    st.markdown("### 🎛️ Hyperparameters")
    max_new_tokens = st.slider("Max New Tokens", min_value=16, max_value=256, value=96, step=16)
    
    decoding_mode = st.radio("Decoding Strategy", options=["Greedy (Deterministic)", "Sampling (Creative)"], index=0)
    do_sample = (decoding_mode == "Sampling (Creative)")

    if do_sample:
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
        top_p = st.slider("Top-P (Nucleus)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    else:
        temperature = 0.0
        top_p = 1.0

    st.markdown("---")

    # 3. Hardware & GPU Status
    st.markdown("### 🖥️ Hardware Status")
    hw = get_hardware_info()
    if hw["cuda_available"]:
        st.markdown(
            f"""
            <div class="stat-card">
                <b>GPU:</b> {hw['gpu_name']}<br>
                <b>Total VRAM:</b> {hw['total_vram_gb']:.1f} GB<br>
                <b>Allocated:</b> {get_gpu_memory_mb():.1f} MB<br>
                <b>Precision:</b> BF16 / FP16 Accelerated
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Running on CPU (CUDA not detected)")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ==============================================================================
# Main Page Layout
# ==============================================================================
st.markdown('<div class="main-title">✨ Custom Vision-Language Model</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactively inspect and test our custom-trained VLM (ViT-B/16 + MLP Projector + SmolLM2-360M) on your own images.</div>',
    unsafe_allow_html=True,
)

# Tech Stack Badges
st.markdown(
    """
    <div class="badge-container">
        <span class="tech-badge">👁️ Vision: ViT-B/16 (224px)</span>
        <span class="tech-badge">🔗 Projector: Linear(768→1536→960)</span>
        <span class="tech-badge">💬 Language: SmolLM2-360M-Instruct</span>
        <span class="tech-badge">⚡ Fusion: Direct inputs_embeds</span>
        <span class="tech-badge">🎯 Phase 1: Aligned on Real LLaVA Data</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Chat History
if "history" not in st.session_state:
    st.session_state["history"] = []

# Load Model Pipeline
with st.spinner("🚀 Loading VLM model into GPU memory..."):
    model, image_processor, tokenizer, device = load_vlm_pipeline(selected_checkpoint)

col_left, col_right = st.columns([1, 1.2], gap="large")

# ------------------------------------------------------------------------------
# Left Column: Image Upload & Input
# ------------------------------------------------------------------------------
with col_left:
    st.markdown("### 🖼️ Step 1: Upload or Choose Image")
    
    input_method = st.radio("Image Source", ["Upload Custom Image", "Sample Demo Image"], horizontal=True)
    input_image: Optional[Image.Image] = None

    if input_method == "Upload Custom Image":
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Choose an image from your computer to analyze.",
        )
        if uploaded_file is not None:
            try:
                input_image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image: {e}")
    else:
        # Check if local sample images exist
        sample_img_dir = Path("data/raw/llava/images")
        sample_imgs = list(sample_img_dir.glob("*.jpg"))[:10] if sample_img_dir.exists() else []
        
        if sample_imgs:
            selected_sample = st.selectbox(
                "Choose Demo Image",
                options=[p.name for p in sample_imgs],
            )
            img_path = sample_img_dir / selected_sample
            input_image = Image.open(img_path).convert("RGB")
        else:
            st.info("No demo images found. Please upload a custom image.")

    if input_image is not None:
        st.image(input_image, caption=f"Selected Image ({input_image.size[0]}x{input_image.size[1]})", use_container_width=True)
    else:
        st.info("👆 Please upload an image or choose a demo image to get started.")

# ------------------------------------------------------------------------------
# Right Column: Interactive QA & Generation
# ------------------------------------------------------------------------------
with col_right:
    st.markdown("### 💬 Step 2: Ask a Question")

    # Quick Preset Buttons
    st.caption("Quick question suggestions:")
    preset_col1, preset_col2 = st.columns(2)
    
    selected_preset = ""
    with preset_col1:
        if st.button("🔍 Describe in detail", use_container_width=True):
            selected_preset = "Describe this image in detail."
        if st.button("👤 What is person doing?", use_container_width=True):
            selected_preset = "What is the person doing in this image?"
            
    with preset_col2:
        if st.button("📦 What objects exist?", use_container_width=True):
            selected_preset = "What objects can you see in the photo?"
        if st.button("🏞️ Describe background", use_container_width=True):
            selected_preset = "Describe the setting, background, and environment."

    # Question Input Field
    default_prompt = selected_preset if selected_preset else "Describe what is happening in this image."
    user_question = st.text_area(
        "Enter your question:",
        value=default_prompt,
        height=90,
        placeholder="e.g. What is the color of the car? Describe the main subject...",
    )

    generate_btn = st.button("🚀 Ask VLM Model", type="primary", use_container_width=True)

    # Generation Trigger
    if generate_btn:
        if input_image is None:
            st.warning("⚠️ Please upload an image first!")
        elif not user_question.strip():
            st.warning("⚠️ Please enter a question prompt.")
        else:
            with st.spinner("🤖 VLM is analyzing image & generating response..."):
                start_time = time.time()
                
                try:
                    generated_text = generate_answer(
                        model=model,
                        image=input_image,
                        question=user_question,
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        device=device,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                    elapsed_time = time.time() - start_time

                    # Save to session history
                    st.session_state["history"].append({
                        "question": user_question,
                        "answer": generated_text,
                        "time": elapsed_time,
                        "checkpoint": Path(selected_checkpoint).name,
                    })

                    # Display Response
                    st.markdown("#### 🎯 Model Answer:")
                    st.markdown(
                        f"""
                        <div class="response-box">
                            {generated_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"⏱️ Generated in `{elapsed_time:.2f}s` using `{Path(selected_checkpoint).name}` on `{device}`")

                except Exception as e:
                    st.error(f"Generation error: {e}")

    # Conversation History
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("### 📜 Session History")
        for idx, item in enumerate(reversed(st.session_state["history"])):
            with st.expander(f"Q: {item['question']}", expanded=(idx == 0)):
                st.markdown(f"**Answer:** {item['answer']}")
                st.caption(f"Checkpoint: `{item['checkpoint']}` | Latency: `{item['time']:.2f}s`")
