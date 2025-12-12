"""
Gradio UI for model inference - SIMPLIFIED COMPATIBLE VERSION
"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio
import matplotlib.pyplot as plt

from src.utils.logger import PipelineLogger

logger = PipelineLogger.get_logger(__name__)


class ModelInferenceUI:
    """Interactive UI for model inference - SIMPLIFIED VERSION"""
    
    def __init__(self, config: Dict, all_models: Dict, prepared_data: Dict, 
                 all_results: Dict, best_models: Dict):
        """
        Initialize UI
        
        Args:
            config: Configuration dictionary
            all_models: All trained models (keys are tuples: (architecture, dataset))
            prepared_data: Prepared datasets
            all_results: All experiment results
            best_models: Best models per dataset
        """
        self.config = config
        self.all_models = all_models
        self.prepared_data = prepared_data
        self.all_results = all_results
        self.best_models = best_models
        
        # Create mapping from model name to model key
        self.model_name_to_key = {}
        for key, model in self.all_models.items():
            self.model_name_to_key[model.name] = key
        
        logger.info(f"UI initialized with {len(self.all_models)} models")
    
    def create_ui(self):
        """Create Gradio UI interface"""
        try:
            import gradio as gr
        except ImportError:
            logger.info("Installing Gradio...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
            import gradio as gr
        
        def predict(image: np.ndarray, selected_model_name: str) -> Tuple[str, plt.Figure]:
            """Predict using selected model"""
            if image is None:
                return "Please upload or select an image", None
            
            try:
                # Get model by name
                if selected_model_name not in self.model_name_to_key:
                    return f"Model '{selected_model_name}' not found", None
                
                model_key = self.model_name_to_key[selected_model_name]
                model = self.all_models[model_key]
                
                # Preprocess image
                # Convert to grayscale if RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_gray = np.mean(image, axis=2, keepdims=True)
                elif len(image.shape) == 2:
                    image_gray = np.expand_dims(image, axis=2)
                else:
                    image_gray = image
                
                # Normalize to 0-1
                image_normalized = image_gray.astype(np.float32) / 255.0
                image_flat = image_normalized.reshape(1, -1)
                
                # Make prediction
                pred_class, probabilities = model.predict(image_flat)
                
                # Create visualization figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Show input image
                ax1.imshow(image_gray.squeeze(), cmap='gray')
                ax1.set_title(f'Input Image')
                ax1.axis('off')
                
                # Show probabilities
                classes = list(range(10))
                bars = ax2.bar(classes, probabilities[0], color='skyblue')
                ax2.set_xlabel('Class')
                ax2.set_ylabel('Probability')
                ax2.set_title(f'Class Probabilities - {model.name}')
                ax2.set_xticks(classes)
                ax2.set_ylim([0, 1])
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Highlight predicted class
                predicted_idx = pred_class[0]
                bars[predicted_idx].set_color('red')
                
                # Add probability values on bars
                for i, prob in enumerate(probabilities[0]):
                    ax2.text(i, prob + 0.02, f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                
                # Create result text
                result_text = f"🎯 **Model:** {model.name}\n\n"
                result_text += f"📊 **Architecture:** {model_key[0]}\n"
                result_text += f"📁 **Dataset:** {model_key[1]}\n"
                result_text += f"🔢 **Layers:** {model.layer_sizes}\n"
                result_text += f"🧮 **Parameters:** {model.num_params:,}\n\n"
                
                # Add accuracy if available
                if model_key in self.all_results:
                    acc = self.all_results[model_key].get('test_accuracy')
                    if acc:
                        result_text += f"✅ **Test Accuracy:** {acc:.2%}\n\n"
                
                result_text += f"🎯 **Predicted Class:** {pred_class[0]}\n"
                result_text += f"📈 **Confidence:** {np.max(probabilities[0]):.2%}\n\n"
                result_text += "📊 **Top 3 Predictions:**\n"
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities[0])[-3:][::-1]
                for idx in top_indices:
                    result_text += f"  Class {idx}: {probabilities[0][idx]:.4f}\n"
                
                # Check if this is a best model
                for dataset, best_info in self.best_models.items():
                    if best_info.get('name') == model.name:
                        result_text += f"\n🏆 **Best model for {dataset.upper()}**\n"
                
                return result_text, fig
                
            except Exception as e:
                import traceback
                error_msg = f"❌ **Error during prediction:** {str(e)}\n"
                logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
                return error_msg, None
        
        # Create model choices from all models
        model_choices = list(self.model_name_to_key.keys())
        
        # Get sample images for examples
        example_images = self._get_example_images()
        example_labels = []
        
        # Create the UI
        with gr.Blocks(title="DNN Model Inference Dashboard") as demo:
            gr.Markdown("# 🧠 DNN Model Inference Dashboard")
            gr.Markdown("Test your trained neural networks on images")
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        label="Select Model",
                        value=model_choices[0] if model_choices else None
                    )
                    
                    image_input = gr.Image(
                        label="Upload or Draw Image",
                        type="numpy",
                        sources=["upload", "clipboard"]  # Removed webcam for compatibility
                    )
                    
                    predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    output_text = gr.Markdown(label="Prediction Results")
                    output_plot = gr.Plot(label="Visualization")
            
            # Add examples if available
            if example_images:
                gr.Markdown("### Sample Images")
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Click on sample images to test"
                )
            
            # Add model information
            with gr.Accordion("📋 Model Information", open=False):
                model_info = "## Available Models:\n\n"
                for i, (model_name, model_key) in enumerate(self.model_name_to_key.items()):
                    model = self.all_models[model_key]
                    model_info += f"**{i+1}. {model_name}**\n"
                    model_info += f"  - **Architecture:** {model_key[0]}\n"
                    model_info += f"  - **Dataset:** {model_key[1]}\n"
                    model_info += f"  - **Layers:** {model.layer_sizes}\n"
                    model_info += f"  - **Parameters:** {model.num_params:,}\n"
                    
                    if model_key in self.all_results:
                        acc = self.all_results[model_key].get('test_accuracy')
                        if acc:
                            model_info += f"  - **Test Accuracy:** {acc:.2%}\n"
                    
                    # Check if this is a best model
                    is_best = False
                    for dataset, best_info in self.best_models.items():
                        if best_info.get('name') == model_name:
                            model_info += f"  - 🏆 **Best model for {dataset.upper()}**\n"
                            is_best = True
                    
                    model_info += "\n"
                
                gr.Markdown(model_info)
            
            # Add tips
            gr.Markdown("### 💡 Tips:")
            gr.Markdown("- Draw digits 0-9 for MNIST models")
            gr.Markdown("- Draw clothing items for Fashion MNIST")
            gr.Markdown("- Use simple, centered images")
            gr.Markdown("- Try different models to compare performance")
            
            # Connect prediction button
            predict_btn.click(
                fn=predict,
                inputs=[image_input, model_dropdown],
                outputs=[output_text, output_plot]
            )
        
        return demo
    
    def _get_example_images(self) -> List[np.ndarray]:
        """Get example images for the UI"""
        example_images = []
        
        if not self.prepared_data:
            return example_images
        
        # Get 2-3 examples from each dataset
        for dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
            if dataset_name in self.prepared_data:
                data_info = self.prepared_data[dataset_name]
                X_test = data_info['data']['X_test']
                
                # Take 1-2 sample images
                for i in range(min(2, len(X_test))):
                    img = X_test[i]
                    
                    # Handle different image formats
                    if len(img.shape) == 2:  # (H, W)
                        img = np.expand_dims(img, -1)  # Add channel
                    
                    if img.shape[-1] == 1:  # Grayscale
                        img = np.repeat(img, 3, axis=-1)  # Convert to RGB
                    
                    # Normalize to 0-255
                    img = (img * 255).astype(np.uint8)
                    example_images.append(img)
        
        return example_images
    
    def launch(self, share: bool = False):
        """Launch the UI"""
        demo = self.create_ui()
        demo.launch(share=share)