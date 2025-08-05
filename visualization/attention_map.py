#!/usr/bin/env python3
"""
Batch Reasoning Milestones Processing for Where2Place Dataset

This script processes the entire Where2Place dataset and generates reasoning part-specific 
attention visualizations for each image with predicted points and ground truth overlays.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
import os
import json
import re
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import traceback

# Add RoboPoint to path
sys.path.append('/path/to/robopoint')

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates, SeparatorStyle
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def extract_attention_improved_multihead_max(outputs, img_start, img_end):
    """Extract spatial attention using improved multi-head max method"""
    if not (hasattr(outputs, 'attentions') and outputs.attentions):
        return None
    
    attention_maps = []
    
    # Use last few layers (they tend to be most semantic)
    for layer_idx in [-3, -2, -1]:
        layer_attn = outputs.attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
        
        if layer_attn.shape[-1] > img_end:
            # Get attention from last tokens to image region
            seq_len = layer_attn.shape[-1]
            
            # Use last 5 tokens (likely reasoning/answer tokens)
            text_positions = list(range(max(img_end, seq_len-10), seq_len))
            
            for text_pos in text_positions:
                # Get attention from this position to image tokens
                img_attn = layer_attn[0, :, text_pos, img_start:img_end]  # [heads, 576]
                
                # Take max across heads (most attentive head)
                max_attn = torch.max(img_attn, dim=0)[0]
                
                # Reshape to spatial
                spatial_dim = int(np.sqrt(len(max_attn)))
                if spatial_dim * spatial_dim == len(max_attn):
                    spatial_attn = max_attn.reshape(spatial_dim, spatial_dim)
                    attention_maps.append(spatial_attn.cpu().numpy())
    
    if attention_maps:
        # Combine multiple attention maps
        combined = np.stack(attention_maps, axis=0)
        
        # Use max across all maps
        final_attention = np.max(combined, axis=0)
        
        # Better normalization
        if final_attention.max() > final_attention.min():
            final_attention = (final_attention - final_attention.min()) / (final_attention.max() - final_attention.min())
        
        return final_attention
    
    return None


def generate_reasoning_with_attention_milestones(model, tokenizer, image_tensor, image, question):
    """Generate reasoning and extract attention at key milestones"""
    # Use the structured prompt that works
    reasoning_prompt = '''

Please provide your answer in the following structured format:

**Reasoning Process:**
1. **Identify the reference object:** [Explain what reference objects you see]
2. **Define the target area:** [Describe the spatial area you need to identify]  
3. **Determine the goal's subtype:** [Classify the task type]
4. **Generate the output:** [Explain how you determine coordinates]

**Final Answer:**
[Your coordinate answer]'''
    
    qs = question + reasoning_prompt
    
    if DEFAULT_IMAGE_TOKEN not in qs:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    # Find image token positions
    image_token_positions = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
    if len(image_token_positions) > 0:
        img_start = image_token_positions[0].item()
        img_end = img_start + 576
    else:
        return None, {}
    
    # Generate reasoning step by step with attention tracking
    attention_milestones = {}
    
    with torch.inference_mode():
        # Use the same generation approach that works
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=50,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Get the complete generated sequence
        full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Extract attention at different stages of the complete sequence
        generated_tokens = output_ids[0][input_ids.shape[1]:]
        total_generated = len(generated_tokens)
        
        if total_generated > 0:
            # Extract attention at 5 key milestones during reasoning
            milestone_names = [
                'identify_reference',
                'define_target', 
                'determine_subtype',
                'generate_output',
                'final_answer'
            ]
            
            milestone_positions = [
                int(total_generated * 0.2),   # 20% through
                int(total_generated * 0.4),   # 40% through  
                int(total_generated * 0.6),   # 60% through
                int(total_generated * 0.8),   # 80% through
                total_generated - 1           # End
            ]
            
            for milestone_name, pos in zip(milestone_names, milestone_positions):
                # Build sequence up to this milestone
                current_sequence = torch.cat([input_ids, generated_tokens[:pos+1].unsqueeze(0)], dim=1)
                
                # Extract attention at this point
                outputs = model(
                    input_ids=current_sequence,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    output_attentions=True,
                    return_dict=True
                )
                
                attention_map = extract_attention_improved_multihead_max(outputs, img_start, img_end)
                
                if attention_map is not None:
                    attention_milestones[milestone_name] = attention_map
    
    return full_output, attention_milestones


def parse_coordinates_from_response(response):
    """Extract coordinate points from the model response"""
    coordinates = []
    
    # Pattern for bounding box: (min_x, max_x, min_y, max_y)
    bbox_pattern = r'\(([0-9\.\s]+),\s*([0-9\.\s]+),\s*([0-9\.\s]+),\s*([0-9\.\s]+)\)'
    bbox_matches = re.findall(bbox_pattern, response)
    
    for match in bbox_matches:
        try:
            coords = [float(x.strip()) for x in match]
            if len(coords) == 4 and all(0 <= c <= 1 for c in coords):
                # Convert bounding box to corner points
                min_x, max_x, min_y, max_y = coords
                # Add the 4 corners of the bounding box
                coordinates.extend([
                    (min_x, min_y), (max_x, min_y),  # Top corners
                    (min_x, max_y), (max_x, max_y)   # Bottom corners
                ])
                break  # Use first valid bounding box
        except ValueError:
            continue
    
    # Also look for individual coordinate pairs if no bounding box found
    if not coordinates:
        pair_pattern = r'\(([0-9\.\s]+),\s*([0-9\.\s]+)\)'
        pair_matches = re.findall(pair_pattern, response)
        
        for match in pair_matches:
            try:
                x, y = float(match[0].strip()), float(match[1].strip())
                if 0 <= x <= 1 and 0 <= y <= 1:
                    coordinates.append((x, y))
            except ValueError:
                continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_coords = []
    for coord in coordinates:
        coord_rounded = (round(coord[0], 3), round(coord[1], 3))
        if coord_rounded not in seen:
            seen.add(coord_rounded)
            unique_coords.append(coord)
    
    return unique_coords


def load_ground_truth_mask(image_path):
    """Load ground truth mask if available"""
    mask_path = image_path.replace('images', 'masks')
    
    # Try both .jpg and .png extensions
    for ext in ['.jpg', '.png']:
        mask_path_with_ext = os.path.splitext(mask_path)[0] + ext
        if os.path.exists(mask_path_with_ext):
            try:
                mask = Image.open(mask_path_with_ext).convert('RGB')
                return mask
            except Exception as e:
                continue
    return None


def parse_reasoning_content(full_response):
    """Parse the actual reasoning content from the response"""
    reasoning_content = {
        'identify_reference': '',
        'define_target': '',
        'determine_subtype': '',
        'generate_output': '',
        'final_answer': ''
    }
    
    # Look for structured reasoning sections
    if "**Reasoning Process:**" in full_response and "**Final Answer:**" in full_response:
        # Extract numbered sections
        numbered_sections = re.findall(r'(\d+)\.\s*\*\*(.*?)\*\*(.*?)(?=\d+\.\s*\*\*|Final Answer|$)', 
                                     full_response, re.DOTALL)
        
        for num, title, content in numbered_sections:
            title_clean = title.lower().strip()
            content_clean = content.strip()
            
            if 'identify' in title_clean and 'reference' in title_clean:
                reasoning_content['identify_reference'] = f"{num}. **{title.strip()}:** {content_clean}"
            elif 'define' in title_clean and 'target' in title_clean:
                reasoning_content['define_target'] = f"{num}. **{title.strip()}:** {content_clean}"
            elif 'determine' in title_clean:
                reasoning_content['determine_subtype'] = f"{num}. **{title.strip()}:** {content_clean}"
            elif 'generate' in title_clean and 'output' in title_clean:
                reasoning_content['generate_output'] = f"{num}. **{title.strip()}:** {content_clean}"
        
        # Extract final answer
        final_match = re.search(r'\*\*Final Answer:\*\*(.*)', full_response, re.DOTALL)
        if final_match:
            reasoning_content['final_answer'] = f"5. **Final Answer:** {final_match.group(1).strip()}"
    
    return reasoning_content


def save_individual_milestone_images(image, question, full_response, attention_milestones, output_dir, question_id):
    """Save individual milestone images with descriptive names and transparent overlays"""
    
    # Parse reasoning content for filenames
    reasoning_content = parse_reasoning_content(full_response)
    
    # Parse predicted coordinates for final image
    predicted_coords = parse_coordinates_from_response(full_response)
    
    # Load ground truth mask
    ground_truth_mask = load_ground_truth_mask(image.name) if hasattr(image, 'name') else None
    
    # Create subfolder for this image
    image_folder = os.path.join(output_dir, 'individual_milestones', f'image_{question_id:02d}')
    os.makedirs(image_folder, exist_ok=True)
    
    # Resize image for processing
    img_array = np.array(image.resize((224, 224)))
    
    milestone_keys = [
        'identify_reference',
        'define_target',
        'determine_subtype', 
        'generate_output',
        'final_answer'
    ]
    
    saved_files = []
    
    for i, milestone_key in enumerate(milestone_keys):
        # Create figure for this milestone
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Show base image
        ax.imshow(img_array)
        
        if milestone_key in attention_milestones:
            attention_map = attention_milestones[milestone_key]
            
            # Upscale attention to image size
            attention_upscaled = cv2.resize(attention_map.astype(np.float32), (224, 224), 
                                          interpolation=cv2.INTER_CUBIC)
            attention_upscaled = cv2.GaussianBlur(attention_upscaled, (5, 5), 1.0)
            
            # Create transparent overlay - only show attention, make background transparent
            attention_colored = plt.cm.hot(attention_upscaled)  # Apply hot colormap
            
            # Set alpha based on attention intensity - high attention = opaque, low attention = transparent
            attention_colored[:, :, 3] = attention_upscaled * 0.8  # Scale alpha by attention intensity
            
            # Overlay with transparent background
            ax.imshow(attention_colored)
        
        # Add special overlays for final answer image
        if milestone_key == 'final_answer':
            # Add ground truth mask if available
            if ground_truth_mask:
                gt_mask_resized = np.array(ground_truth_mask.resize((224, 224)))
                gt_binary = np.any(gt_mask_resized > 50, axis=2)
                
                # Create transparent cyan overlay for ground truth
                gt_overlay = np.zeros((*gt_binary.shape, 4))  # RGBA
                gt_overlay[gt_binary] = [0, 1, 1, 0.3]  # Cyan with 30% opacity
                ax.imshow(gt_overlay)
            
            # Add predicted points
            if predicted_coords:
                for coord in predicted_coords:
                    x_img = coord[0] * 224
                    y_img = coord[1] * 224
                    ax.plot(x_img, y_img, 'ro', markersize=6, markerfacecolor='red', 
                           markeredgecolor='white', markeredgewidth=2)
        
        ax.axis('off')
        
        # Generate filename from reasoning content
        if reasoning_content[milestone_key]:
            # Clean the reasoning text for filename
            reasoning_text = reasoning_content[milestone_key]
            # Remove markdown and limit length
            filename_text = re.sub(r'\*\*', '', reasoning_text)
            filename_text = re.sub(r'[<>:"/\\|?*]', '', filename_text)  # Remove invalid filename chars
            filename_text = filename_text[:120] + '...' if len(filename_text) > 120 else filename_text
            filename = f"{filename_text}.png"
        else:
            filename = f"{i+1}. {milestone_key.replace('_', ' ').title()}.png"
        
        # Save individual image
        output_path = os.path.join(image_folder, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='none', transparent=True)
        plt.close()
        
        saved_files.append(output_path)
        print(f"💾 Saved milestone {i+1}: {filename[:60]}...")
    
    return saved_files


def create_milestone_visualization(image, question, full_response, attention_milestones, output_path, question_id):
    """Create visualization showing attention at reasoning milestones with predicted points and ground truth"""
    # Parse predicted coordinates from response
    predicted_coords = parse_coordinates_from_response(full_response)
    
    # Load ground truth mask
    ground_truth_mask = load_ground_truth_mask(image.name) if hasattr(image, 'name') else None
    
    # Create figure with 2 rows x 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    milestone_titles = [
        'Input Image + Question',
        'Identify Reference Object',
        'Define Target Area',
        'Determine Goal Subtype', 
        'Generate Output',
        'Final Answer + Results'
    ]
    
    milestone_keys = [
        None,  # Input image
        'identify_reference',
        'define_target',
        'determine_subtype', 
        'generate_output',
        'final_answer'
    ]
    
    # Resize image for visualization
    img_array = np.array(image.resize((224, 224)))
    
    for i, (title, key) in enumerate(zip(milestone_titles, milestone_keys)):
        ax = axes[i]
        
        if i == 0:
            # Input image with question
            ax.imshow(img_array)
            ax.set_title(f'{title}\nImage {question_id:02d}', fontsize=12, fontweight='bold')
            
            # Add question text (truncated)
            question_short = question[:80] + "..." if len(question) > 80 else question
            ax.text(0.5, -0.12, f"Q: {question_short}", transform=ax.transAxes,
                   fontsize=8, ha='center', va='top', wrap=True,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                   
        elif i == 5:  # Final answer panel with results
            # Show image with attention, predicted points, and ground truth
            ax.imshow(img_array)
            
            if key in attention_milestones:
                attention_map = attention_milestones[key]
                
                # Upscale attention
                attention_upscaled = cv2.resize(attention_map.astype(np.float32), (224, 224), 
                                              interpolation=cv2.INTER_CUBIC)
                attention_upscaled = cv2.GaussianBlur(attention_upscaled, (5, 5), 1.0)
                
                # Overlay attention with lower alpha
                ax.imshow(attention_upscaled, cmap='hot', alpha=0.4)
            
            # Overlay ground truth mask if available
            if ground_truth_mask:
                gt_mask_resized = np.array(ground_truth_mask.resize((224, 224)))
                gt_binary = np.any(gt_mask_resized > 50, axis=2)
                
                gt_overlay = np.zeros_like(img_array)
                gt_overlay[gt_binary] = [0, 255, 255]  # Cyan for ground truth
                ax.imshow(gt_overlay, alpha=0.3)
            
            # Overlay predicted points
            if predicted_coords:
                for j, coord in enumerate(predicted_coords):
                    x_img = coord[0] * 224
                    y_img = coord[1] * 224
                    
                    ax.plot(x_img, y_img, 'ro', markersize=6, markerfacecolor='red', 
                           markeredgecolor='white', markeredgewidth=1)
            
            ax.set_title(f'{title}\n({len(predicted_coords)} predicted points)', fontsize=12, fontweight='bold')
            
        else:
            # Regular attention visualization panels
            ax.imshow(img_array)
            
            if key in attention_milestones:
                attention_map = attention_milestones[key]
                
                # Upscale attention
                attention_upscaled = cv2.resize(attention_map.astype(np.float32), (224, 224), 
                                              interpolation=cv2.INTER_CUBIC)
                attention_upscaled = cv2.GaussianBlur(attention_upscaled, (5, 5), 1.0)
                
                # Overlay attention
                ax.imshow(attention_upscaled, cmap='hot', alpha=0.6)
                
                # Add stats
                stats_text = f"std: {attention_map.std():.3f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=8, va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            else:
                # No attention available
                ax.text(0.5, 0.5, 'No attention\navailable', transform=ax.transAxes,
                       fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.8))
            
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    # Add overall title
    plt.suptitle(f'TRACE Reasoning: Image {question_id:02d} - Attention Evolution with Results', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Save visualization
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return len(predicted_coords), ground_truth_mask is not None


def load_questions(questions_file):
    """Load questions from JSONL file"""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions


def process_single_image(model, tokenizer, image_processor, question_data, dataset_dir, output_dir):
    """Process a single image and return results"""
    question_id = question_data['question_id']
    image_filename = question_data['image']
    question_text = question_data['text']
    category = question_data['category']
    
    try:
        # Load image
        image_path = os.path.join(dataset_dir, 'images', image_filename)
        image = Image.open(image_path).convert('RGB')
        image.name = image_path  # Store path for ground truth loading
        image_tensor = process_images([image], image_processor, model.config)[0]
        
        # Generate reasoning with milestone attention
        full_response, attention_milestones = generate_reasoning_with_attention_milestones(
            model, tokenizer, image_tensor, image, question_text
        )
        
        if full_response and attention_milestones:
            # Create main visualization
            viz_output_path = os.path.join(output_dir, 'visualizations', f'reasoning_milestones_{question_id:02d}.png')
            os.makedirs(os.path.dirname(viz_output_path), exist_ok=True)
            
            num_predicted_points, has_ground_truth = create_milestone_visualization(
                image, question_text, full_response, attention_milestones, viz_output_path, question_id
            )
            
            # Save individual milestone images with descriptive names
            individual_files = save_individual_milestone_images(
                image, question_text, full_response, attention_milestones, output_dir, question_id
            )
            
            # Parse coordinates for analysis
            predicted_coords = parse_coordinates_from_response(full_response)
            
            # Calculate attention statistics
            attention_stats = {}
            for milestone_name, attention_map in attention_milestones.items():
                attention_stats[milestone_name] = {
                    'mean': float(attention_map.mean()),
                    'std': float(attention_map.std()),
                    'max': float(attention_map.max()),
                    'min': float(attention_map.min())
                }
            
            return {
                'question_id': question_id,
                'image': image_filename,
                'category': category,
                'question': question_text,
                'response': full_response,
                'predicted_coords': predicted_coords,
                'num_predicted_points': num_predicted_points,
                'has_ground_truth': has_ground_truth,
                'attention_stats': attention_stats,
                'num_milestones': len(attention_milestones),
                'individual_files': individual_files,
                'success': True,
                'error': None
            }
        else:
            return {
                'question_id': question_id,
                'image': image_filename,
                'category': category,
                'success': False,
                'error': 'Failed to generate response or extract attention'
            }
            
    except Exception as e:
        return {
            'question_id': question_id,
            'image': image_filename,
            'category': category,
            'success': False,
            'error': str(e)
        }


def create_summary_dashboard(results, output_dir):
    """Create summary dashboard with dataset-wide statistics"""
    print("📊 Creating summary dashboard...")
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful results to analyze")
        return
    
    # Create summary statistics
    summary_stats = {
        'total_images': len(results),
        'successful_processes': len(successful_results),
        'success_rate': len(successful_results) / len(results),
        'seen_category': len([r for r in successful_results if r['category'] == 'seen']),
        'unseen_category': len([r for r in successful_results if r['category'] == 'unseen']),
        'avg_predicted_points': np.mean([r['num_predicted_points'] for r in successful_results]),
        'ground_truth_coverage': len([r for r in successful_results if r['has_ground_truth']]) / len(successful_results)
    }
    
    # Aggregate attention statistics across milestones
    milestone_aggregated_stats = {}
    milestone_names = ['identify_reference', 'define_target', 'determine_subtype', 'generate_output', 'final_answer']
    
    for milestone in milestone_names:
        milestone_stats = []
        for result in successful_results:
            if 'attention_stats' in result and milestone in result['attention_stats']:
                milestone_stats.append(result['attention_stats'][milestone])
        
        if milestone_stats:
            milestone_aggregated_stats[milestone] = {
                'mean_std': np.mean([s['std'] for s in milestone_stats]),
                'mean_max': np.mean([s['max'] for s in milestone_stats]),
                'std_variance': np.std([s['std'] for s in milestone_stats])
            }
    
    # Create visualization dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Success rate pie chart
    ax1 = plt.subplot(3, 3, 1)
    success_data = [summary_stats['successful_processes'], 
                   summary_stats['total_images'] - summary_stats['successful_processes']]
    ax1.pie(success_data, labels=['Success', 'Failed'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Processing Success Rate')
    
    # 2. Category distribution
    ax2 = plt.subplot(3, 3, 2)
    category_data = [summary_stats['seen_category'], summary_stats['unseen_category']]
    ax2.pie(category_data, labels=['Seen', 'Unseen'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Category Distribution')
    
    # 3. Attention std evolution across milestones
    ax3 = plt.subplot(3, 3, 3)
    milestone_stds = [milestone_aggregated_stats[m]['mean_std'] for m in milestone_names if m in milestone_aggregated_stats]
    milestone_labels = [m.replace('_', ' ').title() for m in milestone_names if m in milestone_aggregated_stats]
    ax3.plot(milestone_labels, milestone_stds, 'o-', linewidth=2, markersize=8)
    ax3.set_title('Attention Standard Deviation Evolution')
    ax3.set_ylabel('Mean Std')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Number of predicted points distribution
    ax4 = plt.subplot(3, 3, 4)
    point_counts = [r['num_predicted_points'] for r in successful_results]
    ax4.hist(point_counts, bins=10, alpha=0.7, edgecolor='black')
    ax4.set_title('Distribution of Predicted Points')
    ax4.set_xlabel('Number of Points')
    ax4.set_ylabel('Frequency')
    
    # 5. Ground truth coverage
    ax5 = plt.subplot(3, 3, 5)
    gt_data = [len([r for r in successful_results if r['has_ground_truth']]),
              len([r for r in successful_results if not r['has_ground_truth']])]
    ax5.pie(gt_data, labels=['Has GT', 'No GT'], autopct='%1.1f%%', startangle=90)
    ax5.set_title('Ground Truth Coverage')
    
    # 6. Attention variance across milestones (heatmap style)
    ax6 = plt.subplot(3, 3, 6)
    variance_matrix = []
    for milestone in milestone_names:
        if milestone in milestone_aggregated_stats:
            variance_matrix.append([
                milestone_aggregated_stats[milestone]['mean_std'],
                milestone_aggregated_stats[milestone]['mean_max'],
                milestone_aggregated_stats[milestone]['std_variance']
            ])
    
    if variance_matrix:
        im = ax6.imshow(variance_matrix, cmap='hot', aspect='auto')
        ax6.set_xticks(range(3))
        ax6.set_xticklabels(['Mean Std', 'Mean Max', 'Std Variance'])
        ax6.set_yticks(range(len(variance_matrix)))
        ax6.set_yticklabels([m.replace('_', ' ').title() for m in milestone_names if m in milestone_aggregated_stats])
        ax6.set_title('Attention Statistics Heatmap')
        plt.colorbar(im, ax=ax6)
    
    # 7-9. Sample successful visualizations (thumbnails)
    sample_results = successful_results[:3]
    for idx, result in enumerate(sample_results):
        ax = plt.subplot(3, 3, 7 + idx)
        viz_path = os.path.join(output_dir, 'visualizations', f'reasoning_milestones_{result["question_id"]:02d}.png')
        if os.path.exists(viz_path):
            sample_img = plt.imread(viz_path)
            ax.imshow(sample_img)
            ax.set_title(f'Sample: Image {result["question_id"]:02d}')
        else:
            ax.text(0.5, 0.5, f'Sample {idx+1}\nNot Available', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.suptitle(f'Where2Place Dataset: TRACE Reasoning Analysis Summary\n'
                f'{summary_stats["successful_processes"]}/{summary_stats["total_images"]} images processed successfully '
                f'({summary_stats["success_rate"]*100:.1f}%)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, 'dataset_summary_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics
    detailed_stats = {
        'summary': summary_stats,
        'milestone_stats': milestone_aggregated_stats,
        'processing_timestamp': datetime.now().isoformat(),
        'successful_results': successful_results
    }
    
    stats_path = os.path.join(output_dir, 'detailed_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(detailed_stats, f, indent=2)
    
    print(f"✅ Summary dashboard saved: {dashboard_path}")
    print(f"✅ Detailed statistics saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch process Where2Place dataset with reasoning milestone attention')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model-base', type=str, required=True, help='Path to the base model')
    parser.add_argument('--dataset-dir', type=str, default='/path/to/where2place', 
                       help='Path to Where2Place dataset directory')
    parser.add_argument('--output-dir', type=str, default='where2place_reasoning_milestones_results', 
                       help='Output directory for results')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index for processing')
    parser.add_argument('--end-idx', type=int, default=None, help='End index for processing (exclusive)')
    parser.add_argument('--resume', action='store_true', help='Resume from existing results')
    
    args = parser.parse_args()
    
    print("🎯 Where2Place Dataset: Batch Reasoning Milestones Processing")
    print("=" * 80)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("🤖 Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load questions
    questions_file = os.path.join(args.dataset_dir, 'bbox_questions.jsonl')
    questions = load_questions(questions_file)
    print(f"📋 Loaded {len(questions)} questions")
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    questions_to_process = questions[start_idx:end_idx]
    
    print(f"🔄 Processing questions {start_idx} to {end_idx-1} ({len(questions_to_process)} total)")
    
    # Check for existing results if resuming
    results_file = os.path.join(args.output_dir, 'batch_results.json')
    if args.resume and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        processed_ids = {r['question_id'] for r in existing_results}
        questions_to_process = [q for q in questions_to_process if q['question_id'] not in processed_ids]
        print(f"📄 Resuming: {len(questions_to_process)} questions remaining")
        results = existing_results
    else:
        results = []
    
    # Process each question
    print("🚀 Starting batch processing...")
    for question_data in tqdm(questions_to_process, desc="Processing images"):
        result = process_single_image(
            model, tokenizer, image_processor, question_data, args.dataset_dir, args.output_dir
        )
        results.append(result)
        
        # Print progress for important milestones
        if result['success']:
            print(f"✅ Image {result['question_id']:02d}: {result['num_predicted_points']} points, "
                  f"{result['num_milestones']} milestones")
        else:
            print(f"❌ Image {result['question_id']:02d}: {result['error']}")
        
        # Save intermediate results every 10 images
        if len(results) % 10 == 0:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary dashboard
    create_summary_dashboard(results, args.output_dir)
    
    # Print final summary
    successful_count = len([r for r in results if r['success']])
    print(f"\n🎉 Batch processing complete!")
    print(f"📊 Results: {successful_count}/{len(results)} images processed successfully")
    print(f"📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()