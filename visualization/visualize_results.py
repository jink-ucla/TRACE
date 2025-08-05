#!/usr/bin/env python3

from PIL import Image
from matplotlib import colormaps
from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import json
import numpy as np
import os
import re
import random


def find_vectors(text, width=640, height=480):
    """Extract vectors and points from model output text"""
    vectors, points = [], []
    
    # Remove any text before "Final Answer:" to focus on coordinates
    if "Final Answer:" in text:
        # Handle both "Final Answer:" and "**Final Answer:**" formats
        if "**Final Answer:**" in text:
            text = text.split("**Final Answer:**")[-1].strip()
        else:
            text = text.split("Final Answer:")[-1].strip()
    
    # Pattern 1: Standard tuple format (x, y) - multiple tuples in list
    pattern1 = r"\(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\)"
    matches1 = re.finditer(pattern1, text)
    for match in matches1:
        x, y = match.groups()
        x = float(x) if '.' in x else int(x)
        y = float(y) if '.' in y else int(y)
        if isinstance(x, float) or isinstance(y, float):
            x = int(x * width)
            y = int(y * height)
        vectors.append((x, y))
        points.append((x, y))
    
    # Pattern 2: Single coordinate in square brackets [x, y]
    pattern2 = r"\[([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\]"
    matches2 = re.finditer(pattern2, text)
    for match in matches2:
        x, y = match.groups()
        x = float(x) if '.' in x else int(x)
        y = float(y) if '.' in y else int(y)
        if isinstance(x, float) or isinstance(y, float):
            x = int(x * width)
            y = int(y * height)
        vectors.append((x, y))
        points.append((x, y))
    
    # Pattern 3: Mixed format like [x, y), (x, y] - handle broken brackets
    pattern3 = r"\[([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\)"
    matches3 = re.finditer(pattern3, text)
    for match in matches3:
        x, y = match.groups()
        x = float(x) if '.' in x else int(x)
        y = float(y) if '.' in y else int(y)
        if isinstance(x, float) or isinstance(y, float):
            x = int(x * width)
            y = int(y * height)
        vectors.append((x, y))
        points.append((x, y))
    
    # Pattern 4: Simple coordinate pairs without brackets (but not in already matched areas)
    # This should be the last pattern to catch any remaining coordinates
    pattern4 = r"([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)"
    matches4 = re.finditer(pattern4, text)
    for match in matches4:
        start, end = match.span()
        # Check if this match overlaps with any already processed patterns
        overlap = False
        for pattern_matches in [matches1, matches2, matches3]:
            for pm in pattern_matches:
                pm_start, pm_end = pm.span()
                if (start >= pm_start and start < pm_end) or (end > pm_start and end <= pm_end):
                    overlap = True
                    break
            if overlap:
                break
        
        if not overlap:
            x, y = match.groups()
            x = float(x) if '.' in x else int(x)
            y = float(y) if '.' in y else int(y)
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            vectors.append((x, y))
            points.append((x, y))
    
    return vectors, np.array(points)


def word_wrap(s, n):
    """Wrap text to n words per line"""
    words = s.split()
    for i in range(n, len(words), n):
        words[i-1] += '\n'
    return ' '.join(words)


def filter_unseen_questions(questions, num_questions=30, seed=42):
    """Filter and sample unseen category questions for where2place(h)"""
    unseen_questions = [q for q in questions if q.get('category') == 'unseen']
    
    print(f"🔍 Found {len(unseen_questions)} 'unseen' category questions for visualization")
    
    if len(unseen_questions) == 0:
        print("⚠️ No 'unseen' category questions found. Available categories:")
        categories = set(q.get('category', 'no_category') for q in questions)
        for cat in sorted(categories):
            count = len([q for q in questions if q.get('category') == cat])
            print(f"   - {cat}: {count} questions")
        return [], []
    
    # Sample the requested number of questions
    if len(unseen_questions) > num_questions:
        random.seed(seed)
        selected_questions = random.sample(unseen_questions, num_questions)
        print(f"✅ Sampled {num_questions} from {len(unseen_questions)} unseen questions")
    else:
        selected_questions = unseen_questions
        print(f"✅ Using all {len(selected_questions)} unseen questions")
    
    # Get indices of unseen questions in the original dataset
    unseen_indices = []
    for unseen_q in selected_questions:
        for i, q in enumerate(questions):
            if q.get('question_id') == unseen_q.get('question_id') and q.get('image') == unseen_q.get('image'):
                unseen_indices.append(i)
                break
    
    return selected_questions, unseen_indices


def visualize_comparison(answer_files, labels, data_dir, question_file, output_dir, num_questions=None, where2place_h=False, where2place_h_count=30):
    """Create visualizations comparing different model outputs"""
    
    # Load questions
    with open(f"{data_dir}/{question_file}", 'r') as file:
        questions = [json.loads(line) for line in file]
    
    # Determine which questions to visualize
    if where2place_h:
        # Use unseen category questions for where2place(h)
        selected_questions, question_indices = filter_unseen_questions(questions, where2place_h_count)
        if not selected_questions:
            print("❌ No unseen questions found for where2place(h) visualization")
            return
        viz_questions = selected_questions
        suffix = "_where2place_h"
        print(f"🎨 Creating WHERE2PLACE(H) visualizations for {len(viz_questions)} unseen questions")
    else:
        # Use standard questions for where2place
        if num_questions is None:
            num_questions = len(questions)
        viz_questions = questions[:num_questions]
        question_indices = list(range(len(viz_questions)))
        suffix = "_where2place"
        print(f"🎨 Creating WHERE2PLACE visualizations for {len(viz_questions)} questions")
    
    colors = colormaps['Set1']
    answers = {}
    
    # Load all answer files
    for fname, label in zip(answer_files, labels):
        with open(fname, 'r') as file:
            all_answers = [json.loads(line) for line in file]
            
        # Filter answers according to the question indices
        if where2place_h:
            # For where2place(h), select answers corresponding to unseen questions
            filtered_answers = [all_answers[i] for i in question_indices if i < len(all_answers)]
        else:
            # For standard where2place, use first num_questions answers
            filtered_answers = all_answers[:len(viz_questions)]
            
        answers[label] = filtered_answers
    
    # Create output directory with suffix
    final_output_dir = output_dir + suffix
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📁 Output directory: {final_output_dir}")

    for idx, question in enumerate(tqdm(viz_questions, desc=f"Creating {'where2place(h)' if where2place_h else 'where2place'} visualizations")):
        # Get the original question index for mask loading
        if where2place_h:
            original_idx = question_indices[idx]
        else:
            original_idx = idx
            
        img_path = f"{data_dir}/images/{question['image']}"
        mask_path = f"{data_dir}/masks/{question['question_id']}.png"
        
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue
            
        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found: {mask_path}")
            continue
        
        img = Image.open(img_path)
        mask = np.array(Image.open(mask_path)) / 255.
        
        labels_with_acc, vectors = ['Ground Truth'], {}
        
        # Process each model's predictions
        for key, ans_list in answers.items():
            try:
                if idx < len(ans_list):
                    vectors[key], pts = find_vectors(ans_list[idx]['text'])
                else:
                    print(f"⚠️ No answer available for {key} at index {idx}")
                    pts = []
                    vectors[key] = []
            except Exception as e:
                print(f'Failed to parse answer for question {idx} from {key}: {e}')
                pts = []
                vectors[key] = []
            
            # Calculate accuracy
            acc = 0
            if len(pts) > 0:
                in_range = (pts[:, 0] >= 0) & (pts[:, 0] < mask.shape[1]) \
                         & (pts[:, 1] >= 0) & (pts[:, 1] < mask.shape[0])
                acc = np.concatenate([
                    mask[pts[in_range, 1], pts[in_range, 0]],
                    np.zeros(pts.shape[0] - in_range.sum())
                ]).mean()
            labels_with_acc.append(f"{key}\nAccuracy: {acc:.2f}")
        
        # Create the plot
        fig, ax = plt.subplots(1, figsize=(12, 8))
        plt.imshow(img)
        
        # Show ground truth mask
        colored_mask = np.ones_like(mask)[..., None] * colors(0)
        colored_mask[..., 3] = mask * 0.6
        plt.imshow(colored_mask)
        plt.axis('off')
        
        # Draw predictions from each model
        for i, (model_name, vec) in enumerate(vectors.items()):
            for v in vec:
                if len(v) == 2:
                    x, y = v
                    c1 = patches.Circle((x, y), 3, color=colors(i+1), fill=True)
                    ax.add_patch(c1)
                    c2 = patches.Circle((x, y), 10, color=colors(i+1), fill=False)
                    ax.add_patch(c2)
                elif len(v) == 4:
                    x0, y0, x1, y1 = v
                    rect = patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, linewidth=2,
                        edgecolor=colors(i+1), facecolor='none'
                    )
                    ax.add_patch(rect)
        
        # Create legend
        handles = [
            patches.Patch(color=colors(i), label=label)
            for i, label in enumerate(labels_with_acc)
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.005, 1))
        
        # Set title with benchmark name
        q_txt = question['text'].split("Your answer should")[0].strip()
        if '<image>' in q_txt:
            q_txt = q_txt.split('\n')[1]
        
        benchmark_name = "WHERE2PLACE(H)" if where2place_h else "WHERE2PLACE"
        title = f"[{benchmark_name}] {word_wrap(q_txt, 15)}"
        plt.suptitle(title, y=0.995)
        plt.subplots_adjust(left=0.005, right=0.83, bottom=0.01, top=0.955)
        
        # Save the plot
        img_name = question['image'].split('/')[-1]
        img_path = f"{final_output_dir}/{idx:03d}_{img_name}"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ {'WHERE2PLACE(H)' if where2place_h else 'WHERE2PLACE'} visualizations saved to: {final_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Where2Place evaluation results")
    parser.add_argument("--answer-files", nargs='+', required=True,
                        help="Paths to model output files (.jsonl)")
    parser.add_argument("--labels", nargs='+', required=True,
                        help="Labels for each model (same order as answer-files)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to Where2Place dataset directory")
    parser.add_argument("--question-file", default="point_questions.jsonl",
                        help="Question file name within data directory")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Number of questions to visualize (default: all)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for visualization images")
    
    # New where2place(h) options
    parser.add_argument("--where2place-h", action="store_true",
                        help="Create visualizations for where2place(h) unseen category")
    parser.add_argument("--where2place-h-count", type=int, default=30,
                        help="Number of unseen questions for where2place(h)")
    parser.add_argument("--create-both", action="store_true",
                        help="Create both where2place and where2place(h) visualizations")
    
    args = parser.parse_args()

    if len(args.answer_files) != len(args.labels):
        print("Error: Number of answer files must match number of labels")
        exit(1)

    if args.create_both:
        # Create both where2place and where2place(h) visualizations
        print("🎨 Creating both WHERE2PLACE and WHERE2PLACE(H) visualizations")
        
        # Standard where2place
        print("\n" + "="*60)
        print("🎨 Creating WHERE2PLACE visualizations")
        visualize_comparison(
            args.answer_files, 
            args.labels, 
            args.data_dir, 
            args.question_file, 
            args.output_dir, 
            args.num_questions,
            where2place_h=False
        )
        
        # where2place(h)
        print("\n" + "="*60)
        print("🎨 Creating WHERE2PLACE(H) visualizations")
        visualize_comparison(
            args.answer_files, 
            args.labels, 
            args.data_dir, 
            args.question_file, 
            args.output_dir, 
            args.num_questions,
            where2place_h=True,
            where2place_h_count=args.where2place_h_count
        )
    else:
        # Create only the requested type
        visualize_comparison(
            args.answer_files, 
            args.labels, 
            args.data_dir, 
            args.question_file, 
            args.output_dir, 
            args.num_questions,
            where2place_h=args.where2place_h,
            where2place_h_count=args.where2place_h_count
        )