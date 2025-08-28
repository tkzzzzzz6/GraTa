#!/usr/bin/env python3
"""
评估脚本 - 用于生成可视化结果
"""

import argparse
import json
import os
from utils.visualization import create_visualization_results, EvaluationVisualizer

def load_results_from_json(json_file):
    """从JSON文件加载评估结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results_to_json(results_dict, source_domain, save_dir='./results'):
    """保存评估结果到JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(save_dir, f'{source_domain}_{timestamp}_results.json')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    parser.add_argument('--results_file', type=str, help='JSON file containing evaluation results')
    parser.add_argument('--source_domain', type=str, help='Source domain name')
    parser.add_argument('--save_dir', type=str, default='./visualization_results', 
                       help='Directory to save visualization results')
    parser.add_argument('--mode', type=str, choices=['json', 'manual'], default='json',
                       help='Mode: json (load from file) or manual (input manually)')
    
    args = parser.parse_args()
    
    if args.mode == 'json':
        if not args.results_file:
            print("Error: --results_file is required when mode is 'json'")
            return
        
        if not os.path.exists(args.results_file):
            print(f"Error: Results file {args.results_file} not found")
            return
        
        results_dict = load_results_from_json(args.results_file)
        source_domain = args.source_domain or "Unknown"
        
    else:  # manual mode
        print("Manual input mode")
        source_domain = input("Enter source domain name: ")
        
        results_dict = {}
        domains = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
        
        for domain in domains:
            if domain != source_domain:
                print(f"\nEntering results for {domain}:")
                disc_dice = float(input(f"  Disc Dice: "))
                cup_dice = float(input(f"  Cup Dice: "))
                disc_asd = float(input(f"  Disc ASSD: "))
                cup_asd = float(input(f"  Cup ASSD: "))
                
                results_dict[domain] = {
                    'Disc_Dice': disc_dice,
                    'Cup_Dice': cup_dice,
                    'Disc_ASSD': disc_asd,
                    'Cup_ASSD': cup_asd
                }
    
    # 生成可视化结果
    print(f"\nGenerating visualizations for {source_domain}...")
    create_visualization_results(results_dict, source_domain, args.save_dir)
    
    # 保存结果到JSON文件
    save_results_to_json(results_dict, source_domain)

if __name__ == '__main__':
    from datetime import datetime
    main()
