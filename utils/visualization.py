import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EvaluationVisualizer:
    """评估结果可视化类"""
    
    def __init__(self, save_dir='./visualization_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_metrics_comparison(self, results_dict, title="Metrics Comparison Across Domains", save_name=None):
        """
        绘制不同域之间的指标对比图
        
        Args:
            results_dict: 格式为 {domain_name: {metric_name: value}}
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        metrics = ['Disc_Dice', 'Cup_Dice', 'Disc_ASSD', 'Cup_ASSD']
        metric_names = ['Disc Dice', 'Cup Dice', 'Disc ASSD', 'Cup ASSD']
        
        domains = list(results_dict.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            
            values = [results_dict[domain].get(metric, 0) for domain in domains]
            
            bars = ax.bar(domains, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score' if 'Dice' in metric else 'Distance')
            ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)
            
            # 在柱状图上添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_heatmap(self, results_dict, title="Performance Heatmap", save_name=None):
        """
        绘制性能热力图
        
        Args:
            results_dict: 格式为 {domain_name: {metric_name: value}}
            title: 图表标题
            save_name: 保存文件名
        """
        # 转换为DataFrame
        df = pd.DataFrame(results_dict).T
        
        # 标准化数值（Dice分数越高越好，ASSD越低越好）
        df_normalized = df.copy()
        for col in df.columns:
            if 'Dice' in col:
                df_normalized[col] = df[col] / 100.0  # Dice分数归一化到0-1
            else:
                # ASSD距离，需要反转（距离越小越好）
                max_val = df[col].max()
                df_normalized[col] = 1 - (df[col] / max_val)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_normalized, annot=df.round(3), fmt='.3f', 
                   cmap='RdYlGn', center=0.5, 
                   cbar_kws={'label': 'Normalized Performance'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Target Domains', fontsize=12)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_radar_chart(self, results_dict, title="Performance Radar Chart", save_name=None):
        """
        绘制雷达图
        
        Args:
            results_dict: 格式为 {domain_name: {metric_name: value}}
            title: 图表标题
            save_name: 保存文件名
        """
        metrics = ['Disc_Dice', 'Cup_Dice', 'Disc_ASSD', 'Cup_ASSD']
        metric_labels = ['Disc Dice', 'Cup Dice', 'Disc ASSD', 'Cup ASSD']
        
        # 标准化数据
        domains = list(results_dict.keys())
        data_normalized = {}
        
        for domain in domains:
            data_normalized[domain] = []
            for metric in metrics:
                value = results_dict[domain].get(metric, 0)
                if 'Dice' in metric:
                    normalized_value = value / 100.0  # Dice分数归一化
                else:
                    # ASSD距离反转并归一化
                    max_asd = max([results_dict[d].get(metric, 0) for d in domains])
                    normalized_value = 1 - (value / max_asd) if max_asd > 0 else 0
                data_normalized[domain].append(normalized_value)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        
        for idx, (domain, values) in enumerate(data_normalized.items()):
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=domain, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, y=1.08, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_segmentation_results(self, image, pred_mask, true_mask, title="Segmentation Results", save_name=None):
        """
        绘制分割结果对比图
        
        Args:
            image: 原始图像 (H, W, C)
            pred_mask: 预测掩码 (H, W)
            true_mask: 真实掩码 (H, W)
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # 真实掩码
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('Ground Truth', fontweight='bold')
        axes[1].axis('off')
        
        # 预测掩码
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Prediction', fontweight='bold')
        axes[2].axis('off')
        
        # 叠加显示
        overlay = image.copy()
        overlay[pred_mask > 0] = [255, 0, 0]  # 红色表示预测区域
        overlay[true_mask > 0] = [0, 255, 0]  # 绿色表示真实区域
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay (Red: Pred, Green: GT)', fontweight='bold')
        axes[3].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_curves(self, metrics_history, title="Training Progress", save_name=None):
        """
        绘制训练曲线
        
        Args:
            metrics_history: 格式为 {metric_name: [values]}
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        metrics = ['Disc_Dice', 'Cup_Dice', 'Disc_ASSD', 'Cup_ASSD']
        metric_names = ['Disc Dice', 'Cup Dice', 'Disc ASSD', 'Cup ASSD']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx//2, idx%2]
            
            if metric in metrics_history:
                values = metrics_history[metric]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, 'o-', linewidth=2, markersize=4)
                ax.set_title(f'{metric_name} Progress', fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Score' if 'Dice' in metric else 'Distance')
                ax.grid(True, alpha=0.3)
                
                # 添加趋势线
                if len(values) > 1:
                    z = np.polyfit(epochs, values, 1)
                    p = np.poly1d(z)
                    ax.plot(epochs, p(epochs), "--", alpha=0.8, color='red')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_evaluation_report(self, results_dict, source_domain, save_name=None):
        """
        生成详细的评估报告
        
        Args:
            results_dict: 评估结果字典
            source_domain: 源域名称
            save_name: 保存文件名
        """
        report = f"""
# 评估报告

## 基本信息
- 源域: {source_domain}
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 目标域数量: {len(results_dict)}

## 详细结果

"""
        
        # 计算平均分数
        avg_disc_dice = np.mean([results_dict[domain].get('Disc_Dice', 0) for domain in results_dict])
        avg_cup_dice = np.mean([results_dict[domain].get('Cup_Dice', 0) for domain in results_dict])
        avg_disc_asd = np.mean([results_dict[domain].get('Disc_ASSD', 0) for domain in results_dict])
        avg_cup_asd = np.mean([results_dict[domain].get('Cup_ASSD', 0) for domain in results_dict])
        
        report += f"""
## 平均性能
- 平均 Disc Dice: {avg_disc_dice:.3f}
- 平均 Cup Dice: {avg_cup_dice:.3f}
- 平均 Disc ASSD: {avg_disc_asd:.3f}
- 平均 Cup ASSD: {avg_cup_asd:.3f}

## 各域详细结果

"""
        
        for domain, metrics in results_dict.items():
            report += f"""
### {domain}
- Disc Dice: {metrics.get('Disc_Dice', 0):.3f}
- Cup Dice: {metrics.get('Cup_Dice', 0):.3f}
- Disc ASSD: {metrics.get('Disc_ASSD', 0):.3f}
- Cup ASSD: {metrics.get('Cup_ASSD', 0):.3f}

"""
        
        if save_name:
            with open(os.path.join(self.save_dir, f'{save_name}.md'), 'w', encoding='utf-8') as f:
                f.write(report)
        
        print(report)
        return report

def create_visualization_results(results_dict, source_domain, save_dir='./visualization_results'):
    """
    创建完整的可视化结果
    
    Args:
        results_dict: 评估结果字典
        source_domain: 源域名称
        save_dir: 保存目录
    """
    visualizer = EvaluationVisualizer(save_dir)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{source_domain}_{timestamp}"
    
    # 创建各种可视化
    visualizer.plot_metrics_comparison(results_dict, 
                                      title=f"Metrics Comparison - {source_domain}",
                                      save_name=f"{base_name}_comparison")
    
    visualizer.plot_heatmap(results_dict,
                           title=f"Performance Heatmap - {source_domain}",
                           save_name=f"{base_name}_heatmap")
    
    visualizer.plot_radar_chart(results_dict,
                               title=f"Performance Radar - {source_domain}",
                               save_name=f"{base_name}_radar")
    
    visualizer.generate_evaluation_report(results_dict, source_domain,
                                        save_name=f"{base_name}_report")
    
    print(f"可视化结果已保存到: {save_dir}")
    print(f"生成的文件:")
    print(f"- {base_name}_comparison.png (指标对比图)")
    print(f"- {base_name}_heatmap.png (性能热力图)")
    print(f"- {base_name}_radar.png (雷达图)")
    print(f"- {base_name}_report.md (评估报告)")
