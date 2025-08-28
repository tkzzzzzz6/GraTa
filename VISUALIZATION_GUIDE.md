# 评估结果可视化指南

## 概述

本项目新增了强大的评估结果可视化功能，可以生成多种类型的图表和报告，帮助更好地理解和分析模型性能。

## 功能特性

### 1. 多种可视化图表
- **指标对比图**: 柱状图显示不同域之间的性能对比
- **性能热力图**: 热力图展示各域在各指标上的表现
- **雷达图**: 雷达图显示综合性能分布
- **分割结果图**: 显示预测结果与真实标签的对比
- **训练曲线**: 显示训练过程中的指标变化

### 2. 自动报告生成
- 生成详细的 Markdown 格式评估报告
- 包含平均性能统计
- 各域详细结果对比

### 3. 高质量输出
- 高分辨率图片 (300 DPI)
- 支持中文显示
- 自动保存到指定目录

## 使用方法

### 方法1: 自动集成（推荐）

运行 TTA 训练后，系统会自动生成可视化结果：

```bash
python TTA.py --Source_Dataset RIM_ONE_r3 --path_save_model ./models --dataset_root ./Dataset/Fundus
```

训练完成后，会在 `./visualization_results` 目录下生成：
- `RIM_ONE_r3_20250828_123456_comparison.png` - 指标对比图
- `RIM_ONE_r3_20250828_123456_heatmap.png` - 性能热力图
- `RIM_ONE_r3_20250828_123456_radar.png` - 雷达图
- `RIM_ONE_r3_20250828_123456_report.md` - 评估报告

### 方法2: 独立评估脚本

使用独立的评估脚本：

```bash
# 从JSON文件加载结果
python evaluate.py --results_file ./results/my_results.json --source_domain RIM_ONE_r3

# 手动输入结果
python evaluate.py --mode manual --source_domain RIM_ONE_r3
```

### 方法3: 编程接口

```python
from utils.visualization import EvaluationVisualizer, create_visualization_results

# 创建可视化器
visualizer = EvaluationVisualizer(save_dir='./my_results')

# 准备结果数据
results = {
    'REFUGE': {
        'Disc_Dice': 85.2,
        'Cup_Dice': 78.9,
        'Disc_ASSD': 12.3,
        'Cup_ASSD': 15.7
    },
    'ORIGA': {
        'Disc_Dice': 82.1,
        'Cup_Dice': 76.4,
        'Disc_ASSD': 14.2,
        'Cup_ASSD': 17.8
    }
}

# 生成可视化
create_visualization_results(results, 'RIM_ONE_r3', './my_results')
```

## 输出文件说明

### 1. 指标对比图 (`*_comparison.png`)
- 2x2 子图布局
- 显示 Disc Dice、Cup Dice、Disc ASSD、Cup ASSD
- 每个域用不同颜色表示
- 数值标签显示具体分数

### 2. 性能热力图 (`*_heatmap.png`)
- 行：目标域
- 列：评估指标
- 颜色：标准化性能（绿色=好，红色=差）
- 数值：原始分数

### 3. 雷达图 (`*_radar.png`)
- 极坐标系统
- 4个维度：Disc Dice、Cup Dice、Disc ASSD、Cup ASSD
- 每个域用不同颜色填充
- 便于比较综合性能

### 4. 评估报告 (`*_report.md`)
- Markdown 格式
- 包含基本信息、平均性能、详细结果
- 可直接用于论文或报告

## 自定义配置

### 修改保存目录
```python
visualizer = EvaluationVisualizer(save_dir='./custom_results')
```

### 自定义图表样式
```python
# 修改颜色方案
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# 修改字体大小
plt.rcParams['font.size'] = 12
```

### 添加新的可视化类型
```python
def plot_custom_chart(self, data, title="Custom Chart"):
    # 自定义绘图代码
    pass
```

## 依赖要求

确保安装以下依赖：
```bash
pip install matplotlib seaborn pandas numpy scikit-learn
```

## 故障排除

### 1. 中文显示问题
如果中文显示为方块，请安装中文字体：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```

### 2. 内存不足
对于大量数据，可以降低图片分辨率：
```python
plt.savefig(filename, dpi=150)  # 降低DPI
```

### 3. 保存失败
检查目录权限：
```bash
chmod 755 ./visualization_results
```

## 示例结果

运行后你会看到类似这样的输出：

```
Generating visualization results...
可视化结果已保存到: ./visualization_results
生成的文件:
- RIM_ONE_r3_20250828_123456_comparison.png (指标对比图)
- RIM_ONE_r3_20250828_123456_heatmap.png (性能热力图)
- RIM_ONE_r3_20250828_123456_radar.png (雷达图)
- RIM_ONE_r3_20250828_123456_report.md (评估报告)
Visualization completed successfully!
```

## 扩展功能

### 1. 添加新的评估指标
在 `utils/metrics.py` 中添加新指标，然后在可视化模块中更新相应的处理逻辑。

### 2. 支持更多图表类型
可以添加散点图、箱线图、小提琴图等更多可视化类型。

### 3. 交互式可视化
可以集成 Plotly 等库来创建交互式图表。

## 联系支持

如果遇到问题或有改进建议，请提交 Issue 或联系开发团队。
