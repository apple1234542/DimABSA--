# DimABSA Task 1：RMSE_VA = 1.2390 的实验流程与结果复盘

> 整理日期：2026-10-18。依据本次对话中提供的运行日志，以及保留的 `10_retrain_aonly.py`、`dimasr_task1_affine_postcal_fixed.py` 和 `06_affine_postcal.sh` 整理。本文记录的是重训练得到的 **1.2390 版本**，不是已删除历史约1.22版本的精确恢复，也不是任务三的实验结果。

## 1. 最终结果及核心方法

最终模型在日志中的名称为：

```text
TRAIN_OOF_BLEND_PLUS_DEV_AONLY_AFFINE_RETRAIN
```

它不是一轮单模型训练的结果，而是以下流程的组合：

1. 复用 RoBERTa、BGE 多种子回归模型与 Qwen 预测，构建 V、A 的基础融合结果。
2. 对 A 增加 Ridge 残差修正，得到 Ridge-residual 分支。
3. 用新种子重训只预测 A 的 RoBERTa 专家，并生成该专家的 TRAIN OOF 预测。
4. 在 TRAIN OOF 上，用 Ridge 代理分支和专家分支选择融合权重：专家0.12，Ridge0.88。
5. 将该权重用于实际 DEV/TEST 的专家与 Ridge-residual 预测融合。
6. 最后只对 A 做 DEV 仿射校准，V 保持此前融合结果不变，得到 TEST RMSE_VA = **1.2390**。

最终 TEST 指标如下：

| 指标 | 数值 | 方向 |
|---|---:|---|
| RMSE_V | 0.9956 | 越低越好 |
| RMSE_A | 0.7375 | 越低越好 |
| RMSE_VA | **1.2390** | 越低越好 |
| MAE_V | 0.7173 | 越低越好 |
| MAE_A | 0.5480 | 越低越好 |
| Pearson r_V | 0.8692 | 越高越好 |
| Pearson r_A | 0.5084 | 越高越好 |

## 2. 任务与指标如何定义

Task 1 对给定文本及 Aspect 预测两个连续值：Valence（V，情感效价）与 Arousal（A，情感唤醒度）。本项目将预测限制在1到9的范围内。

以下公式使用普通文本显示：sqrt表示平方根，Σ表示对所有样本求和，pred表示预测值，true表示真实标签；clip(x, 1, 9)表示将x限制在1到9。

单维 RMSE 为：

```text
RMSE_V = sqrt( Σ(V_pred[i] - V_true[i])² / N )
RMSE_A = sqrt( Σ(A_pred[i] - A_true[i])² / N )
```

本项目使用的联合指标为：

```text
RMSE_VA = sqrt( RMSE_V² + RMSE_A² )
```

因此1.2390不是两个RMSE的算术平均，也不是再除以2后的均方根。用日志中已经四舍五入的数字核算：

```text
sqrt(0.9956² + 0.7375²) ≈ 1.2390
```

MAE衡量平均绝对误差；Pearson相关系数反映预测与标签的线性相关程度。相关性较高不保证绝对误差较低，因为预测仍可能存在整体偏移或尺度偏差。

## 3. 数据与运行基础

服务器项目目录为：

```text
/root/autodl-tmp/.autodl/project
```

对应数据文件：

| 划分 | 文件 | 在这版流程中的作用 |
|---|---|---|
| TRAIN | eng_laptop_train_alltasks.jsonl | 基础模型训练、专家训练、OOF权重估计 |
| DEV | eng_laptop_dev_task1.jsonl | checkpoint选择、基础融合及残差相关选择、最终A校准 |
| TEST | eng_laptop_test_task1.jsonl | 报告该版本结果；已在历史实验中多次查看 |

这里列的是输入文件，不等于训练样本数。一个文本中可能有多个Aspect；过滤、展开及样本对齐需遵循原脚本。当前终端摘要没有列出各划分的实际有效样本数，本文不补造。

## 4. 基础模型：先获得互补的预测

### 4.1 RoBERTa 与 BGE

RoBERTa和BGE分别输出V、A预测。同一模型使用多个随机种子训练，再对预测取平均，以减少单次随机初始化、数据打乱等带来的波动。

复用的checkpoint包括：

```text
dimasr_roberta_output/best_model.pt
dimasr_roberta_output_seed123/best_model.pt
dimasr_roberta_output_seed2026/best_model.pt

dimasr_bge_aspect_attention_output/best_model.pt
dimasr_bge_output_seed123/best_model.pt
dimasr_bge_output_seed2026/best_model.pt
```

此前上传的终端摘要中，RoBERTa某次运行TEST RMSE_VA为1.3956，BGE某次运行为1.5258。这些是当时对应运行的结果，不能直接当成完整三种子均值的消融指标，也不能据此断言BGE在最终融合中无用。

### 4.2 Qwen

Qwen提供额外预测及置信度信息。旧流程会处理Qwen预测的校准，并在DEV上分别搜索V、A的三路融合权重。Qwen原始A预测曾明显较差，因此不能简单把它与编码器输出无条件等权平均。

**本次1.2390终端摘要未包含最终三路权重、Qwen前置校准参数及完整中间结果。** 这些应从本次运行完整日志或metrics文件读取，不能拿其他版本的数值填入。

### 4.3 分维融合

V和A不必使用相同权重。基础融合可以写作：

```text
V_base = w_RV × V_RoBERTa + w_BV × V_BGE + w_QV × V_Qwen
```

```text
A_base = w_RA × A_RoBERTa + w_BA × A_BGE + w_QA × A_Qwen
```

上述公式解释结构；这里的具体权重并非0.12/0.88。**0.12/0.88属于后面的专家与Ridge两分支融合。**

## 5. A维 Ridge 残差修正

原脚本在基础融合之后构造低维特征，训练Ridge预测A的残差：

```text
A_residual[i] = A_true[i] - A_base[i]
```

残差特征包括基础融合A、RoBERTa A、BGE A、校准后Qwen A、Qwen置信度、融合V，以及模型间差值等，总计10维。

脚本在DEV交叉验证中选择Ridge正则系数、修正强度和修正上限，随后在完整DEV上重拟合残差模型，用于DEV与TEST预测。概念上为：

```text
A_ridge = clip(A_base + Correction(A_residual_pred), 1, 9)
```

这里的Correction受修正强度与上限控制，不能理解为无约束直接加残差。完整DEV重拟合后的DEV指标属于拟合后结果，不是独立验证结果。

最终日志中该Ridge分支对应的TEST联合RMSE为 **1.2677**。当前摘要未给出它单独的V、A完整指标和具体残差超参数，本文不推算为原始精确结果。

## 6. 用新种子重训 Arousal 专家

此次重训练脚本使用的新随机种子为：

```text
17, 2027, 3407
```

训练设置中可以从包装脚本确认：

| 项目 | 配置 |
|---|---|
| 专家基础模型 | 本地RoBERTa-base |
| 预测目标 | Arousal，即A维 |
| 专家训练epochs参数 | 8 |
| 专家OOF折数 | 3 |
| 专家OOF每折epochs参数 | 4 |
| 专家OOF种子 | 17、2027、3407 |
| 混合精度 | FP16 |

这里的8是传入训练器的epochs参数，不表示最终checkpoint一定来自第8轮。具体最佳epoch需看专家训练日志。

专家训练结合A的回归误差、相关性损失和序数分类损失：

```text
Loss = MSE + λ_corr × (1 - Pearson_r) + λ_ord × ordinal_loss
```

这使模型同时学习数值接近、相对变化及强度区间。各损失系数及其实际生效值应以本次训练参数或checkpoint配置为准。

## 7. TRAIN OOF 如何选出0.12/0.88

### 7.1 OOF的含义

将TRAIN分为3折。每次用其中2折训练，在未参与该次训练的1折上预测，合并得到每条训练样本的折外预测。多种子结果再按原脚本集成。

此次包装脚本把专家OOF及代理Ridge交叉拟合的分组键，从ID改为规范化文本：转小写并合并连续空白。同规范化文本进入同组，减少重复文本跨折的风险。

这项修改**不等于既有RoBERTa/BGE缓存也被重新按该规则生成**。基础OOF缓存被复用，不能声称整条链路已经完成严格嵌套、统一分组的交叉验证。

### 7.2 日志中的OOF结果

```text
TRAIN OOF Specialist RMSE_A=0.7568
TRAIN OOF proxy Ridge RMSE_A=0.6969
TRAIN OOF selected alpha=0.12 specialist + 0.88 ridge | RMSE_A=0.6959
```

由此得到：

```text
A_blend = 0.12 × A_specialist + 0.88 × A_ridge
```

专家单独的OOF误差高于Ridge代理，但少量加入后，融合误差从0.6969降至0.6959，说明在这组训练侧预测中存在一定误差互补。改善幅度很小，不应描述为显著提升。

### 7.3 代理Ridge与实际Ridge分支不是同一个对象

权重搜索时使用的是基于TRAIN基础OOF特征交叉拟合得到的 **proxy Ridge**；应用到DEV/TEST时，使用的是前述DEV残差修正后的 **Ridge-residual分支**。

所以这是一种“用训练侧代理预测估计融合权重，再迁移到实际分支”的方法。TRAIN OOF的0.6959不能当成最终整个系统的无偏泛化误差，也不能保证迁移后的权重一定最优。

## 8. 融合后：TEST为1.2642

固定0.12/0.88后，运行日志为：

| 划分 | RMSE_V | RMSE_A | RMSE_VA | MAE_V | MAE_A | r_V | r_A |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLEND DEV | 0.6023 | 0.6815 | 0.9095 | 0.4663 | 0.5268 | 0.9345 | 0.6090 |
| BLEND TEST | 0.9956 | 0.7791 | 1.2642 | 0.7173 | 0.5899 | 0.8692 | 0.5084 |

此时最终分数还不是1.2390。决定最后一步改善的是A维的仿射校准。

## 9. 只校准A：从1.2642得到1.2390

### 9.1 校准公式

在DEV上拟合线性回归：

```text
A_true[i] ≈ a × A_blend[i] + b
```

本次日志给出的参数为：

```text
a = 0.916995
b = 0.623076
```

因此最终预测为：

```text
A_final = clip(0.916995 × A_blend + 0.623076, 1, 9)
```

V维保持不变：

```text
V_final = V_blend
```

这里“V不变”特指**最后仿射校准这一步**不修改V，不表示前面的V没有进行模型融合或其他处理。

### 9.2 如何决定采用校准

脚本比较同一个DEV上的校准前、后A维RMSE：

```text
POSTCAL A | base_DEV_RMSE=0.6815 affine_DEV_RMSE=0.6786
selected=affine a=0.916995 b=0.623076
```

因为0.6786低于0.6815，采用A仿射变换。参数只在DEV拟合，随后固定并应用到TEST。

**这一选择用的是校准拟合后的DEV误差，不是独立校准验证分数，也不是后续其他实验使用的group-CV校准方案。**

### 9.3 校准前后对照

| 划分/阶段 | RMSE_V | RMSE_A | RMSE_VA | MAE_A | r_A |
|---|---:|---:|---:|---:|---:|
| DEV：融合后 | 0.6023 | 0.6815 | 0.9095 | 0.5268 | 0.6090 |
| DEV：仅A校准后 | 0.6023 | 0.6786 | 0.9073 | 0.5215 | 0.6090 |
| TEST：融合后 | 0.9956 | 0.7791 | 1.2642 | 0.5899 | 0.5084 |
| TEST：仅A校准后 | 0.9956 | **0.7375** | **1.2390** | **0.5480** | 0.5084 |

TEST上，A的RMSE下降0.0416，联合RMSE下降0.0252，约为融合后联合误差的1.99%。这些差值按日志四舍五入后的数字计算。

斜率小于1意味着压缩预测尺度，同时加截距补偿整体位置。它主要修正数值尺度和偏移，不会增加新的文本理解信息。在没有裁剪影响且斜率为正时，仿射变换保持Pearson相关系数；本次日志中r_A也保持0.5084，但不能据此断言所有样本都未被裁剪。

## 10. 同次运行中各分支的TEST对照

```text
TEST comparison (diagnostic only) |
Ridge=1.2677 | Specialist-only=1.2680 | Blend=1.2642 | Selected=1.2390
```

| 配置 | TEST RMSE_VA |
|---|---:|
| 既有V + Ridge-residual A | 1.2677 |
| 既有V + 专家A | 1.2680 |
| 既有V + 0.12专家A + 0.88 Ridge A | 1.2642 |
| 上述融合 + 仅A仿射校准 | **1.2390** |

“Specialist-only”是A分支替换实验，仍搭配既有V；不是该专家独立预测了V和A。

这张表来自同次日志，适合解释组合过程。它不是控制所有训练因素、跨多种子重复的严格消融研究；也不能因为看到了TEST对照，就据此重新用TEST挑权重或校准参数。

## 11. 与旧1.2500版本的区别

旧1.2500版本的TEST摘要为V=1.0093、A=0.7375；本次1.2390为V=0.9956、A=0.7375。按四位小数看，两版A相同，本版V更低。

这说明需要区分两个比较：

- **本次融合前后校准比较：** 1.2642→1.2390，改善来自最后一步A校准，V不变。
- **与旧1.2500版本比较：** 本版在最后一步保留较好的V结果，避免重新改动V；与此同时也重训了专家，不能视为所有其他条件完全相同的单因素对照。

因此不能简单写成“重新训练A专家使分数从1.2500降至1.2390”，也不能把旧版校准参数或权重与本次结果拼接。

## 12. 复现入口与文件位置

本次包装脚本为 `10_retrain_aonly.py`。其工作目录为：

```text
/root/autodl-tmp/.autodl/project/dimasr_aonly_retrain_v5/
├── config.json
├── aonly_pipeline.py
├── run_aonly.sh
├── specialist/
├── specialist.log
├── aonly_pipeline.log
└── final/
    ├── test_predictions_arousal_specialist_oof_blend_final.jsonl
    └── metrics_arousal_specialist_oof_blend_final.json
```

上面是脚本定义的输出结构，不代表本文重新检查了服务器上的文件是否仍然存在。`final`下还可能有缓存和其他中间文件。

包装脚本会：

1. 检查基础checkpoint、数据、OOF缓存及模型配置是否存在。
2. 将最后校准循环改为只处理A，并修改模型名称。
3. 将专家和proxy Ridge的OOF分组替换成规范化文本。
4. 用17/2027/3407训练专家，并记录日志与完成标记。
5. 调用修改后的第六阶段流程，复用原基础模型与缓存。
6. 保存最终预测、指标和校准信息。

所以“重训练”在这里主要指**重训A专家及相关OOF流程**，不是把所有RoBERTa、BGE、Qwen从头训练一遍。

真正复现需保留数据、基础模型checkpoint、OOF数组、Qwen缓存、专家checkpoint、代码版本、运行参数、完整日志及校准参数。只有1.2390这一行终端输出不足以精确复现。

## 13. 可以怎样写进论文

> 在多模型回归预测的基础上，本文对情感效价与唤醒度采用分维处理。唤醒度分支结合Ridge残差修正和RoBERTa专门回归模型，利用训练集折外代理预测估计融合权重，得到0.12的专家权重和0.88的Ridge分支权重。随后在开发集上拟合唤醒度仿射校准函数，并保持效价预测不变。该版本在已使用的测试划分上取得RMSE_V=0.9956、RMSE_A=0.7375、RMSE_VA=1.2390。由于开发集参与融合、残差修正及校准，且测试划分已在多轮历史实验中查看，该结果作为当前实验划分上的开发结果报告，不作为全新独立测试上的无偏泛化估计。

不宜声称：所有超参数都只在TRAIN确定；DEV为完全独立验证；本方法稳定达到1.2390；已经恢复删除的1.22版本；或OOF分数代表严格嵌套交叉验证结果。

## 14. 当前记录的边界

已经确认的关键数据：新专家种子、OOF折数和epochs参数、0.12/0.88融合、A校准参数、校准前后DEV/TEST指标及最终模型名称。

仍需从本次完整归档补齐的内容：实际样本数、各专家最佳epoch、基础三路融合权重、前置Qwen校准配置、Ridge残差最优超参数、运行环境版本，以及最终预测文件的完整哈希。

**1.2390的直接计算链是：基础多模型预测 → A残差修正 → TRAIN OOF估计的专家融合 → DEV仅A仿射校准。最关键的末步是将A从0.7791降至0.7375，同时把V保持在0.9956。**
