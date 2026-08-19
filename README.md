# MilkSense · Milk Quality Prediction

[中文](#中文) · [English](#english)

## 中文

### 在线演示

GitHub Pages 静态演示：`https://ahvsjags.github.io/milk-quality-optimization/`

该演示可在浏览器中直接运行已冻结的预测模型；不需要账户、后端服务或上传样品数据。页面右上角可切换中文 / English。

### 已冻结的生产模型

- 部署物：`models/svr_rbf_selectkbest30_v1.joblib`
- 清单：`models/svr_rbf_selectkbest30_v1.manifest.json`
- 管线：`StandardScaler → SelectKBest(f_regression, k=30) → SVR(kernel='rbf')`
- 六项感官属性：0–5；喜好度：0–10
- 完整性：启动 Flask 服务时检查 SHA-256；浏览器演示显示同一部署物的摘要。

六项感官模型与训练期优化策略固定为：奶香味 / 氧化味 / 青草味使用 PSO，蒸煮味使用 SA，脂香味 / 甜味使用 AFSA。页面中的模型不会自动重训或更新。

### 适用域保护

输入会经过有限数值、非负浓度、训练最小 / 最大范围以及 LedoitWolf Mahalanobis 距离检查。超出域时，系统不会崩溃；推理仅使用截断后的边界值，输出按合法量表限制，并显示：

> 该数据已超出本模型的适用范围（Out of Domain），预测结果仅供参考

公共网页包含“己醛 ×1000”抗压测试和“非牛奶 / Non-milk”样品选项，可验证该流程。

### GitHub Pages 的浏览器推理

`docs/` 是可直接发布到 GitHub Pages 的独立网站。`export_browser_model.py` 将不可变 joblib 模型**仅导出推理参数**（缩放、30 个特征索引、SVR-RBF 支持向量、量程与适用域守卫）至 `docs/assets/frozen-model-v1.json`，不重新训练模型。

```powershell
cd D:\Downloads\牛奶建模\milk_optimization_website
python export_browser_model.py
python -m http.server 8000 --directory docs
```

然后浏览 `http://127.0.0.1:8000`。

### Flask 完整工作台（本地 / 服务器部署）

Flask 版本保留原仪表盘、结果表、交互式可视化、OOD 预测 API 和启发式算法实验工作台：

```powershell
pip install -r requirements.txt
python run.py
```

访问 `http://127.0.0.1:5000`。`/algorithms` 提供 SVR-RBF、Extra Trees、Random Forest 和 Gradient Boosting 的有界实验，并支持 PSO、SA、AFSA、GA、DE、GWO、WOA、ACO 八种启发式策略；实验结果不会覆盖冻结模型。

### 验证

```powershell
python -m pytest -q
npx playwright test tests/public-pages.spec.cjs --reporter=line
```

## English

### Live demo

GitHub Pages demo: `https://ahvsjags.github.io/milk-quality-optimization/`

The public demo runs the immutable prediction model in the visitor's browser. No account, server-side prediction service, or sample upload is required. Use the top-right language control to switch between Chinese and English.

### Frozen deployment model

- Artifact: `models/svr_rbf_selectkbest30_v1.joblib`
- Pipeline: `StandardScaler → SelectKBest(f_regression, k=30) → SVR(kernel='rbf')`
- Six sensory attributes: 0–5; preference: 0–10
- Integrity: SHA-256 verification is performed by Flask, and the same artifact digest is shown by the browser demo.

The public page performs range and LedoitWolf Mahalanobis applicability-domain checks. Out-of-domain inputs remain available for a guarded, bounded prediction and show a clear warning instead of crashing.

`docs/` is the standalone GitHub Pages site. It uses inference-only parameters exported from the immutable joblib bundle by `export_browser_model.py`; it never retrains the model.
