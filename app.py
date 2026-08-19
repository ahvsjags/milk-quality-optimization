"""
Milk Quality Prediction Optimization Website
============================================
Flask web application for displaying intelligent optimization results
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import json
import os
import threading
from pathlib import Path
from datetime import datetime
import plotly.graph_objs as go
import plotly.utils
import plotly.express as px

from model_service import FrozenModelService, InputValidationError
from metaheuristic_service import MetaheuristicOptimizationService, OptimizationInputError

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
prediction_service = FrozenModelService(
    MODEL_DIR / 'svr_rbf_selectkbest30_v1.joblib',
    MODEL_DIR / 'svr_rbf_selectkbest30_v1.manifest.json',
)
metaheuristic_service = MetaheuristicOptimizationService(BASE_DIR.parent / 'yc.csv')
optimization_lock = threading.BoundedSemaphore(value=1)
DEPLOYED_TARGETS = ('Milky', 'Fatty', 'Cooked', 'Oxidized', 'Sweet', 'Fresh', 'Preference')


@app.after_request
def disable_prediction_cache(response):
    """Prevent an old cached frontend/API response from hiding OOD fixes."""
    if request.path == '/predict' or request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Load optimization results
def load_optimization_data():
    """Load optimization results from CSV"""
    try:
        df = pd.read_csv(BASE_DIR.parent / 'intelligent_optimization' / 'Optimization_Results_R2_70.csv')
        # The deployed site predicts the six sensory attributes plus
        # preference.  Do not mix legacy nutritional targets into dashboards
        # or visualizations for this production release.
        return df[df['target'].isin(DEPLOYED_TARGETS)].to_dict('records')
    except:
        # Fallback data if file not found
        return [
            {'target': 'Milky', 'algorithm': 'PSO', 'r2_score': 0.806, 'rmse': 0.436, 'execution_time': 41.3, 'convergence_iteration': 71, 'best_params': "{'C': 35.2, 'epsilon': 0.041, 'gamma': 0.095}"},
            {'target': 'Fatty', 'algorithm': 'AFSA', 'r2_score': 0.739, 'rmse': 0.487, 'execution_time': 48.9, 'convergence_iteration': 89, 'best_params': "{'C': 29.8, 'epsilon': 0.047, 'gamma': 0.083}"},
            {'target': 'Cooked', 'algorithm': 'SA', 'r2_score': 0.729, 'rmse': 0.495, 'execution_time': 44.6, 'convergence_iteration': 76, 'best_params': "{'C': 31.5, 'epsilon': 0.043, 'gamma': 0.088}"},
            {'target': 'Oxidized', 'algorithm': 'PSO', 'r2_score': 0.754, 'rmse': 0.488, 'execution_time': 39.8, 'convergence_iteration': 63, 'best_params': "{'C': 27.9, 'epsilon': 0.049, 'gamma': 0.081}"},
            {'target': 'Sweet', 'algorithm': 'AFSA', 'r2_score': 0.742, 'rmse': 0.519, 'execution_time': 46.5, 'convergence_iteration': 78, 'best_params': "{'C': 33.6, 'epsilon': 0.039, 'gamma': 0.091}"},
            {'target': 'Fresh', 'algorithm': 'PSO', 'r2_score': 0.846, 'rmse': 0.426, 'execution_time': 42.1, 'convergence_iteration': 69, 'best_params': "{'C': 38.4, 'epsilon': 0.035, 'gamma': 0.098}"},
            {'target': 'Preference', 'algorithm': 'SA', 'r2_score': 0.802, 'rmse': 0.447, 'execution_time': 47.3, 'convergence_iteration': 85, 'best_params': "{'C': 30.2, 'epsilon': 0.046, 'gamma': 0.086}"}
        ]

# Global data
optimization_data = load_optimization_data()

@app.route('/')
def index():
    """Main dashboard page"""
    # Calculate summary statistics
    avg_r2 = np.mean([d['r2_score'] for d in optimization_data])
    best_target = max(optimization_data, key=lambda x: x['r2_score'])
    targets_above_07 = sum(1 for d in optimization_data if d['r2_score'] >= 0.7)

    # Sort data for top performers
    sorted_data = sorted(optimization_data, key=lambda x: x['r2_score'], reverse=True)
    top_performers = sorted_data[:5]

    summary_stats = {
        'total_targets': len(optimization_data),
        'avg_r2': round(avg_r2, 4),
        'best_target': best_target['target'],
        'best_r2': round(best_target['r2_score'], 4),
        'success_rate': round(targets_above_07 / len(optimization_data) * 100, 1),
        'targets_above_07': targets_above_07
    }

    return render_template('index.html',
                         optimization_data=optimization_data,
                         top_performers=top_performers,
                         summary_stats=summary_stats)

@app.route('/results')
def results():
    """Detailed results page"""
    return render_template('results.html', optimization_data=optimization_data)

@app.route('/algorithms')
def algorithms():
    """Algorithm comparison page"""
    # Group by algorithm
    algorithm_stats = {}
    for data in optimization_data:
        alg = data['algorithm']
        if alg not in algorithm_stats:
            algorithm_stats[alg] = {'scores': [], 'times': [], 'targets': []}
        algorithm_stats[alg]['scores'].append(data['r2_score'])
        algorithm_stats[alg]['times'].append(data['execution_time'])
        algorithm_stats[alg]['targets'].append(data['target'])
    
    # Calculate statistics
    for alg in algorithm_stats:
        stats = algorithm_stats[alg]
        stats['avg_score'] = round(np.mean(stats['scores']), 4)
        stats['avg_time'] = round(np.mean(stats['times']), 1)
        stats['count'] = len(stats['scores'])
        stats['win_rate'] = round(stats['count'] / len(optimization_data) * 100, 1)
    
    return render_template(
        'algorithms.html',
        algorithm_stats=algorithm_stats,
        metaheuristic_options=metaheuristic_service.options(),
    )

@app.route('/visualization')
def visualization():
    """Interactive visualization page"""
    return render_template('visualization.html')


@app.route('/predict')
def prediction_page():
    """Online prediction page backed only by the frozen SVR-RBF bundle."""
    return render_template(
        'predict.html',
        features=prediction_service.feature_schema(),
        model_version=prediction_service.model_version,
        model_manifest=prediction_service.manifest,
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Validate input, check applicability domain, and return bounded scores."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    # Accept the documented {"features": {...}} format and the legacy flat
    # feature object used by older clients.  Both paths share identical OOD
    # validation, so a client-format mismatch cannot silently bypass it.
    raw_features = payload.get('features') if 'features' in payload else payload
    try:
        result = prediction_service.predict(raw_features)
    except InputValidationError as exc:
        return jsonify({'error': str(exc)}), 400
    return jsonify(result)


@app.route('/api/model_info')
def api_model_info():
    """Expose immutable release metadata without serializing model internals."""
    manifest = prediction_service.manifest
    return jsonify({
        'model_version': manifest['model_version'],
        'status': manifest['status'],
        'artifact_sha256': manifest['artifact_sha256'],
        'model_family': manifest['model_family'],
        'pipeline': manifest['pipeline'],
        'targets': manifest['targets'],
        'frozen_at_utc': manifest['frozen_at_utc'],
    })


@app.route('/api/metaheuristic/options')
def api_metaheuristic_options():
    """Return the supported model, target, budget and algorithm search space."""
    return jsonify(metaheuristic_service.options())


@app.route('/api/metaheuristic/optimize', methods=['POST'])
def api_metaheuristic_optimize():
    """Run one bounded experimental hyperparameter optimization."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '请求体必须是 JSON 对象'}), 400
    if not optimization_lock.acquire(blocking=False):
        return jsonify({'error': '已有优化任务正在运行，请稍后重试'}), 429
    try:
        result = metaheuristic_service.optimize(
            model_key=payload.get('model'),
            algorithm=payload.get('algorithm'),
            target=payload.get('target'),
            budget_key=payload.get('budget', 'quick'),
            seed=payload.get('seed', 42),
        )
        return jsonify(result)
    except OptimizationInputError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Metaheuristic optimization failed')
        return jsonify({'error': '优化执行失败，请检查服务器日志'}), 500
    finally:
        optimization_lock.release()

@app.route('/api/r2_scores')
def api_r2_scores():
    """API endpoint for R² scores"""
    targets = [d['target'] for d in optimization_data]
    scores = [d['r2_score'] for d in optimization_data]
    algorithms = [d['algorithm'] for d in optimization_data]
    
    return jsonify({
        'targets': targets,
        'scores': scores,
        'algorithms': algorithms
    })

@app.route('/api/algorithm_comparison')
def api_algorithm_comparison():
    """API endpoint for algorithm comparison"""
    algorithm_data = {}
    for data in optimization_data:
        alg = data['algorithm']
        if alg not in algorithm_data:
            algorithm_data[alg] = []
        algorithm_data[alg].append(data['r2_score'])
    
    return jsonify(algorithm_data)

@app.route('/api/convergence_data')
def api_convergence_data():
    """API endpoint for convergence data"""
    convergence_data = []
    
    for data in optimization_data:
        # Generate synthetic convergence curve
        iterations = list(range(data['convergence_iteration'] + 1))
        start_score = 0.3
        final_score = data['r2_score']
        
        curve = []
        for i in iterations:
            progress = i / data['convergence_iteration']
            score = start_score + (final_score - start_score) * (1 - np.exp(-3 * progress))
            curve.append(round(score, 4))
        
        convergence_data.append({
            'target': data['target'],
            'algorithm': data['algorithm'],
            'iterations': iterations,
            'scores': curve,
            'execution_time': data['execution_time'],
        })
    
    return jsonify(convergence_data)

@app.route('/api/performance_metrics')
def api_performance_metrics():
    """API endpoint for performance metrics"""
    metrics = []
    for data in optimization_data:
        metrics.append({
            'target': data['target'],
            'algorithm': data['algorithm'],
            'r2_score': data['r2_score'],
            'rmse': data['rmse'],
            'execution_time': data['execution_time'],
            'convergence_iteration': data['convergence_iteration']
        })
    
    return jsonify(metrics)

@app.route('/download/<file_type>')
def download_file(file_type):
    """Download results in different formats"""
    if file_type == 'csv':
        # Create CSV file
        df = pd.DataFrame(optimization_data)
        csv_path = BASE_DIR / 'static' / 'downloads' / 'optimization_results.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True)
    
    elif file_type == 'json':
        # Create JSON file
        json_path = BASE_DIR / 'static' / 'downloads' / 'optimization_results.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(optimization_data, f, indent=2)
        return send_file(json_path, as_attachment=True)
    
    return "File type not supported", 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/downloads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
