import Head from 'next/head';
import Image from 'next/image';
import { useState, useEffect } from 'react';

export default function Home() {
  const [modelsData, setModelsData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('overview');
  
  useEffect(() => {
    fetch('/data/models_data.json')
      .then(res => res.json())
      .then(data => setModelsData(data))
      .catch(err => console.error('Failed to load model data:', err));
  }, []);

  if (!modelsData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading model data...</p>
        </div>
      </div>
    );
  }

  const champion = modelsData._champion;
  const models = Object.keys(modelsData).filter(key => key !== '_champion');
  const imageFiles = [
    'model_analysis_mlp.png',
    'model_analysis_mlp_20231109_20231209.png',
    'model_analysis_random_forest_20221231_20231209.png',
    'model_analysis_random_forest_20230910_20231209.png',
    'model_analysis_random_forest_20231109_20231209.png'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <Head>
        <title>Gate Token Predictive Modeling Dashboard</title>
        <meta name="description" content="ML Pipeline Analysis & Model Performance Dashboard" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                📊 Gate Token ML Dashboard
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Predictive Modeling Pipeline • Generated on {new Date().toLocaleDateString()}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center bg-yellow-100 px-3 py-2 rounded-lg">
                🏆
                <span className="text-sm font-medium text-yellow-800 ml-2">
                  Champion: {champion.replace('_', ' ').toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto py-4">
            <button
              onClick={() => setSelectedModel('overview')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${
                selectedModel === 'overview'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              📊 Overview
            </button>
            {models.map(model => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap flex items-center ${
                  selectedModel === model
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {model === champion && '🏆 '}
                {model.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {selectedModel === 'overview' ? (
          <OverviewSection modelsData={modelsData} champion={champion} imageFiles={imageFiles} />
        ) : (
          <ModelSection model={selectedModel} data={modelsData[selectedModel]} champion={champion} imageFiles={imageFiles} />
        )}
      </main>

      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-sm text-gray-300">
              Gate Token Predictive Modeling System • Built with Next.js & Deployed on Vercel
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Advanced ML Pipeline with Feature Engineering & Model Optimization
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function OverviewSection({ modelsData, champion, imageFiles }) {
  const models = Object.keys(modelsData).filter(key => key !== '_champion');
  
  return (
    <div className="space-y-8">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
          🏗️ System Architecture Overview
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">🏗️ Data Pipeline</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• 6-Stage Processing Pipeline</li>
              <li>• Excel → CSV → Features → Embeddings</li>
              <li>• 64-Dimensional Dense Encoding</li>
              <li>• Autoencoder Feature Learning</li>
            </ul>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="font-semibold text-green-900 mb-2">🤖 Model Training</h3>
            <ul className="text-sm text-green-800 space-y-1">
              <li>• {models.length} Different Algorithms</li>
              <li>• Optuna Hyperparameter Optimization</li>
              <li>• Champion Model Selection</li>
              <li>• Realtime Prediction System</li>
            </ul>
          </div>
          
          <div className="bg-purple-50 rounded-lg p-4">
            <h3 className="font-semibold text-purple-900 mb-2">📊 Analysis Tools</h3>
            <ul className="text-sm text-purple-800 space-y-1">
              <li>• Enhanced EDA Visualization</li>
              <li>• Time Range Selection</li>
              <li>• Interactive Model Explorer</li>
              <li>• System Transparency</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">📈 Model Performance Summary</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map(model => {
            const data = modelsData[model];
            const isChampion = model === champion;
            return (
              <div
                key={model}
                className={`rounded-lg p-4 border-2 ${
                  isChampion 
                    ? 'border-yellow-300 bg-yellow-50' 
                    : 'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">
                    {model.replace('_', ' ').toUpperCase()}
                  </h3>
                  {isChampion && '🏆'}
                </div>
                <div className="space-y-1 text-sm">
                  <p className="text-gray-600">
                    <span className="font-medium">Data Type:</span> {data.metadata?.data_type || 'Unknown'}
                  </p>
                  <p className="text-gray-600">
                    <span className="font-medium">Predictions:</span> {data.total_predictions?.toLocaleString() || 'N/A'}
                  </p>
                  {data.metadata?.best_params && (
                    <p className="text-gray-600">
                      <span className="font-medium">Optimized:</span> ✅
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">📊 Visualization Gallery</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {imageFiles.map(imageName => (
            <div key={imageName} className="relative">
              <h3 className="font-medium text-gray-900 mb-3 capitalize">
                {imageName.replace('model_analysis_', '').replace('.png', '').replace(/_/g, ' ')}
              </h3>
              <div className="bg-gray-100 rounded-lg overflow-hidden">
                <Image
                  src={`/images/${imageName}`}
                  alt={imageName}
                  width={800}
                  height={600}
                  className="w-full h-auto hover:scale-105 transition-transform cursor-pointer"
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ModelSection({ model, data, champion, imageFiles }) {
  const isChampion = model === champion;
  const metadata = data.metadata || {};
  
  return (
    <div className="space-y-8">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold text-gray-900 flex items-center">
            {isChampion && '🏆 '}
            {model.replace('_', ' ').toUpperCase()} Model
          </h2>
          <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
            isChampion ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-700'
          }`}>
            {isChampion ? '🏆 Champion Model' : '📈 Performance Model'}
          </div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900 text-lg">🏗️ Architecture</h3>
            <div className="space-y-2 text-sm">
              <p><span className="font-medium">Data Type:</span> {metadata.data_type || 'Unknown'}</p>
              <p><span className="font-medium">Model:</span> {metadata.model || model}</p>
              {metadata.data_type === 'dense' && (
                <div className="bg-blue-50 p-2 rounded text-blue-800">
                  ✅ Using 64-dimensional autoencoder embeddings
                </div>
              )}
              {metadata.data_type === 'sparse' && (
                <div className="bg-green-50 p-2 rounded text-green-800">
                  📊 Using direct feature scaling
                </div>
              )}
            </div>
          </div>

          {metadata.best_params && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900 text-lg">⚙️ Hyperparameters</h3>
              <div className="space-y-1 text-sm">
                {Object.entries(metadata.best_params).map(([key, value]) => (
                  <p key={key}>
                    <span className="font-medium">{key.replace('_', ' ')}:</span>{' '}
                    {Array.isArray(value) ? value.join(', ') : String(value)}
                  </p>
                ))}
              </div>
            </div>
          )}

          {data.total_predictions && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900 text-lg">📊 Statistics</h3>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Total Predictions:</span> {data.total_predictions.toLocaleString()}</p>
                <p><span className="font-medium">Training Date:</span> {metadata.training_timestamp?.split('T')[0] || 'Unknown'}</p>
              </div>
            </div>
          )}
        </div>

        {model === 'mlp' && metadata.best_params && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-lg mb-3">🧠 Neural Network Architecture</h3>
            <div className="text-sm text-gray-700">
              <p className="mb-2">
                <span className="font-medium">Network Structure:</span> 
                Input (64) → {metadata.best_params.hidden_layer_sizes?.[0] || 50} → {metadata.best_params.hidden_layer_sizes?.[1] || 50} → Output (1)
              </p>
              <p className="mb-2">
                <span className="font-medium">Regularization (α):</span> {metadata.best_params.alpha?.toFixed(6) || 'Unknown'}
              </p>
              <p className="mb-2">
                <span className="font-medium">Learning Rate:</span> {metadata.best_params.learning_rate_init?.toFixed(6) || 'Unknown'}
              </p>
              <p>
                <span className="font-medium">Max Iterations:</span> {metadata.best_params.max_iter || 'Unknown'}
              </p>
            </div>
          </div>
        )}
      </div>

      {data.sample_predictions && data.sample_predictions.length > 0 && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">📋 Sample Predictions</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">True Count</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Predicted Count</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Error</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.sample_predictions.slice(0, 10).map((pred, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {new Date(pred.timestamp).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {pred.true_count?.toFixed(0) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {pred.pred_count?.toFixed(0) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {pred.true_count && pred.pred_count ? (
                        <span className={`${
                          Math.abs(pred.true_count - pred.pred_count) < 10 
                            ? 'text-green-600' 
                            : Math.abs(pred.true_count - pred.pred_count) < 25 
                              ? 'text-yellow-600' 
                              : 'text-red-600'
                        }`}>
                          {(pred.true_count - pred.pred_count).toFixed(1)}
                        </span>
                      ) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4">📊 Model Visualizations</h3>
        <div className="grid gap-6">
          {imageFiles
            .filter(img => img.includes(model))
            .map(imageName => (
              <div key={imageName} className="relative">
                <h4 className="font-medium text-gray-700 mb-3">
                  {imageName.replace('model_analysis_', '').replace('.png', '').replace(/_/g, ' ').replace(model, '').trim() || 'Analysis Results'}
                </h4>
                <div className="bg-gray-100 rounded-lg overflow-hidden">
                  <Image
                    src={`/images/${imageName}`}
                    alt={imageName}
                    width={1200}
                    height={900}
                    className="w-full h-auto"
                  />
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
