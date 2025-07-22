#!/usr/bin/env python3
"""
Vercel EDA Deployment Script
Creates a Next.js web application for deploying model analysis results to Vercel
"""

import os
import json
import base64
import pandas as pd
import yaml
from pathlib import Path
import shutil
from datetime import datetime

class VercelEDADeployer:
    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self.vercel_dir = self.project_dir / "vercel-eda-app"
        
    def create_vercel_app_structure(self):
        """Create Next.js app structure for Vercel"""
        print("🏗️ Creating Vercel app structure...")
        
        # Clean and create directories
        if self.vercel_dir.exists():
            shutil.rmtree(self.vercel_dir)
        
        # Create directory structure
        directories = [
            "pages",
            "components",
            "public/images",
            "public/data",
            "styles"
        ]
        
        for dir_path in directories:
            (self.vercel_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created")
    
    def generate_package_json(self):
        """Generate package.json for Next.js app"""
        package_json = {
            "name": "gate-token-eda-dashboard",
            "version": "1.0.0",
            "description": "Gate Token Predictive Modeling EDA Dashboard",
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint"
            },
            "dependencies": {
                "next": "13.5.6",
                "react": "18.2.0",
                "react-dom": "18.2.0",
                "recharts": "^2.8.0",
                "lucide-react": "^0.287.0",
                "@headlessui/react": "^1.7.17"
            },
            "devDependencies": {
                "@types/node": "20.8.7",
                "@types/react": "18.2.31",
                "@types/react-dom": "18.2.14",
                "autoprefixer": "10.4.16",
                "postcss": "8.4.31",
                "tailwindcss": "3.3.5",
                "typescript": "5.2.2"
            }
        }
        
        with open(self.vercel_dir / "package.json", "w") as f:
            json.dump(package_json, f, indent=2)
        
        print("✅ package.json created")
    
    def copy_visualization_images(self):
        """Copy all visualization images to public directory"""
        print("📊 Copying visualization images...")
        
        image_files = []
        for img_file in self.project_dir.glob("model_analysis_*.png"):
            dest_path = self.vercel_dir / "public" / "images" / img_file.name
            shutil.copy2(img_file, dest_path)
            image_files.append(img_file.name)
            print(f"  📄 Copied {img_file.name}")
        
        return image_files
    
    def collect_model_data(self):
        """Collect all model data for dashboard"""
        print("🤖 Collecting model data...")
        
        models_data = {}
        predictions_dir = self.project_dir / "data" / "predictions"
        
        for model_dir in predictions_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                models_data[model_name] = {}
                
                # Get latest files
                test_files = list(model_dir.glob("*_test_preds_*.csv"))
                metadata_files = list(model_dir.glob("*_metadata_*.yaml"))
                
                if test_files:
                    latest_test = max(test_files, key=lambda x: x.stat().st_mtime)
                    # Read sample predictions data
                    df = pd.read_csv(latest_test)
                    models_data[model_name]["sample_predictions"] = df.head(10).to_dict('records')
                    models_data[model_name]["total_predictions"] = len(df)
                
                if metadata_files:
                    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_metadata, 'r') as f:
                            content = f.read()
                            # Filter out problematic NumPy serializations
                            lines = content.split('\n')
                            filtered_lines = []
                            skip_lines = False
                            
                            for line in lines:
                                if 'test_metrics:' in line or 'train_metrics:' in line:
                                    skip_lines = True
                                    continue
                                if skip_lines and (line.startswith('  ') or line.strip() == ''):
                                    continue
                                if skip_lines and not line.startswith(' '):
                                    skip_lines = False
                                
                                if not skip_lines:
                                    filtered_lines.append(line)
                            
                            filtered_content = '\n'.join(filtered_lines)
                            metadata = yaml.safe_load(filtered_content)
                            models_data[model_name]["metadata"] = metadata
                    except Exception as e:
                        print(f"  ⚠️ Could not load metadata for {model_name}: {e}")
                        models_data[model_name]["metadata"] = {"model": model_name}
        
        # Get champion model
        champion_file = self.project_dir / "models" / "champion.txt"
        champion_model = "unknown"
        if champion_file.exists():
            champion_model = champion_file.read_text().strip()
        
        models_data["_champion"] = champion_model
        
        # Save to JSON
        with open(self.vercel_dir / "public" / "data" / "models_data.json", "w") as f:
            json.dump(models_data, f, indent=2)
        
        print(f"✅ Collected data for {len(models_data)-1} models")
        return models_data
    
    def generate_next_config(self):
        """Generate next.config.js"""
        config_content = '''/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
  output: 'export'
}

module.exports = nextConfig
'''
        with open(self.vercel_dir / "next.config.js", "w") as f:
            f.write(config_content)
        
        print("✅ next.config.js created")
    
    def generate_tailwind_config(self):
        """Generate Tailwind CSS configuration"""
        tailwind_config = '''/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
'''
        with open(self.vercel_dir / "tailwind.config.js", "w") as f:
            f.write(tailwind_config)
        
        postcss_config = '''module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
'''
        with open(self.vercel_dir / "postcss.config.js", "w") as f:
            f.write(postcss_config)
        
        print("✅ Tailwind configuration created")
    
    def generate_main_page(self, image_files, models_data):
        """Generate main dashboard page"""
        page_content = f'''import Head from 'next/head';
import Image from 'next/image';
import {{ useState, useEffect }} from 'react';
import {{ ChevronDownIcon, TrophyIcon, CpuChipIcon, ChartBarIcon }} from 'lucide-react';

export default function Home() {{
  const [modelsData, setModelsData] = useState(null);
  const [selectedModel, setSelectedModel] = useState('overview');
  
  useEffect(() => {{
    fetch('/data/models_data.json')
      .then(res => res.json())
      .then(data => setModelsData(data));
  }}, []);

  if (!modelsData) {{
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading model data...</p>
        </div>
      </div>
    );
  }}

  const champion = modelsData._champion;
  const models = Object.keys(modelsData).filter(key => key !== '_champion');

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
                <ChartBarIcon className="h-8 w-8 text-blue-600 mr-3" />
                Gate Token ML Dashboard
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Predictive Modeling Pipeline • Generated on {{new Date().toLocaleDateString()}}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center bg-yellow-100 px-3 py-2 rounded-lg">
                <TrophyIcon className="h-5 w-5 text-yellow-600 mr-2" />
                <span className="text-sm font-medium text-yellow-800">
                  Champion: {{champion.replace('_', ' ').toUpperCase()}}
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
              onClick={{() => setSelectedModel('overview')}}
              className={{`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap ${{
                selectedModel === 'overview'
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }}`}}
            >
              📊 Overview
            </button>
            {{models.map(model => (
              <button
                key={{model}}
                onClick={{() => setSelectedModel(model)}}
                className={{`px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap flex items-center ${{
                  selectedModel === model
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }}`}}
              >
                {{model === champion && <TrophyIcon className="h-4 w-4 mr-1 text-yellow-500" />}}
                {{model.replace('_', ' ').toUpperCase()}}
              </button>
            ))}}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {{selectedModel === 'overview' ? (
          <OverviewSection modelsData={{modelsData}} champion={{champion}} />
        ) : (
          <ModelSection model={{selectedModel}} data={{modelsData[selectedModel]}} champion={{champion}} />
        )}}
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
}}

function OverviewSection({{ modelsData, champion }}) {{
  const models = Object.keys(modelsData).filter(key => key !== '_champion');
  
  return (
    <div className="space-y-8">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
          <CpuChipIcon className="h-6 w-6 text-blue-600 mr-2" />
          System Architecture Overview
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
              <li>• {{models.length}} Different Algorithms</li>
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
          {{models.map(model => {{
            const data = modelsData[model];
            const isChampion = model === champion;
            return (
              <div
                key={{model}}
                className={{`rounded-lg p-4 border-2 ${{
                  isChampion 
                    ? 'border-yellow-300 bg-yellow-50' 
                    : 'border-gray-200 bg-gray-50'
                }}`}}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">
                    {{model.replace('_', ' ').toUpperCase()}}
                  </h3>
                  {{isChampion && (
                    <TrophyIcon className="h-5 w-5 text-yellow-500" />
                  )}}
                </div>
                <div className="space-y-1 text-sm">
                  <p className="text-gray-600">
                    <span className="font-medium">Data Type:</span> {{data.metadata?.data_type || 'Unknown'}}
                  </p>
                  <p className="text-gray-600">
                    <span className="font-medium">Predictions:</span> {{data.total_predictions?.toLocaleString() || 'N/A'}}
                  </p>
                  {{data.metadata?.best_params && (
                    <p className="text-gray-600">
                      <span className="font-medium">Optimized:</span> ✅
                    </p>
                  )}}
                </div>
              </div>
            );
          }})}}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">📊 Visualization Gallery</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {{{JSON.stringify(image_files)}.map(imageName => (
            <div key={{imageName}} className="relative">
              <h3 className="font-medium text-gray-900 mb-3 capitalize">
                {{imageName.replace('model_analysis_', '').replace('.png', '').replace(/_/g, ' ')}}
              </h3>
              <div className="bg-gray-100 rounded-lg overflow-hidden">
                <Image
                  src={{`/images/${{imageName}}`}}
                  alt={{imageName}}
                  width={{800}}
                  height={{600}}
                  className="w-full h-auto hover:scale-105 transition-transform cursor-pointer"
                />
              </div>
            </div>
          ))}}
        </div>
      </div>
    </div>
  );
}}

function ModelSection({{ model, data, champion }}) {{
  const isChampion = model === champion;
  const metadata = data.metadata || {{}};
  
  return (
    <div className="space-y-8">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-3xl font-bold text-gray-900 flex items-center">
            {{isChampion && <TrophyIcon className="h-8 w-8 text-yellow-500 mr-3" />}}
            {{model.replace('_', ' ').toUpperCase()}} Model
          </h2>
          <div className={{`px-4 py-2 rounded-lg text-sm font-medium ${{
            isChampion ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-700'
          }}`}}>
            {{isChampion ? '🏆 Champion Model' : '📈 Performance Model'}}
          </div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-900 text-lg">🏗️ Architecture</h3>
            <div className="space-y-2 text-sm">
              <p><span className="font-medium">Data Type:</span> {{metadata.data_type || 'Unknown'}}</p>
              <p><span className="font-medium">Model:</span> {{metadata.model || model}}</p>
              {{metadata.data_type === 'dense' && (
                <div className="bg-blue-50 p-2 rounded text-blue-800">
                  ✅ Using 64-dimensional autoencoder embeddings
                </div>
              )}}
              {{metadata.data_type === 'sparse' && (
                <div className="bg-green-50 p-2 rounded text-green-800">
                  📊 Using direct feature scaling
                </div>
              )}}
            </div>
          </div>

          {{metadata.best_params && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900 text-lg">⚙️ Hyperparameters</h3>
              <div className="space-y-1 text-sm">
                {{Object.entries(metadata.best_params).map(([key, value]) => (
                  <p key={{key}}>
                    <span className="font-medium">{{key.replace('_', ' ')}}:</span>{' '}
                    {{Array.isArray(value) ? value.join(', ') : String(value)}}
                  </p>
                ))}}
              </div>
            </div>
          )}}

          {{data.total_predictions && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-900 text-lg">📊 Statistics</h3>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Total Predictions:</span> {{data.total_predictions.toLocaleString()}}</p>
                <p><span className="font-medium">Training Date:</span> {{metadata.training_timestamp?.split('T')[0] || 'Unknown'}}</p>
              </div>
            </div>
          )}}
        </div>

        {{model === 'mlp' && metadata.best_params && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-lg mb-3">🧠 Neural Network Architecture</h3>
            <div className="text-sm text-gray-700">
              <p className="mb-2">
                <span className="font-medium">Network Structure:</span> 
                Input (64) → {{metadata.best_params.hidden_layer_sizes?.[0] || 50}} → {{metadata.best_params.hidden_layer_sizes?.[1] || 50}} → Output (1)
              </p>
              <p className="mb-2">
                <span className="font-medium">Regularization (α):</span> {{metadata.best_params.alpha?.toFixed(6) || 'Unknown'}}
              </p>
              <p className="mb-2">
                <span className="font-medium">Learning Rate:</span> {{metadata.best_params.learning_rate_init?.toFixed(6) || 'Unknown'}}
              </p>
              <p>
                <span className="font-medium">Max Iterations:</span> {{metadata.best_params.max_iter || 'Unknown'}}
              </p>
            </div>
          </div>
        )}}
      </div>

      {{data.sample_predictions && (
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
                {{data.sample_predictions.slice(0, 10).map((pred, idx) => (
                  <tr key={{idx}} className={{idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{new Date(pred.timestamp).toLocaleString()}}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{pred.true_count?.toFixed(0) || 'N/A'}}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {{pred.pred_count?.toFixed(0) || 'N/A'}}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {{pred.true_count && pred.pred_count ? (
                        <span className={{`${{
                          Math.abs(pred.true_count - pred.pred_count) < 10 
                            ? 'text-green-600' 
                            : Math.abs(pred.true_count - pred.pred_count) < 25 
                              ? 'text-yellow-600' 
                              : 'text-red-600'
                        }}`}}>
                          {{(pred.true_count - pred.pred_count).toFixed(1)}}
                        </span>
                      ) : 'N/A'}}
                    </td>
                  </tr>
                ))}}
              </tbody>
            </table>
          </div>
        </div>
      )}}

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4">📊 Model Visualizations</h3>
        <div className="grid gap-6">
          {{{JSON.stringify(image_files)}
            .filter(img => img.includes(model))
            .map(imageName => (
              <div key={{imageName}} className="relative">
                <h4 className="font-medium text-gray-700 mb-3">
                  {{imageName.replace('model_analysis_', '').replace('.png', '').replace(/_/g, ' ').replace(model, '').trim() || 'Analysis Results'}}
                </h4>
                <div className="bg-gray-100 rounded-lg overflow-hidden">
                  <Image
                    src={{`/images/${{imageName}}`}}
                    alt={{imageName}}
                    width={{1200}}
                    height={{900}}
                    className="w-full h-auto"
                  />
                </div>
              </div>
            ))}}
        </div>
      </div>
    </div>
  );
}}
'''
        
        with open(self.vercel_dir / "pages" / "index.js", "w") as f:
            f.write(page_content)
        
        print("✅ Main page generated")
    
    def generate_styles(self):
        """Generate global styles"""
        styles_content = '''@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html {
    font-family: system-ui, sans-serif;
  }
}

@layer components {
  .card {
    @apply bg-white rounded-lg shadow-md p-6;
  }
  
  .btn-primary {
    @apply bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors;
  }
  
  .metric-card {
    @apply bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200;
  }
}
'''
        
        with open(self.vercel_dir / "styles" / "globals.css", "w") as f:
            f.write(styles_content)
        
        print("✅ Global styles generated")
    
    def generate_vercel_config(self):
        """Generate vercel.json configuration"""
        vercel_config = {
            "version": 2,
            "name": "gate-token-eda-dashboard",
            "builds": [
                {
                    "src": "package.json",
                    "use": "@vercel/next"
                }
            ],
            "routes": [
                {
                    "src": "/images/(.*)",
                    "dest": "/images/$1"
                }
            ]
        }
        
        with open(self.vercel_dir / "vercel.json", "w") as f:
            json.dump(vercel_config, f, indent=2)
        
        print("✅ vercel.json created")
    
    def generate_deployment_info(self):
        """Generate deployment information"""
        deploy_info = {
            "project_name": "gate-token-eda-dashboard",
            "description": "Gate Token Predictive Modeling EDA Dashboard",
            "framework": "nextjs",
            "created": datetime.now().isoformat(),
            "deployment_steps": [
                "1. cd vercel-eda-app",
                "2. Install Vercel CLI: npm i -g vercel",
                "3. Login to Vercel: vercel login",
                "4. Deploy: vercel --prod"
            ]
        }
        
        with open(self.vercel_dir / "deployment-info.json", "w") as f:
            json.dump(deploy_info, f, indent=2)
        
        return deploy_info
    
    def create_readme(self):
        """Create README for the Vercel app"""
        readme_content = '''# Gate Token EDA Dashboard

This is a Next.js web application that displays the analysis results from the Gate Token Predictive Modeling System.

## Features

- 📊 **Model Performance Overview**: Compare all trained models
- 🏆 **Champion Model Highlighting**: Automatic champion model identification
- 🧠 **System Architecture Display**: Show data encoding and model details
- 📈 **Interactive Visualizations**: Display all generated analysis plots
- 📋 **Sample Predictions**: Show prediction samples for each model
- 🎯 **Responsive Design**: Works on desktop and mobile devices

## Deployment on Vercel

### Prerequisites
```bash
npm install -g vercel
```

### Steps
1. Navigate to the app directory:
   ```bash
   cd vercel-eda-app
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy to production:
   ```bash
   vercel --prod
   ```

### Environment
- Framework: Next.js 13.5.6
- Styling: Tailwind CSS
- Charts: Recharts
- Icons: Lucide React

## Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
vercel-eda-app/
├── pages/
│   └── index.js          # Main dashboard page
├── public/
│   ├── images/           # Visualization images
│   └── data/            # Model data JSON
├── styles/
│   └── globals.css      # Global styles
├── package.json         # Dependencies
├── next.config.js       # Next.js configuration
└── vercel.json         # Vercel deployment config
```

## Data Sources

- Model metadata from YAML files
- Prediction samples from CSV files  
- Visualization images (PNG files)
- Champion model information

Generated from Gate Token Predictive Modeling System.
'''
        
        with open(self.vercel_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print("✅ README.md created")
    
    def deploy(self):
        """Main deployment function"""
        print("🚀 Starting Vercel EDA Deployment...")
        print("="*50)
        
        # Create app structure
        self.create_vercel_app_structure()
        
        # Generate configuration files
        self.generate_package_json()
        self.generate_next_config()
        self.generate_tailwind_config()
        self.generate_vercel_config()
        
        # Copy assets and collect data
        image_files = self.copy_visualization_images()
        models_data = self.collect_model_data()
        
        # Generate pages and styles
        self.generate_main_page(image_files, models_data)
        self.generate_styles()
        
        # Generate documentation
        deploy_info = self.generate_deployment_info()
        self.create_readme()
        
        print("="*50)
        print("✅ Vercel app created successfully!")
        print(f"📁 Location: {self.vercel_dir}")
        print(f"📊 Models included: {len(models_data)-1}")
        print(f"🖼️ Visualizations: {len(image_files)}")
        print(f"🏆 Champion model: {models_data.get('_champion', 'unknown')}")
        
        return deploy_info

def main():
    deployer = VercelEDADeployer()
    deploy_info = deployer.deploy()
    
    print("\n" + "="*50)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*50)
    
    print("\n📝 Next Steps:")
    for step in deploy_info["deployment_steps"]:
        print(f"   {step}")
    
    print(f"\n🌐 After deployment, your dashboard will be available at:")
    print("   https://your-project-name.vercel.app")
    
    print(f"\n📋 Vercel Project Settings:")
    print(f"   • Project Name: {deploy_info['project_name']}")
    print(f"   • Framework: {deploy_info['framework']}")
    print(f"   • Build Command: npm run build")
    print(f"   • Output Directory: out (or leave default)")
    
    print("\n💡 Tips:")
    print("   • Make sure you have a Vercel account")
    print("   • The deployment process is automatic")
    print("   • Your dashboard will update when you redeploy")
    print("   • All images and data are included in the build")

if __name__ == "__main__":
    main()
