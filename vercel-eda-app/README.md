# Gate Token EDA Dashboard

This is a Next.js web application that displays the analysis results from the Gate Token Predictive Modeling System.

## 🌟 Features

- 📊 **Model Performance Overview**: Compare all trained models
- 🏆 **Champion Model Highlighting**: Automatic champion model identification  
- 🧠 **System Architecture Display**: Show data encoding and model details
- 📈 **Interactive Visualizations**: Display all generated analysis plots
- 📋 **Sample Predictions**: Show prediction samples for each model
- 🎯 **Responsive Design**: Works on desktop and mobile devices

## 🚀 Deployment on Vercel

### Prerequisites
```bash
npm install -g vercel
```

### Quick Deploy Steps
1. **Navigate to app directory:**
   ```bash
   cd vercel-eda-app
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy to production:**
   ```bash
   vercel --prod
   ```

### Alternative: GitHub Integration
1. Push this folder to a GitHub repository
2. Connect the repository to Vercel dashboard
3. Auto-deploy on every push

## 🛠️ Vercel Configuration

When setting up the project in Vercel dashboard:

- **Framework Preset**: `Next.js`
- **Build Command**: `npm run build` (default)
- **Output Directory**: `out` (or leave default)  
- **Install Command**: `npm install` (default)
- **Development Command**: `npm run dev` (default)

## 📋 Project Structure

```
vercel-eda-app/
├── pages/
│   └── index.js          # Main dashboard page
├── public/
│   ├── images/           # Visualization images (5 files)
│   └── data/            # Model data JSON
├── styles/
│   └── globals.css      # Global styles
├── package.json         # Dependencies
├── next.config.js       # Next.js configuration
├── tailwind.config.js   # Tailwind CSS config
├── postcss.config.js    # PostCSS config
└── vercel.json         # Vercel deployment config
```

## 🎯 Dashboard Sections

### Overview Page
- **System Architecture**: 6-stage data pipeline overview
- **Model Performance Summary**: All 7+ trained models
- **Visualization Gallery**: All analysis plots

### Individual Model Pages
- **Architecture Details**: Data encoding type, hyperparameters
- **MLP Special Section**: Neural network architecture diagram
- **Sample Predictions**: Interactive prediction table
- **Model Visualizations**: Filtered plots for specific model

## 📊 Data Sources

- **Model Metadata**: From YAML files in predictions directory
- **Prediction Samples**: From CSV files (first 10 rows)
- **Visualization Images**: PNG files from model analysis
- **Champion Model**: From models/champion.txt

## 🔧 Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## 📱 Responsive Features

- **Desktop**: Full dashboard with side-by-side layouts
- **Tablet**: Stacked cards with horizontal scrolling navigation
- **Mobile**: Single column layout with touch-friendly navigation

## 🎨 Technologies Used

- **Framework**: Next.js 13.5.6
- **Styling**: Tailwind CSS 3.3.5
- **Icons**: Unicode emojis (universal compatibility)
- **Images**: Next.js optimized images
- **Data**: Static JSON (no database required)

## 🔍 Key Features Detail

### Champion Model Highlighting
- Golden background and trophy icon for current champion
- Automatic detection from champion.txt file
- Prominently displayed in navigation and cards

### MLP Architecture Visualization
- Special section for MLP model showing neural network structure
- Input → Hidden Layer 1 → Hidden Layer 2 → Output diagram
- Detailed hyperparameters with scientific notation

### Data Encoding Display
- Dense models: "Using 64-dimensional autoencoder embeddings"
- Sparse models: "Using direct feature scaling"
- Visual badges with explanations

### Interactive Elements
- **Model Navigation**: Click to switch between models
- **Image Hover**: Zoom effect on visualization images
- **Error Color Coding**: Green/Yellow/Red based on prediction accuracy
- **Responsive Tables**: Horizontal scroll on mobile

## 🚀 Performance Optimizations

- **Static Generation**: All pages pre-built at deploy time
- **Image Optimization**: Next.js automatic image optimization
- **CSS Purging**: Tailwind CSS purges unused styles
- **Gzip Compression**: Vercel automatic compression

## 📈 Analytics Ready

The dashboard is ready for analytics integration:
- **Vercel Analytics**: Add `@vercel/analytics` package
- **Google Analytics**: Add tracking script to `_document.js`
- **Custom Events**: Track model selections and interactions

---

Generated from Gate Token Predictive Modeling System.
**Dashboard URL**: Will be provided after Vercel deployment.
