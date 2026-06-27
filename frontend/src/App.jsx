import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Sparkles, 
  RefreshCw, 
  Sliders, 
  Award, 
  AlertCircle,
  HelpCircle,
  FileImage,
  ArrowRight,
  Maximize2
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const MODEL_LABELS = {
  cnn: { name: 'CNN', desc: 'Simple CNN (MSE)' },
  unet_final: { name: 'UNet', desc: 'U-Net (MSE)' },
  unet_vgg: { name: 'UNet + VGG', desc: 'U-Net (Perceptual Loss)' },
  resnet_vgg: { name: 'ResNet + VGG', desc: 'ResNet-U-Net (Perceptual)' },
  resnet_gan: { name: 'ResNet + GAN', desc: 'ResNet-U-Net (Adversarial)' }
};

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [data, setData] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Slider states
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const sliderContainerRef = useRef(null);

  // Best model based on PSNR
  const [bestModelKey, setBestModelKey] = useState(null);

  // Handle Drag Events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle Drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        processFile(file);
      } else {
        setError('Please upload an image file.');
      }
    }
  };

  // Handle File Input Change
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setData(null);
    setError(null);
    setBestModelKey(null);
  };

  // Trigger Upload and Inference
  const runDeblurring = async () => {
    if (!imageFile) return;

    setLoading(true);
    setError(null);
    setData(null);
    setBestModelKey(null);

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch(`${API_BASE_URL}/deblur`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Failed to run models.');
      }

      const result = await response.json();
      setData(result);
      
      // Determine the best model key based on highest PSNR
      if (result.results && Object.keys(result.results).length > 0) {
        let highestPsnr = -1;
        let bestKey = null;
        Object.entries(result.results).forEach(([key, val]) => {
          if (val.psnr > highestPsnr) {
            highestPsnr = val.psnr;
            bestKey = key;
          }
        });
        setBestModelKey(bestKey);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during inference. Make sure the backend server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  // Reset App
  const resetAll = () => {
    setImageFile(null);
    setImagePreview(null);
    setData(null);
    setError(null);
    setBestModelKey(null);
    setSliderPosition(50);
  };

  // Slider Mouse/Touch Handlers
  const handleSliderMove = (clientX) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e) => {
    if (isDraggingSlider && e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e) => {
    if (isDraggingSlider) {
      handleSliderMove(e.clientX);
    }
  };

  // Listen globally for mouseup to stop sliding
  useEffect(() => {
    const handleMouseUp = () => {
      setIsDraggingSlider(false);
    };

    if (isDraggingSlider) {
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchend', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDraggingSlider]);

  // Color mapping for Blur Severity
  const getBlurColorClass = (level) => {
    switch (level) {
      case 'SHARP': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
      case 'MILD': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
      case 'MEDIUM': return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
      case 'HEAVY': return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/20';
    }
  };

  // Render PSNR SVG Bar Chart
  const renderBarChart = () => {
    if (!data || !data.results) return null;

    const chartHeight = 180;
    const paddingLeft = 40;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 40;
    const width = 600;
    const height = chartHeight + paddingTop + paddingBottom;

    const entries = Object.entries(data.results); // [['cnn', {psnr, ssim}], ...]
    
    // Scale PSNR scores (usually between 15 and 45)
    // Find min and max for scaling
    const psnrs = entries.map(([_, v]) => v.psnr);
    const minPsnr = 0; // Baseline
    const maxPsnr = Math.max(...psnrs, 35) + 5; // Give headroom

    const getX = (index) => {
      const step = (width - paddingLeft - paddingRight) / entries.length;
      return paddingLeft + index * step + step / 4;
    };

    const getY = (psnrValue) => {
      const scaleHeight = height - paddingTop - paddingBottom;
      const ratio = (psnrValue - minPsnr) / (maxPsnr - minPsnr);
      return height - paddingBottom - ratio * scaleHeight;
    };

    const barWidth = (width - paddingLeft - paddingRight) / entries.length / 2;

    return (
      <div className="w-full flex justify-center mt-8">
        <div className="w-full max-w-2xl glass-panel p-6">
          <h3 className="text-lg font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-blue-400" /> Model Performance comparison (PSNR Score in dB)
          </h3>
          <div className="relative">
            <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
              <defs>
                <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
                <linearGradient id="bestBarGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#059669" />
                </linearGradient>
              </defs>

              {/* Grid Lines */}
              {[10, 20, 30, 40].map((tick) => (
                <g key={tick} className="opacity-15">
                  <line 
                    x1={paddingLeft} 
                    y1={getY(tick)} 
                    x2={width - paddingRight} 
                    y2={getY(tick)} 
                    stroke="#ffffff" 
                    strokeWidth="1" 
                    strokeDasharray="4 4"
                  />
                  <text 
                    x={paddingLeft - 8} 
                    y={getY(tick) + 4} 
                    fill="#ffffff" 
                    fontSize="11" 
                    textAnchor="end"
                  >
                    {tick}
                  </text>
                </g>
              ))}

              {/* Bars */}
              {entries.map(([key, val], idx) => {
                const isBest = key === bestModelKey;
                const barHeight = height - paddingBottom - getY(val.psnr);
                const x = getX(idx);
                const y = getY(val.psnr);

                return (
                  <g key={key} className="group">
                    {/* Badge label above the bar (always visible) */}
                    <rect
                      x={x}
                      y={y - 32}
                      width={barWidth}
                      height="20"
                      rx="4"
                      fill="#0f172a"
                      stroke={isBest ? "#10b981" : "#3b82f6"}
                      strokeWidth="1"
                    />
                    <text
                      x={x + barWidth / 2}
                      y={y - 22}
                      dy="0.35em"
                      fill={isBest ? "#34d399" : "#f8fafc"}
                      fontSize="10"
                      fontWeight="bold"
                      textAnchor="middle"
                    >
                      {val.psnr} dB
                    </text>

                    {/* Bar */}
                    <rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={barHeight}
                      rx="6"
                      fill={`url(#${isBest ? 'bestBarGradient' : 'barGradient'})`}
                      className="transition-all duration-300 group-hover:brightness-125 cursor-pointer"
                    />

                    {/* X axis Labels */}
                    <text
                      x={x + barWidth / 2}
                      y={height - 20}
                      fill="#94a3b8"
                      fontSize="11"
                      fontWeight="500"
                      textAnchor="middle"
                    >
                      {MODEL_LABELS[key]?.name || key}
                    </text>
                  </g>
                );
              })}

              {/* Bottom line axis */}
              <line 
                x1={paddingLeft} 
                y1={height - paddingBottom} 
                x2={width - paddingRight} 
                y2={height - paddingBottom} 
                stroke="rgba(255,255,255,0.15)" 
                strokeWidth="1.5"
              />
            </svg>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container min-h-screen flex flex-col">
      {/* Header */}
      <header className="mb-10 text-center relative">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider mb-3">
          <Sparkles className="w-3.5 h-3.5" /> Deep Learning restoration
        </div>
        <h1 className="text-4xl md:text-5xl font-black tracking-tight text-white mb-3">
          Image Deblurring <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500">Arena</span>
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto text-base">
          Compare classical CNN, U-Net, and ResNet architectures trained with MSE, Perceptual (VGG) and GAN loss functions.
        </p>

        {/* Reset button at top if results loaded */}
        {data && (
          <button 
            onClick={resetAll}
            className="absolute top-0 right-0 flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-medium transition duration-200 border border-slate-700 shadow-lg text-sm"
          >
            <RefreshCw className="w-4 h-4" /> Reset Workspace
          </button>
        )}
      </header>

      {/* Main Workspace Area */}
      {!data && !loading && (
        <div className="flex-1 flex flex-col items-center justify-center max-w-xl mx-auto w-full">
          {/* Upload card */}
          <div className="w-full glass-panel p-8 mb-6">
            <div 
              className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
            >
              <input 
                id="file-upload" 
                type="file" 
                className="hidden" 
                accept="image/*" 
                onChange={handleFileChange} 
              />
              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                <div className="w-16 h-16 rounded-2xl bg-blue-500/10 flex items-center justify-center mb-4 text-blue-400 border border-blue-500/15">
                  <Upload className="w-8 h-8" />
                </div>
                <p className="text-lg font-semibold text-slate-200 mb-1">Drag and drop blurry image</p>
                <p className="text-sm text-slate-400 mb-4">Supports PNG, JPG, JPEG</p>
                <span className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-lg shadow-lg hover:shadow-blue-500/20 transition duration-200">
                  Select File
                </span>
              </label>
            </div>
            
            {imagePreview && (
              <div className="mt-6 p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-lg overflow-hidden border border-slate-700">
                    <img src={imagePreview} alt="Preview" className="w-100 h-100 object-cover" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-slate-200 truncate max-w-[200px]">
                      {imageFile.name}
                    </p>
                    <p className="text-xs text-slate-500">
                      {(imageFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button 
                  onClick={runDeblurring}
                  className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white font-semibold rounded-lg shadow-lg transition duration-200"
                >
                  <Sparkles className="w-4 h-4" /> Run Deblur
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="w-full flex items-start gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm">
              <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Execution Failed</p>
                <p className="text-rose-400/80">{error}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="flex-1 flex flex-col items-center justify-center py-20">
          <div className="spinner mb-6"></div>
          <h2 className="text-xl font-bold text-white mb-2">Running all models...</h2>
          <p className="text-slate-400 max-w-sm text-center text-sm">
            Calculating predictions across CNN, U-Net, ResNet and GAN generators on CUDA. This takes just a moment...
          </p>
        </div>
      )}

      {/* Results View */}
      {data && (
        <div className="space-y-10 animate-fade-in w-full flex flex-col items-center">
          {/* Top Metadata & Severity display */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-12 p-6 rounded-2xl glass-panel mx-auto w-full max-w-4xl text-center">
            <div className="flex flex-col items-center">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Analyzed Image</p>
              <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2 justify-center mt-1">
                <FileImage className="w-4 h-4 text-indigo-400" /> {imageFile.name}
              </h4>
            </div>

            <div className="hidden md:block h-8 w-px bg-slate-800"></div>

            <div className="flex flex-col items-center">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Laplacian Variance</p>
              <p className="text-sm font-bold text-slate-300 mt-1">{data.laplacian_var}</p>
            </div>

            <div className="hidden md:block h-8 w-px bg-slate-800"></div>

            <div className="flex flex-col items-center">
              <div className={`px-4 py-2 border rounded-xl text-sm font-bold flex items-center gap-2 ${getBlurColorClass(data.blur_level)}`}>
                <span className="w-2.5 h-2.5 rounded-full bg-current"></span>
                Blur Level: {data.blur_level}
              </div>
            </div>
          </div>

          {/* horizontal comparison grid with 6 panels */}
          <div className="w-full flex flex-col items-center">
            <h2 className="text-xl font-bold text-white mb-6 flex items-center justify-center gap-2 text-center">
              <Maximize2 className="w-5 h-5 text-indigo-400" /> Horizontal Comparison Grid
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-6 md:gap-8 justify-center justify-items-center w-full">
              {/* Panel 1: Original Blurry Input */}
              <div className="glass-panel overflow-hidden flex flex-col w-full">
                <div className="aspect-square relative w-full overflow-hidden bg-slate-950 flex items-center justify-center">
                  <img 
                    src={data.original_image} 
                    alt="Original Blurry" 
                    className="w-100 h-100 object-cover" 
                  />
                  <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-slate-900/90 text-slate-300 text-xs px-2.5 py-1 rounded-md font-semibold border border-slate-700 backdrop-blur whitespace-nowrap">
                    Original
                  </div>
                </div>
                <div className="p-3 bg-slate-900/40 flex-1 flex flex-col justify-center items-center text-center">
                  <p className="text-sm font-bold text-slate-300">Original Input</p>
                  <p className="text-xs text-slate-500">Blurry reference</p>
                </div>
              </div>

              {/* Panels 2-6: Model Outputs */}
              {Object.entries(MODEL_LABELS).map(([key, info]) => {
                const modelResult = data.results[key];
                if (!modelResult) return null;
                const isBest = key === bestModelKey;

                return (
                  <div 
                    key={key} 
                    className={`glass-panel overflow-hidden flex flex-col w-full ${isBest ? 'best-glow-animate' : ''}`}
                  >
                    <div className="aspect-square relative w-full overflow-hidden bg-slate-950 flex items-center justify-center">
                      <img 
                        src={modelResult.image} 
                        alt={info.name} 
                        className="w-100 h-100 object-cover" 
                      />
                      <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-slate-900/90 text-slate-300 text-xs px-2.5 py-1 rounded-md font-semibold border border-slate-700 backdrop-blur whitespace-nowrap">
                        {info.name}
                      </div>
                      {isBest && (
                        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-emerald-500 text-slate-950 text-xs px-2.5 py-0.5 rounded-full font-extrabold flex items-center gap-1 shadow-md shadow-emerald-500/25 whitespace-nowrap">
                          <Award className="w-3.5 h-3.5" /> BEST PSNR
                        </div>
                      )}
                    </div>
                    <div className="p-3 bg-slate-900/40 flex-1 flex flex-col justify-between items-center text-center">
                      <div className="text-center mb-3">
                        <p className="text-sm font-bold text-slate-200">{info.name}</p>
                        <p className="text-[10px] text-slate-500 truncate max-w-[120px] mx-auto">{info.desc}</p>
                      </div>

                      {/* Badges container */}
                      <div className="space-y-1.5 w-full">
                        <div className="flex justify-center items-center gap-1.5 text-xs bg-slate-950/60 px-2.5 py-1 rounded-md border border-slate-800">
                          <span className="text-slate-500">PSNR:</span>
                          <span className={`font-bold ${isBest ? 'text-emerald-400' : 'text-slate-300'}`}>
                            {modelResult.psnr} dB
                          </span>
                        </div>
                        <div className="flex justify-center items-center gap-1.5 text-xs bg-slate-950/60 px-2.5 py-1 rounded-md border border-slate-800">
                          <span className="text-slate-500">SSIM:</span>
                          <span className="font-bold text-slate-300">
                            {modelResult.ssim}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Interactive slider between Original and Best Result */}
          {bestModelKey && data.results[bestModelKey] && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center justify-center pt-4 max-w-5xl mx-auto w-full">
              <div className="lg:col-span-5 space-y-4 text-center flex flex-col items-center justify-center">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase mb-2 mx-auto">
                  <Sliders className="w-3.5 h-3.5" /> Slider Comparison
                </div>
                <h3 className="text-2xl font-bold text-white tracking-tight text-center">
                  Drag Side-by-Side Comparison
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed text-center max-w-md mx-auto">
                  Interactively compare the <strong>Original Blurry Input</strong> (left side) directly with the <strong>{MODEL_LABELS[bestModelKey].name}</strong> output (right side). 
                </p>
                <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 text-xs text-slate-400 w-full max-w-sm mx-auto text-left">
                  <div className="flex items-center gap-2 justify-center">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    <span>Left Side: Original Blurry Input</span>
                  </div>
                  <div className="flex items-center gap-2 justify-center">
                    <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                    <span>Right Side: {MODEL_LABELS[bestModelKey].name} ({data.results[bestModelKey].psnr} dB)</span>
                  </div>
                </div>
              </div>

              <div className="lg:col-span-7 flex justify-center w-full">
                <div 
                  ref={sliderContainerRef}
                  className="slider-container max-w-md w-full shadow-2xl mx-auto"
                  onMouseMove={handleMouseMove}
                  onTouchMove={handleTouchMove}
                  onMouseDown={() => setIsDraggingSlider(true)}
                  onTouchStart={() => setIsDraggingSlider(true)}
                >
                  {/* Underlay Image: Original Blurry */}
                  <img 
                    src={data.original_image} 
                    alt="Blurry Original" 
                    className="slider-image" 
                  />

                  {/* Overlay Image: Best Result, Clipped */}
                  <img 
                    src={data.results[bestModelKey].image} 
                    alt="Best Restored" 
                    className="slider-image"
                    style={{
                      clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
                    }}
                  />

                  {/* Vertical separator line */}
                  <div 
                    className="slider-handle"
                    style={{ left: `${sliderPosition}%` }}
                  />

                  {/* Slider drag button */}
                  <div 
                    className="slider-button"
                    style={{ left: `${sliderPosition}%` }}
                  >
                    <Sliders className="w-4 h-4" />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PSNR score bar chart */}
          {renderBarChart()}
        </div>
      )}
      
      {/* Footer */}
      {!data && (
        <footer className="mt-auto py-8 text-center border-t border-slate-900 text-xs text-slate-600">
          Image Deblurring Model Arena &bull; GPU Acceleration (CUDA) &bull; Built with FastAPI & React
        </footer>
      )}
    </div>
  );
}
