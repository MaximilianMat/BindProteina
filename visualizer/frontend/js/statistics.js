/**
 * Statistics and visualization module for round analysis.
 */

class RoundStatistics {
    constructor() {
        this.container = null;
        this.globalRanges = {}; // Store global min/max for consistent scales
    }
    
    /**
     * Calculate statistics from a list of designs
     */
    calculateStats(designs) {
        if (!designs || designs.length === 0) {
            return null;
        }
        
        const stats = {
            totalDesigns: designs.length,
            i_pTM: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            pLDDT: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            dG: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            dSASA: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            shapeComplementarity: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            linkerLength: {
                values: [],
                average: 0,
                median: 0,
                min: Infinity,
                max: -Infinity,
                stdDev: 0
            },
            passingFilters: 0,
            hasAF2: 0,
            hasRelaxed: 0
        };
        
        // Extract values
        designs.forEach(design => {
            const metrics = design.metrics || {};
            
            // i_pTM
            if (metrics.i_pTM !== undefined && metrics.i_pTM !== null && !isNaN(metrics.i_pTM)) {
                stats.i_pTM.values.push(metrics.i_pTM);
            }
            
            // pLDDT (try multiple possible field names)
            const plddt = metrics.average_plddt || metrics.pLDDT || metrics.plddt;
            if (plddt !== undefined && plddt !== null && !isNaN(plddt)) {
                stats.pLDDT.values.push(plddt);
            }
            
            // dG (binding energy - lower is better, try multiple field names)
            const dg = metrics.interface_dG || metrics.dG || metrics.interface_dg;
            if (dg !== undefined && dg !== null && !isNaN(dg)) {
                stats.dG.values.push(dg);
            }
            
            // dSASA (buried surface area, try multiple field names)
            const dsasa = metrics.interface_dSASA || metrics.dSASA || metrics.interface_dsasa;
            if (dsasa !== undefined && dsasa !== null && !isNaN(dsasa)) {
                stats.dSASA.values.push(dsasa);
            }
            
            // Shape Complementarity (try multiple field names)
            const sc = metrics.interface_sc || metrics.shape_complementarity || metrics.sc;
            if (sc !== undefined && sc !== null && !isNaN(sc)) {
                stats.shapeComplementarity.values.push(sc);
            }
            
            // Linker Length (try multiple possible field names)
            const linkerLen = metrics.linker_length || metrics.linkerLength || metrics.link_length || metrics.linker_len;
            if (linkerLen !== undefined && linkerLen !== null && !isNaN(linkerLen)) {
                stats.linkerLength.values.push(linkerLen);
            }
            
            // Count passing filters (assuming pass/fail/skip logic)
            if (metrics.pass === true || metrics.status === 'pass') {
                stats.passingFilters++;
            }
            
            // Count available structures
            if (design.af2_pdb_path) stats.hasAF2++;
            if (design.relaxed_pdb_path) stats.hasRelaxed++;
        });
        
        // Calculate statistics for each metric
        ['i_pTM', 'pLDDT', 'dG', 'dSASA', 'shapeComplementarity', 'linkerLength'].forEach(metric => {
            const values = stats[metric].values;
            if (values.length > 0) {
                // Sort for median calculation
                const sorted = [...values].sort((a, b) => a - b);
                
                stats[metric].average = values.reduce((a, b) => a + b, 0) / values.length;
                stats[metric].median = sorted.length % 2 === 0
                    ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
                    : sorted[Math.floor(sorted.length / 2)];
                stats[metric].min = Math.min(...values);
                stats[metric].max = Math.max(...values);
                
                // Standard deviation
                const variance = values.reduce((sum, val) => sum + Math.pow(val - stats[metric].average, 2), 0) / values.length;
                stats[metric].stdDev = Math.sqrt(variance);
            }
        });
        
        return stats;
    }
    
    /**
     * Create a histogram canvas for a metric
     * @param {Array} values - Array of metric values
     * @param {number} width - Canvas width
     * @param {number} height - Canvas height
     * @param {string} color - Bar color
     * @param {Object} globalRange - Optional {min, max} for fixed scale
     */
    createHistogram(values, width = 400, height = 200, color = '#3498db', globalRange = null) {
        if (!values || values.length === 0) {
            return null;
        }
        
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        // Use global range if provided, otherwise use local range
        const min = globalRange ? globalRange.min : Math.min(...values);
        const max = globalRange ? globalRange.max : Math.max(...values);
        
        // Calculate bins (use fewer bins for smaller width)
        const numBins = width < 300 ? 15 : 20;
        const binWidth = (max - min) / numBins;
        
        // Count values in each bin
        const bins = new Array(numBins).fill(0);
        values.forEach(val => {
            const binIndex = Math.min(Math.floor((val - min) / binWidth), numBins - 1);
            bins[binIndex]++;
        });
        
        const maxCount = Math.max(...bins);
        
        // Draw histogram with reduced padding for compact display
        const padding = width < 300 ? 30 : 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        
        // Clear canvas
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);
        
        // Draw bars
        const barWidth = chartWidth / numBins;
        bins.forEach((count, i) => {
            const barHeight = (count / maxCount) * chartHeight;
            const x = padding + i * barWidth;
            const y = padding + chartHeight - barHeight;
            
            ctx.fillStyle = color;
            ctx.fillRect(x, y, Math.max(1, barWidth - 1), barHeight);
        });
        
        // Draw axes
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, padding + chartHeight);
        ctx.lineTo(padding + chartWidth, padding + chartHeight);
        ctx.stroke();
        
        // Draw labels (smaller font for compact display)
        ctx.fillStyle = '#333';
        ctx.font = width < 300 ? '10px Arial' : '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(min.toFixed(2), padding, height - 3);
        ctx.fillText(max.toFixed(2), padding + chartWidth, height - 3);
        
        // Skip Y-axis label for compact display
        if (width >= 300) {
            ctx.textAlign = 'left';
            ctx.save();
            ctx.translate(5, padding + chartHeight / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Count', 0, 0);
            ctx.restore();
        }
        
        return canvas;
    }
    
    /**
     * Calculate global ranges across all rounds for consistent scales
     */
    calculateGlobalRanges(allDesigns) {
        const allStats = this.calculateStats(allDesigns);
        if (!allStats) return;
        
        // Store global min/max for each metric
        ['i_pTM', 'pLDDT', 'dG', 'dSASA', 'shapeComplementarity', 'linkerLength'].forEach(metric => {
            if (allStats[metric] && allStats[metric].values.length > 0) {
                this.globalRanges[metric] = {
                    min: allStats[metric].min,
                    max: allStats[metric].max
                };
            }
        });
    }
    
    /**
     * Display statistics for a round
     */
    displayStats(designs, containerId = 'roundStats', useGlobalScale = false) {
        const stats = this.calculateStats(designs);
        if (!stats) {
            return;
        }
        
        let container = document.getElementById(containerId);
        if (!container) {
            // Create container if it doesn't exist
            container = document.createElement('div');
            container.id = containerId;
            container.className = 'round-stats';
            
            // Insert into stats view
            const statsView = document.querySelector('.stats-view');
            if (statsView) {
                const headerControls = statsView.querySelector('.stats-header-controls');
                if (headerControls && headerControls.nextSibling) {
                    statsView.insertBefore(container, headerControls.nextSibling);
                } else {
                    statsView.appendChild(container);
                }
            }
        }
        
        container.innerHTML = '';
        
        // Create stats HTML
        const statsHTML = `
            <div class="stats-panel">
                <div class="stats-header">
                    <h3>Summary</h3>
                    <div class="stats-summary">
                        <div class="stat-item">
                            <span class="stat-label">Total Designs:</span>
                            <span class="stat-value">${stats.totalDesigns}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Passing Filters:</span>
                            <span class="stat-value">${stats.passingFilters} (${((stats.passingFilters / stats.totalDesigns) * 100).toFixed(1)}%)</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">AF2 Structures:</span>
                            <span class="stat-value">${stats.hasAF2}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Relaxed Structures:</span>
                            <span class="stat-value">${stats.hasRelaxed}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="stats-panel">
                <div class="stats-metrics">
                    ${this.createMetricSection('i_pTM', stats.i_pTM, '#3498db')}
                    ${this.createMetricSection('Average pLDDT', stats.pLDDT, '#27ae60')}
                    ${this.createMetricSection('Binding Energy (dG)', stats.dG, '#e74c3c')}
                    ${this.createMetricSection('Buried Surface Area (dSASA)', stats.dSASA, '#f39c12')}
                    ${this.createMetricSection('Shape Complementarity', stats.shapeComplementarity, '#9b59b6')}
                    ${this.createMetricSection('Linker Length', stats.linkerLength, '#16a085')}
                </div>
            </div>
        `;
        
        container.innerHTML = statsHTML;
        
        // Add collapsible functionality
        const metricSections = container.querySelectorAll('.metric-section h4');
        metricSections.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                section.classList.toggle('collapsed');
            });
        });
        
        // Add histograms
        const metricsContainer = container.querySelector('.stats-metrics');
        if (metricsContainer) {
            // Add histograms for metrics with data
            const metricsToHistogram = [
                { key: 'i_pTM', label: 'i_pTM', color: '#3498db' },
                { key: 'pLDDT', label: 'Average pLDDT', color: '#27ae60' },
                { key: 'dG', label: 'Binding Energy (dG)', color: '#e74c3c' },
                { key: 'linkerLength', label: 'Linker Length', color: '#16a085' }
            ];
            
            metricsToHistogram.forEach(({ key, label, color }) => {
                const metricData = stats[key];
                if (metricData && metricData.values.length > 0) {
                    // For metrics that are naturally 0-1 (like i_pTM, pLDDT/100), use fixed 0-1 scale
                    // Otherwise use global scale if available
                    let globalRange = null;
                    if (key === 'i_pTM' || key === 'pLDDT') {
                        // Use fixed 0-1 scale for these metrics (pLDDT is 0-100, but we'll use 0-1 for normalized view)
                        // Actually, for pLDDT, values are typically 0-100, so we might want 0-100 scale
                        // But for i_pTM, definitely use 0-1
                        if (key === 'i_pTM') {
                            globalRange = { min: 0, max: 1 };
                        } else if (key === 'pLDDT') {
                            // pLDDT is typically 0-100, but check if values are normalized
                            const maxVal = Math.max(...metricData.values);
                            if (maxVal <= 1.0) {
                                globalRange = { min: 0, max: 1 };
                            } else {
                                globalRange = { min: 0, max: 100 };
                            }
                        }
                    } else if (useGlobalScale && this.globalRanges[key]) {
                        // Use global range for other metrics
                        globalRange = this.globalRanges[key];
                    }
                    
                    const histogram = this.createHistogram(
                        metricData.values, 
                        600, 
                        200, 
                        color,
                        globalRange
                    );
                    if (histogram) {
                        // Match by label (which gets converted to data-metric attribute)
                        const sectionKey = label.replace(/\s+/g, '_');
                        const metricSection = metricsContainer.querySelector(`.metric-section[data-metric="${sectionKey}"]`);
                        if (metricSection) {
                            const metricContent = metricSection.querySelector('.metric-content');
                            if (metricContent) {
                                const histogramContainer = document.createElement('div');
                                histogramContainer.className = 'histogram-container';
                                histogramContainer.appendChild(histogram);
                                metricContent.appendChild(histogramContainer);
                            }
                        }
                    }
                }
            });
        }
        
        this.container = container;
    }
    
    /**
     * Create HTML for a metric section
     */
    createMetricSection(label, metricData, color) {
        const sectionKey = label.replace(/\s+/g, '_');
        if (!metricData || metricData.values.length === 0) {
            return `
                <div class="metric-section" data-metric="${sectionKey}">
                    <h4>${label}</h4>
                    <div class="metric-content">
                        <div class="metric-stats">
                            <span class="no-data">No data available</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        return `
            <div class="metric-section" data-metric="${sectionKey}">
                <h4>${label}</h4>
                <div class="metric-content">
                    <div class="metric-stats">
                        <div class="metric-stat">
                            <span class="metric-label">Avg:</span>
                            <span class="metric-value">${metricData.average.toFixed(2)}</span>
                        </div>
                        <div class="metric-stat">
                            <span class="metric-label">Med:</span>
                            <span class="metric-value">${metricData.median.toFixed(2)}</span>
                        </div>
                        <div class="metric-stat">
                            <span class="metric-label">Min:</span>
                            <span class="metric-value">${metricData.min.toFixed(2)}</span>
                        </div>
                        <div class="metric-stat">
                            <span class="metric-label">Max:</span>
                            <span class="metric-value">${metricData.max.toFixed(2)}</span>
                        </div>
                        <div class="metric-stat">
                            <span class="metric-label">Std:</span>
                            <span class="metric-value">${metricData.stdDev.toFixed(2)}</span>
                        </div>
                        <div class="metric-stat">
                            <span class="metric-label">N:</span>
                            <span class="metric-value">${metricData.values.length}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Hide statistics panel
     */
    hide() {
        if (this.container) {
            this.container.style.display = 'none';
        }
    }
    
    /**
     * Show statistics panel
     */
    show() {
        if (this.container) {
            this.container.style.display = 'block';
        }
    }
}

// Global statistics instance
const roundStatistics = new RoundStatistics();

