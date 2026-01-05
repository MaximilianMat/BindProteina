/**
 * Metrics display and filtering.
 */

class MetricsDisplay {
    constructor() {
        this.currentMetrics = {};
    }
    
    displayMetrics(metrics, containerId = "metricsDisplay") {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        this.currentMetrics = metrics;
        
        // Clear existing content
        container.innerHTML = "";
        
        // Display key metrics
        const keyMetrics = [
            "i_pTM", "pTM", "pLDDT", "average_plddt",
            "interface_dG", "interface_dSASA", "interface_sc",
            "interface_nres", "interface_interface_hbonds",
            "linker_length"
        ];
        
        keyMetrics.forEach(key => {
            if (metrics[key] !== undefined && metrics[key] !== null) {
                const metricItem = document.createElement("div");
                metricItem.className = "metric-item";
                
                const label = document.createElement("span");
                label.className = "metric-label";
                label.textContent = this.formatMetricName(key);
                
                const value = document.createElement("span");
                value.className = "metric-value";
                value.textContent = this.formatMetricValue(key, metrics[key]);
                
                metricItem.appendChild(label);
                metricItem.appendChild(value);
                container.appendChild(metricItem);
            }
        });
    }
    
    formatMetricName(key) {
        const names = {
            "i_pTM": "i-pTM",
            "pTM": "pTM",
            "pLDDT": "pLDDT",
            "average_plddt": "Avg pLDDT",
            "interface_dG": "Interface ΔG",
            "interface_dSASA": "Interface dSASA",
            "interface_sc": "Shape Complementarity",
            "interface_nres": "Interface Residues",
            "interface_interface_hbonds": "H-Bonds",
            "linker_length": "Linker Length"
        };
        return names[key] || key;
    }
    
    formatMetricValue(key, value) {
        if (typeof value === "number") {
            if (key.includes("pTM") || key.includes("pLDDT")) {
                return value.toFixed(2);
            } else if (key.includes("dG") || key.includes("dSASA")) {
                return value.toFixed(1);
            } else {
                return value.toString();
            }
        }
        return String(value);
    }
    
    filterDesigns(designs, filters) {
        return designs.filter(design => {
            for (const [key, condition] of Object.entries(filters)) {
                const value = design.metrics[key];
                if (value === undefined || value === null) {
                    return false;
                }
                
                if (condition.min !== undefined && value < condition.min) {
                    return false;
                }
                if (condition.max !== undefined && value > condition.max) {
                    return false;
                }
            }
            return true;
        });
    }
}

// Global metrics display instance
const metricsDisplay = new MetricsDisplay();

