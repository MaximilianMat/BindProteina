/**
 * 3Dmol.js Viewer wrapper for 3D protein visualization.
 */

class ProteinViewer {
    constructor(containerId) {
        this.containerId = containerId;
        this.viewer = null;
        this.currentColoringMode = 'plddt'; // Track current coloring mode
        this.currentRepresentation = 'cartoon'; // Track current representation type
        this.isBinderHighlighted = false; // Track highlight state
        this.originalStyles = {}; // Store original styles for revert
        this.currentPdbText = null; // Store current PDB text for sequence extraction
        this.init();
    }
    
    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        // Check if $3Dmol is available
        if (typeof $3Dmol === 'undefined') {
            console.error("3Dmol.js is not available. Make sure the 3Dmol script is loaded.");
            container.innerHTML = "<div style='padding: 2rem; text-align: center; color: #e74c3c;'>3Dmol.js Viewer library not loaded. Please check your internet connection and reload the page.</div>";
            return;
        }
        
        try {
            // Ensure container has proper dimensions
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.position = "relative";
            
            // Create 3Dmol viewer
            this.viewer = $3Dmol.createViewer(container, {
                backgroundColor: "black"
            });
            
            // Ensure viewer fills container
            if (this.viewer && this.viewer.container) {
                this.viewer.container.style.width = "100%";
                this.viewer.container.style.height = "100%";
            }
            
            console.log("3Dmol viewer created successfully");
        } catch (error) {
            console.error("Error initializing viewer:", error);
            console.error("Error details:", error.stack);
            container.innerHTML = `<div style='padding: 2rem; text-align: center; color: #e74c3c;'>Error initializing viewer: ${error.message}</div>`;
        }
    }
    
    async loadStructure(pdbPath, options = {}) {
        if (!this.viewer) {
            console.error("Viewer not initialized. Cannot load structure.");
            return null;
        }
        
        if (!pdbPath) {
            console.error("No PDB path provided");
            return null;
        }
        
        console.log("Loading structure from:", pdbPath);
        
        try {
            // Clear existing models
            this.viewer.clear();
            
            // Fetch PDB file
            const response = await fetch(pdbPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch PDB file: ${response.status} ${response.statusText}`);
            }
            
            const pdbText = await response.text();
            
            // Use the shared loading logic
            return this._loadPdbText(pdbText, options);
        } catch (error) {
            console.error("Error loading structure:", error);
            return null;
        }
    }
    
    /**
     * Load structure from preloaded PDB text (for faster animation).
     * @param {string} pdbText - PDB file content as text
     * @param {Object} options - Loading options
     */
    async loadStructureFromText(pdbText, options = {}) {
        if (!this.viewer) {
            console.error("Viewer not initialized. Cannot load structure.");
            return null;
        }
        
        if (!pdbText) {
            console.error("No PDB text provided");
            return null;
        }
        
        return this._loadPdbText(pdbText, options);
    }
    
    /**
     * Internal method to load PDB text into viewer.
     * @private
     */
    _loadPdbText(pdbText, options = {}) {
        try {
            // Clear existing models
            this.viewer.clear();
            
            // Store current PDB text for sequence extraction
            this.currentPdbText = pdbText;
            
            // Store current coloring mode
            this.currentColoringMode = options.usePlddtColoring !== false ? 'plddt' : 'chain';
            
            // Add model to viewer
            this.viewer.addModel(pdbText, "pdb");
            
            // Detect available chains in the structure
            const availableChains = this._detectChains();
            
            // Check if we should use pLDDT coloring
            const usePlddt = options.usePlddtColoring !== false;
            
            if (usePlddt) {
                // Color by pLDDT (B-factor column) - applies to all chains
                this._colorByPlddt();
            } else {
                // Color by chain - detect and color all available chains
                // For LaProteina files (only chain A), display it normally
                // For AF2 files (chains A and B), use red/blue
                if (availableChains.length === 1 && availableChains[0] === 'A') {
                    // LaProteina format: single chain A (contains both target and binder)
                    this.viewer.setStyle({chain: "A"}, {
                        cartoon: { color: "spectrum" }  // Use spectrum to see all parts
                    });
                } else {
                    // AF2 format: multiple chains
                    // Color chain A (target) red, chain B (binder) blue, others green
                    const chainColors = {
                        'A': 'red',
                        'B': 'blue'
                    };
                    availableChains.forEach(chain => {
                        const color = chainColors[chain] || 'green';
                        this.viewer.setStyle({chain: chain}, {
                            cartoon: { color: color }
                        });
                    });
                }
            }
            
            // Render and zoom to fit
            this.viewer.zoomTo();
            this.viewer.render();
            
            // Reapply highlight if it was active
            if (this.isBinderHighlighted) {
                this._applyHighlightStyles();
            }
            
            return { usePlddt, availableChains };
        } catch (error) {
            console.error("Error loading structure:", error);
            console.error("Error details:", error.stack);
            return null;
        }
    }
    
    /**
     * Detect which chains are present in the current model.
     * @private
     */
    _detectChains() {
        if (!this.viewer) return [];
        
        const chains = new Set();
        
        try {
            // Try to get chains from viewer's atom data
            const atoms = this.viewer.selectedAtoms({});
            if (atoms && atoms.length > 0) {
                atoms.forEach(atom => {
                    if (atom.chain) {
                        chains.add(atom.chain);
                    }
                });
            }
        } catch (e) {
            // If detection fails, try parsing from models
            try {
                const models = this.viewer.getModel();
                if (models && models.length > 0) {
                    models.forEach(model => {
                        if (model.selectedAtoms) {
                            model.selectedAtoms.forEach(atom => {
                                if (atom.chain) {
                                    chains.add(atom.chain);
                                }
                            });
                        }
                    });
                }
            } catch (e2) {
                // If all detection fails, return empty array and let caller handle it
                console.warn("Could not detect chains, assuming standard A and B");
                return ['A', 'B'];
            }
        }
        
        // If we found chains, return them sorted
        if (chains.size > 0) {
            return Array.from(chains).sort();
        }
        
        // Fallback: assume standard chains A and B (for AF2 files)
        return ['A', 'B'];
    }
    
    _colorByPlddt() {
        if (!this.viewer) return;
        
        // pLDDT color scheme (using 3Dmol.js colorfunc):
        // Very high confidence (90-100): blue (#0066ff)
        // Confident (70-90): light blue (#66b3ff)
        // Low (50-70): yellow (#ffcc00)
        // Very low (0-50): red (#ff0000)
        
        // Use 3Dmol's colorfunc to color by B-factor (pLDDT)
        this.viewer.setStyle({}, {
            cartoon: {
                colorfunc: (atom) => {
                    const plddt = atom.b || 0;
                    if (plddt >= 90) return '#0066ff'; // Blue - very high
                    if (plddt >= 70) return '#66b3ff'; // Light blue - confident
                    if (plddt >= 50) return '#ffcc00'; // Yellow - low
                    return '#ff0000'; // Red - very low
                }
            }
        });
    }
    
    colorByChain(component, chainColors = {}) {
        if (!this.viewer) return;
        
        // Apply chain colors
        for (const [chain, color] of Object.entries(chainColors)) {
            this.viewer.setStyle({chain: chain}, {
                cartoon: { color: color }
            });
        }
        this.viewer.render();
    }
    
    highlightResidues(component, residueIds, chainId = "A") {
        if (!this.viewer) return;
        
        // Highlight specific residues
        residueIds.forEach(resId => {
            this.viewer.setStyle({chain: chainId, resi: resId}, {
                stick: { color: "yellow", radius: 0.3 }
            });
        });
        this.viewer.render();
    }
    
    setRepresentation(type) {
        if (!this.viewer) return;
        
        // Store current representation
        this.currentRepresentation = type;
        
        const styles = {
            cartoon: { cartoon: { color: "spectrum" } },
            surface: { surface: { opacity: 0.7 } },
            stick: { stick: { radius: 0.2 } },
            sphere: { sphere: { scale: 0.5 } }
        };
        
        const style = styles[type] || styles.cartoon;
        this.viewer.setStyle({}, style);
        
        // Reapply coloring mode after changing representation
        if (this.currentColoringMode === 'plddt') {
            this._colorByPlddt();
        } else {
            this.colorByChain();
        }
        
        this.viewer.render();
    }
    
    toggleChain(chainId, visible) {
        if (!this.viewer) return;
        
        if (visible) {
            // Show chain - restore original coloring based on current mode
            if (this.currentColoringMode === 'plddt') {
                // Re-apply pLDDT coloring for this chain
                this.viewer.setStyle({chain: chainId}, {
                    cartoon: {
                        colorfunc: (atom) => {
                            const plddt = atom.b || 0;
                            if (plddt >= 90) return '#0066ff';
                            if (plddt >= 70) return '#66b3ff';
                            if (plddt >= 50) return '#ffcc00';
                            return '#ff0000';
                        },
                        opacity: 1.0
                    }
                });
            } else {
                // Default chain colors
                if (chainId === "A") {
                    this.viewer.setStyle({chain: chainId}, {
                        cartoon: { color: "red", opacity: 1.0 }
                    });
                } else if (chainId === "B") {
                    this.viewer.setStyle({chain: chainId}, {
                        cartoon: { color: "blue", opacity: 1.0 }
                    });
                }
            }
        } else {
            // Hide chain - make it nearly transparent
            this.viewer.setStyle({chain: chainId}, {
                cartoon: { opacity: 0.05 },
                stick: { opacity: 0.05 },
                sphere: { opacity: 0.05 }
            });
        }
        this.viewer.render();
    }
    
    highlightBinder() {
        if (!this.viewer) return;
        
        // Toggle highlight state
        if (this.isBinderHighlighted) {
            // Revert to original styles
            this._revertHighlight();
            this.isBinderHighlighted = false;
            return;
        }
        
        // Store original styles before highlighting
        this._storeOriginalStyles();
        
        // Apply styles
        this._applyHighlightStyles();
        this.isBinderHighlighted = true;
    }

    /**
     * Apply highlight styles based on current state.
     * @private
     */
    _applyHighlightStyles() {
        if (!this.viewer) return;

        // Detect available chains
        const availableChains = this._detectChains();
        
        // For LaProteina files (only chain A), just ensure it's visible with good coloring
        if (availableChains.length === 1 && availableChains[0] === 'A') {
            // LaProteina format: single chain A contains both target and binder
            // Just ensure it's fully visible with spectrum coloring to see all parts
            if (this.currentColoringMode === 'plddt') {
                this._colorByPlddt();
            } else {
                this.viewer.setStyle({chain: "A"}, {
                    cartoon: { color: "spectrum", opacity: 1.0 }
                });
            }
            this.viewer.render();
            return;
        }
        
        // For AF2 files (chains A and B), highlight binder and dim target
        // Build style object based on current representation
        const targetStyle = {};
        const binderStyle = {};
        
        // Apply styles based on current representation type
        // Keep target's original coloring but optionally reduce opacity
        // Change binder to bright pink (#ff69b4)
        if (this.currentRepresentation === 'cartoon') {
            // Preserve target's original coloring (pLDDT or chain color) but reduce opacity
            if (this.currentColoringMode === 'plddt') {
                targetStyle.cartoon = {
                    colorfunc: (atom) => {
                        const plddt = atom.b || 0;
                        if (plddt >= 90) return '#0066ff';
                        if (plddt >= 70) return '#66b3ff';
                        if (plddt >= 50) return '#ffcc00';
                        return '#ff0000';
                    },
                    opacity: 0.6  // Dim but keep original colors
                };
            } else {
                targetStyle.cartoon = { opacity: 0.6, color: "red" };  // Keep chain color but dim
            }
            binderStyle.cartoon = { opacity: 1.0, color: "#ff69b4" }; // Bright pink
        } else if (this.currentRepresentation === 'surface') {
            // For surface, keep original color but reduce opacity
            targetStyle.surface = { opacity: 0.5 };
            binderStyle.surface = { opacity: 0.9, color: "#ff69b4" }; // Bright pink
        } else if (this.currentRepresentation === 'stick') {
            // For stick, keep original color but reduce opacity
            targetStyle.stick = { opacity: 0.6, radius: 0.2 };
            binderStyle.stick = { opacity: 1.0, color: "#ff69b4", radius: 0.2 }; // Bright pink
        } else if (this.currentRepresentation === 'sphere') {
            // For sphere, keep original color but reduce opacity
            targetStyle.sphere = { opacity: 0.6, scale: 0.5 };
            binderStyle.sphere = { opacity: 1.0, color: "#ff69b4", scale: 0.5 }; // Bright pink
        } else {
            // Default to cartoon if unknown
            if (this.currentColoringMode === 'plddt') {
                targetStyle.cartoon = {
                    colorfunc: (atom) => {
                        const plddt = atom.b || 0;
                        if (plddt >= 90) return '#0066ff';
                        if (plddt >= 70) return '#66b3ff';
                        if (plddt >= 50) return '#ffcc00';
                        return '#ff0000';
                    },
                    opacity: 0.4
                };
            } else {
                targetStyle.cartoon = { opacity: 0.4, color: "red" };
            }
            binderStyle.cartoon = { opacity: 1.0, color: "#ff69b4" }; // Bright pink
        }
        
        // Dim target (chain A) but keep its original coloring
        this.viewer.setStyle({chain: "A"}, targetStyle);
        
        // Make binder (chain B) bright pink - only change color, keep representation
        this.viewer.setStyle({chain: "B"}, binderStyle);
        
        // Don't zoom - just highlight the color
        this.viewer.render();
    }
    
    _storeOriginalStyles() {
        // Store current styles for revert
        this.originalStyles = {
            coloringMode: this.currentColoringMode,
            representation: this.currentRepresentation
        };
    }
    
    _revertHighlight() {
        if (!this.viewer) return;
        
        // Revert to original coloring and representation
        if (this.originalStyles.coloringMode === 'plddt') {
            this._colorByPlddt();
        } else {
            // Revert to chain colors
            const availableChains = this._detectChains();
            availableChains.forEach(chain => {
                if (chain === 'A') {
                    this.viewer.setStyle({chain: chain}, {
                        cartoon: { color: "red", opacity: 1.0 }
                    });
                } else if (chain === 'B') {
                    this.viewer.setStyle({chain: chain}, {
                        cartoon: { color: "blue", opacity: 1.0 }
                    });
                }
            });
        }
        
        // Reapply representation
        this.setRepresentation(this.originalStyles.representation || 'cartoon');
        this.viewer.render();
    }
    
    resetView() {
        if (this.viewer) {
            this.viewer.zoomTo();
            this.viewer.render();
        }
    }
    
    clear() {
        if (this.viewer) {
            this.viewer.clear();
            this.viewer.render();
        }
    }
    
    destroy() {
        if (this.viewer) {
            this.viewer.clear();
            this.viewer = null;
        }
    }

    /**
     * Extract amino acid sequence from PDB text for a specific chain.
     * @param {string} chainId - Chain identifier (e.g., 'A', 'B')
     * @returns {string} Amino acid sequence (single letter codes)
     */
    extractSequence(chainId = 'A') {
        if (!this.currentPdbText) {
            console.warn("No PDB text available for sequence extraction");
            return null;
        }
        
        const lines = this.currentPdbText.split('\n');
        const sequence = [];
        const seenResidues = new Set(); // Track (chain, resnum) to avoid duplicates
        
        for (const line of lines) {
            if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
                const chain = line.charAt(21);
                const resName = line.substring(17, 20).trim();
                const resNum = line.substring(22, 26).trim();
                
                if (chain === chainId) {
                    const key = `${chain}_${resNum}`;
                    if (!seenResidues.has(key)) {
                        seenResidues.add(key);
                        // Convert 3-letter code to 1-letter code
                        const aa = this._threeToOneLetter(resName);
                        if (aa) {
                            sequence.push(aa);
                        }
                    }
                }
            }
        }
        
        return sequence.join('');
    }

    /**
     * Convert 3-letter amino acid code to 1-letter code.
     * @private
     */
    _threeToOneLetter(resName) {
        const aaMap = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'SEC': 'U', 'PYL': 'O' // Selenocysteine and Pyrrolysine
        };
        
        return aaMap[resName.toUpperCase()] || null;
    }
}

// Global viewer instance
let viewer = null;
let viewerReady = false;

// Wait for 3Dmol.js to be available
function waitFor3Dmol(callback, maxAttempts = 50) {
    let attempts = 0;
    
    function check() {
        attempts++;
        
        // Check for $3Dmol
        if (typeof $3Dmol !== 'undefined') {
            console.log("3Dmol.js found, initializing viewer...");
            callback();
        } else if (attempts < maxAttempts) {
            setTimeout(check, 100);
        } else {
            console.error("3Dmol.js failed to load after", maxAttempts, "attempts");
            const container = document.getElementById("ngl-viewer");
            if (container) {
                container.innerHTML = "<div style='padding: 2rem; text-align: center; color: #e74c3c;'>3Dmol.js Viewer library failed to load. Please check your internet connection and reload the page.</div>";
            }
        }
    }
    
    check();
}

// Initialize viewer when DOM and 3Dmol.js are ready
function initializeViewer() {
    // Check if container exists
    const container = document.getElementById("ngl-viewer");
    if (!container) {
        console.error("Viewer container 'ngl-viewer' not found");
        return false;
    }
    
    // Initialize viewer
    try {
        viewer = new ProteinViewer("ngl-viewer");
        viewerReady = true;
        console.log("Viewer initialized successfully");
        return true;
    } catch (error) {
        console.error("Error initializing viewer:", error);
        console.error("Error stack:", error.stack);
        container.innerHTML = `<div style='padding: 2rem; text-align: center; color: #e74c3c;'>Error initializing viewer: ${error.message}</div>`;
        return false;
    }
}

// Start waiting for 3Dmol.js when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        waitFor3Dmol(initializeViewer);
    });
} else {
    // DOM already ready, start waiting immediately
    waitFor3Dmol(initializeViewer);
}
