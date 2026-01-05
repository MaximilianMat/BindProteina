/**
 * Animation controller for evolution progress and conformational changes.
 */

class AnimationController {
    constructor(viewer) {
        this.viewer = viewer;
        this.isPlaying = false;
        this.currentFrame = 0;
        this.frames = [];
        this.speed = 1.0;
        this.animationId = null;
        this.setupSlider();
    }
    
    setupSlider() {
        const slider = document.getElementById("frameSlider");
        if (!slider) {
            // Retry if slider not ready yet
            setTimeout(() => this.setupSlider(), 100);
            return;
        }
        
        // Allow manual control - when user drags slider, jump to that frame
        slider.addEventListener("input", (e) => {
            const frameIndex = parseInt(e.target.value);
            this.goToFrame(frameIndex);
            // Pause auto-play when manually controlling
            if (this.isPlaying) {
                this.pause();
            }
        });
        
        slider.addEventListener("change", (e) => {
            // Ensure frame is loaded when slider is released
            const frameIndex = parseInt(e.target.value);
            this.goToFrame(frameIndex);
        });
    }
    
    /**
     * Create evolution progress animation.
     * @param {Array} designs - Array of designs from different rounds
     * @param {Function} loadCallback - Callback to load structure for each frame
     */
    async createEvolutionAnimation(designs, loadCallback) {
        this.frames = [];
        
        // Sort designs by round number
        const sortedDesigns = [...designs].sort((a, b) => {
            return (a.round_number || 0) - (b.round_number || 0);
        });
        
        // Build frame list with PDB paths
        for (const design of sortedDesigns) {
            // Prefer AF2 prediction over relaxed for animation
            const pdbPath = design.af2_pdb_path || design.relaxed_pdb_path;
            if (pdbPath) {
                this.frames.push({
                    design: design,
                    pdbPath: pdbPath,
                    pdbText: null, // Will be populated during preload
                    loadCallback: loadCallback,
                    roundNumber: design.round_number || 0
                });
            }
        }
        
        // Preload all PDB files
        console.log(`Preloading ${this.frames.length} PDB files...`);
        const preloadPromises = this.frames.map(async (frame, index) => {
            try {
                const pdbUrl = dataParser.getPdbUrl(frame.pdbPath);
                const response = await fetch(pdbUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch PDB: ${response.status}`);
                }
                frame.pdbText = await response.text();
                console.log(`Preloaded frame ${index + 1}/${this.frames.length}`);
            } catch (error) {
                console.error(`Error preloading frame ${index + 1}:`, error);
                frame.pdbText = null; // Mark as failed
            }
        });
        
        await Promise.all(preloadPromises);
        console.log("All PDB files preloaded!");
        
        this.currentFrame = 0;
        this.updateSlider();
    }
    
    updateSlider() {
        const slider = document.getElementById("frameSlider");
        const minLabel = document.getElementById("frameSliderMin");
        const maxLabel = document.getElementById("frameSliderMax");
        
        if (slider) {
            slider.max = Math.max(0, this.frames.length - 1);
            // Only update slider value if not currently being dragged
            // Check if slider is focused (being actively dragged)
            if (document.activeElement !== slider) {
                slider.value = this.currentFrame;
            }
            slider.disabled = this.frames.length === 0;
        }
        
        if (minLabel) {
            minLabel.textContent = "1";
        }
        
        if (maxLabel) {
            maxLabel.textContent = String(this.frames.length);
        }
    }
    
    /**
     * Create conformational change animation between two structures.
     * @param {string} pdbPath1 - Path to first structure
     * @param {string} pdbPath2 - Path to second structure
     * @param {number} numFrames - Number of interpolation frames
     */
    async createMorphAnimation(pdbPath1, pdbPath2, numFrames = 30) {
        // This would require structure interpolation
        // For now, we'll just switch between structures
        this.frames = [
            { pdbPath: pdbPath1 },
            { pdbPath: pdbPath2 }
        ];
        this.currentFrame = 0;
    }
    
    play() {
        if (this.frames.length === 0) {
            console.warn("No frames to animate");
            return;
        }
        
        if (this.isPlaying) {
            return;
        }
        
        this.isPlaying = true;
        this.animate();
    }
    
    pause() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    stop() {
        this.pause();
        this.currentFrame = 0;
        this.updateSlider();
    }
    
    setSpeed(speed) {
        this.speed = Math.max(0.1, Math.min(3.0, speed));
    }
    
    async animate() {
        if (!this.isPlaying || this.frames.length === 0) {
            return;
        }
        
        const frame = this.frames[this.currentFrame];
        
            try {
                // Load structure for current frame using preloaded data
                if (frame.pdbText) {
                    // Use preloaded PDB text for instant loading
                    await viewer.loadStructureFromText(frame.pdbText, {
                        usePlddtColoring: true
                    });
                    
                    // Call callback for UI updates (frame indicator, etc.)
                    if (frame.loadCallback) {
                        await frame.loadCallback(frame.pdbPath, frame.design, frame.pdbText);
                    }
                } else if (frame.loadCallback) {
                    // Fallback to callback if preload failed
                    await frame.loadCallback(frame.pdbPath, frame.design, null);
                } else if (frame.pdbPath) {
                    // Fallback to URL loading if preload failed
                    const pdbUrl = dataParser.getPdbUrl(frame.pdbPath);
                    await viewer.loadStructure(pdbUrl, {
                        usePlddtColoring: true
                    });
                }
            
            // Update slider to show progress
            this.updateSlider();
            
            // Move to next frame
            this.currentFrame = (this.currentFrame + 1) % this.frames.length;
            
            // Schedule next frame
            const delay = 1000 / this.speed; // Base delay of 1 second per frame
            setTimeout(() => {
                if (this.isPlaying) {
                    this.animate();
                }
            }, delay);
        } catch (error) {
            console.error("Error in animation frame:", error);
            this.pause();
        }
    }
    
    updateTimeline() {
        // Timeline bar removed - using slider for progress display only
        // Update slider position
        this.updateSlider();
    }
    
    async goToFrame(frameIndex) {
        if (frameIndex < 0 || frameIndex >= this.frames.length) {
            return;
        }
        
        this.currentFrame = frameIndex;
        
        // Load the frame using preloaded data
        const frame = this.frames[this.currentFrame];
        if (frame) {
            try {
                if (frame.pdbText) {
                    // Use preloaded PDB text for instant loading
                    await viewer.loadStructureFromText(frame.pdbText, {
                        usePlddtColoring: true
                    });

                    // Call callback for UI updates (frame indicator, etc.)
                    if (frame.loadCallback) {
                        await frame.loadCallback(frame.pdbPath, frame.design, frame.pdbText);
                    }
                } else if (frame.loadCallback) {
                    // Fallback to callback if preload failed
                    await frame.loadCallback(frame.pdbPath, frame.design, null);
                } else if (frame.pdbPath) {
                    // Fallback to URL loading if preload failed
                    const pdbUrl = dataParser.getPdbUrl(frame.pdbPath);
                    await viewer.loadStructure(pdbUrl, {
                        usePlddtColoring: true
                    });
                }
                
                this.updateSlider();
            } catch (error) {
                console.error("Error loading frame:", error);
            }
        }
    }
}

// Global animation controller
let animationController = null;

// Initialize when viewer is ready
document.addEventListener("DOMContentLoaded", () => {
    // Will be initialized after viewer is ready
});

