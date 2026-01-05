/**
 * UI controller and event handlers.
 */

let currentRunData = null;
let currentRoundData = null;
let currentDesignData = null;
let allDesigns = []; // Store all designs for animation (all rounds)
let allSeeds = new Map(); // Store designs grouped by seed ID
let currentPdbVersion = 'af2'; // Track current PDB version: 'af2' or 'relaxed'
let preloadedPdbCache = new Map(); // Cache for preloaded PDB files: design_name -> {af2: text, relaxed: text}
let currentStatsRoundIndex = 0; // Current round index in stats view
let allRoundsStats = new Map(); // Cache for round statistics
let isBinderHighlighted = false; // Track highlight state globally

/**
 * Show a toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) {
        console.log(`[${type.toUpperCase()}] ${message}`);
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Add icon based on type
    let icon = 'ℹ️';
    if (type === 'success') icon = '✓';
    else if (type === 'error') icon = '✗';
    else if (type === 'warning') icon = '⚠';
    
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    
    container.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
    
    return toast;
}

// Scoring Table State
let scoringTableData = null;
let scoringTableSortColumn = null;
let scoringTableSortDirection = 'desc';
let scoringTableFilterRound = 'all';
let scoringTableSearchTerm = '';

// Initialize UI when DOM is ready
document.addEventListener("DOMContentLoaded", async () => {
    // Check if backend is available
    const isHealthy = await dataParser.checkHealth();
    if (!isHealthy) {
        alert("Warning: Backend server is not responding. Please start the server first.");
    }
    
    await initializeUI();
    setupEventListeners();
});

async function initializeUI() {
    // Load evolution runs
    const runs = await dataParser.listRuns();
    const runSelector = document.getElementById("runSelector");
    
    runs.forEach(run => {
        const option = document.createElement("option");
        option.value = run;
        option.textContent = run;
        runSelector.appendChild(option);
    });
}

function setupEventListeners() {
    // Load run button
    document.getElementById("loadRunBtn").addEventListener("click", loadRun);
    
    // Save notes button
    const saveNotesBtn = document.getElementById("saveNotesBtn");
    if (saveNotesBtn) {
        saveNotesBtn.addEventListener("click", saveNotes);
    }
    
    // Design search
    const designSearch = document.getElementById("designSearch");
    if (designSearch) {
        designSearch.addEventListener("input", (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const designItems = document.querySelectorAll(".design-item");
            designItems.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    item.style.display = "flex";
                } else {
                    item.style.display = "none";
                }
            });
        });
    }

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {
        // Skip if user is typing in an input
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") {
            return;
        }

        const activeItem = document.querySelector(".design-item.active");
        if (!activeItem) return;

        let nextItem = null;
        if (e.key === "ArrowDown") {
            nextItem = activeItem.nextElementSibling;
            while (nextItem && nextItem.style.display === "none") {
                nextItem = nextItem.nextElementSibling;
            }
        } else if (e.key === "ArrowUp") {
            nextItem = activeItem.previousElementSibling;
            while (nextItem && nextItem.style.display === "none") {
                nextItem = nextItem.previousElementSibling;
            }
        }

        if (nextItem && nextItem.classList.contains("design-item")) {
            e.preventDefault();
            nextItem.click();
            nextItem.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
    });
    
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tabName = e.target.dataset.tab;
            switchTab(tabName);
        });
    });
    
    // Scoring table controls
    const refreshBtn = document.getElementById("refreshScoringTableBtn");
    const roundFilter = document.getElementById("scoringRoundFilter");
    
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            if (currentRunData) {
                loadScoringTable();
            }
        });
    }
    
    if (roundFilter) {
        roundFilter.addEventListener("change", (e) => {
            scoringTableFilterRound = e.target.value;
            if (scoringTableData) {
                renderScoringTable();
            }
        });
    }
    
    // Round navigation in stats view
    document.getElementById("prevRoundBtn").addEventListener("click", () => {
        navigateRound(-1);
    });
    
    document.getElementById("nextRoundBtn").addEventListener("click", () => {
        navigateRound(1);
    });
    
    // Viewer controls
    document.getElementById("resetViewBtn").addEventListener("click", () => {
        if (viewer) viewer.resetView();
    });
    
    // Track chain visibility state
    let targetVisible = true;
    let binderVisible = true;
    
    document.getElementById("toggleTargetBtn").addEventListener("click", () => {
        if (viewer && viewerReady) {
            targetVisible = !targetVisible;
            viewer.toggleChain("A", targetVisible);
        }
    });
    
    document.getElementById("toggleBinderBtn").addEventListener("click", () => {
        if (viewer && viewerReady) {
            binderVisible = !binderVisible;
            viewer.toggleChain("B", binderVisible);
        }
    });
    
    document.getElementById("highlightBinderBtn").addEventListener("click", () => {
        if (viewer && viewerReady) {
            viewer.highlightBinder();
            // Update global state and button text
            isBinderHighlighted = viewer.isBinderHighlighted;
            updateHighlightButtonState();
        }
    });
    
    // Copy sequence buttons
    document.getElementById("copyTargetSeqBtn").addEventListener("click", async () => {
        if (!viewer || !viewerReady) {
            showToast("Viewer is not ready", "error");
            return;
        }
        
        if (!viewer.currentPdbText) {
            showToast("No structure loaded", "error");
            return;
        }
        
        try {
            const sequence = viewer.extractSequence('A');
            if (!sequence || sequence.length === 0) {
                showToast("Could not extract target sequence", "error");
                return;
            }
            
            await navigator.clipboard.writeText(sequence);
            showToast(`✅ Copied target sequence (${sequence.length} residues)`, "success");
        } catch (err) {
            console.error('Failed to copy target sequence:', err);
            // Fallback for older browsers
            try {
                const sequence = viewer.extractSequence('A');
                if (sequence) {
                    const textArea = document.createElement('textarea');
                    textArea.value = sequence;
                    textArea.style.position = 'fixed';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    showToast(`Copied target sequence (${sequence.length} residues)`, "success");
                } else {
                    showToast("Could not extract target sequence", "error");
                }
            } catch (fallbackErr) {
                showToast(`Failed to copy: ${fallbackErr.message}`, "error");
            }
        }
    });
    
    document.getElementById("copyBinderSeqBtn").addEventListener("click", async () => {
        if (!viewer || !viewerReady) {
            showToast("Viewer is not ready", "error");
            return;
        }
        
        if (!viewer.currentPdbText) {
            showToast("No structure loaded", "error");
            return;
        }
        
        try {
            const availableChains = viewer._detectChains();
            let sequence = null;
            
            if (availableChains.includes('B')) {
                // Standard AF2 format: binder is chain B
                sequence = viewer.extractSequence('B');
            } else if (availableChains.includes('A')) {
                // LaProteina format: binder is part of chain A
                sequence = viewer.extractSequence('A');
                if (sequence && currentDesignData && currentDesignData.metrics) {
                    // Try to get target length from metrics if available
                    const targetLength = currentDesignData.metrics.target_length || currentDesignData.metrics.targetLength;
                    if (targetLength && sequence.length > targetLength) {
                        // Extract only binder portion (residues after target)
                        sequence = sequence.substring(targetLength);
                    }
                }
            }
            
            if (!sequence || sequence.length === 0) {
                showToast("Could not extract binder sequence", "error");
                return;
            }
            
            await navigator.clipboard.writeText(sequence);
            showToast(`✅ Copied binder sequence (${sequence.length} residues)`, "success");
        } catch (err) {
            console.error('Failed to copy binder sequence:', err);
            // Fallback for older browsers
            try {
                const availableChains = viewer._detectChains();
                let sequence = null;
                
                if (availableChains.includes('B')) {
                    sequence = viewer.extractSequence('B');
                } else if (availableChains.includes('A')) {
                    sequence = viewer.extractSequence('A');
                    if (sequence && currentDesignData && currentDesignData.metrics) {
                        const targetLength = currentDesignData.metrics.target_length || currentDesignData.metrics.targetLength;
                        if (targetLength && sequence.length > targetLength) {
                            sequence = sequence.substring(targetLength);
                        }
                    }
                }
                
                if (sequence) {
                    const textArea = document.createElement('textarea');
                    textArea.value = sequence;
                    textArea.style.position = 'fixed';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    showToast(`Copied binder sequence (${sequence.length} residues)`, "success");
                } else {
                    showToast("Could not extract binder sequence", "error");
                }
            } catch (fallbackErr) {
                showToast(`Failed to copy: ${fallbackErr.message}`, "error");
            }
        }
    });
    
    document.getElementById("pdbVersionSelect").addEventListener("change", (e) => {
        if (!currentDesignData) {
            return;
        }
        if (!viewer || !viewerReady) {
            showToast("Viewer not ready", "error");
            return;
        }
        const version = e.target.value;
        currentPdbVersion = version; // Update global state
        loadDesignVersion(currentDesignData, version);
    });
    
    document.getElementById("representationSelect").addEventListener("change", (e) => {
        if (viewer && viewerReady) {
            viewer.setRepresentation(e.target.value);
        }
    });
    
    // Animation controls
    document.getElementById("playBtn").addEventListener("click", () => {
        if (animationController) animationController.play();
    });
    
    document.getElementById("pauseBtn").addEventListener("click", () => {
        if (animationController) animationController.pause();
    });
    
    document.getElementById("stopBtn").addEventListener("click", () => {
        if (animationController) animationController.stop();
    });
    
    document.getElementById("speedSlider").addEventListener("input", (e) => {
        const speed = parseFloat(e.target.value);
        document.getElementById("speedLabel").textContent = `Speed: ${speed}x`;
        if (animationController) {
            animationController.setSpeed(speed);
        }
    });
    
    // Create animation button
    document.getElementById("createAnimationBtn").addEventListener("click", createEvolutionAnimation);
    
    // Seed animation button
    document.getElementById("createSeedAnimationBtn").addEventListener("click", createSeedAnimation);
    
    // Seed selector change
    document.getElementById("seedSelector").addEventListener("change", (e) => {
        const seedBtn = document.getElementById("createSeedAnimationBtn");
        seedBtn.style.display = e.target.value ? "block" : "none";
    });
}

function updateSeedSelector() {
    const seedSelector = document.getElementById("seedSelector");
    const seedAnimationBtn = document.getElementById("createSeedAnimationBtn");
    
    if (allSeeds.size === 0) {
        seedSelector.style.display = "none";
        seedAnimationBtn.style.display = "none";
        return;
    }
    
    // Populate seed selector
    seedSelector.innerHTML = '<option value="">Select seed for animation...</option>';
    
    const sortedSeeds = Array.from(allSeeds.keys()).sort();
    sortedSeeds.forEach(seedId => {
        const seedDesigns = allSeeds.get(seedId);
        const rounds = new Set(seedDesigns.map(d => d.round_number));
        const option = document.createElement("option");
        option.value = seedId;
        option.textContent = `${seedId} (${seedDesigns.length} designs, ${rounds.size} rounds)`;
        seedSelector.appendChild(option);
    });
    
    seedSelector.style.display = "block";
}

async function createSeedAnimation() {
    const seedSelector = document.getElementById("seedSelector");
    const selectedSeed = seedSelector.value;
    
    if (!selectedSeed || !allSeeds.has(selectedSeed)) {
        alert("Please select a seed");
        return;
    }
    
    if (!animationController) {
        alert("Animation controller not initialized");
        return;
    }
    
    const seedDesigns = allSeeds.get(selectedSeed);
    
    if (seedDesigns.length === 0) {
        alert("No designs found for this seed");
        return;
    }
    
    // Sort by round number, then by i_pTM within each round
    seedDesigns.sort((a, b) => {
        const roundDiff = (a.round_number || 0) - (b.round_number || 0);
        if (roundDiff !== 0) return roundDiff;
        const a_ptm = a.metrics.i_pTM || 0;
        const b_ptm = b.metrics.i_pTM || 0;
        return b_ptm - a_ptm;
    });
    
    // Show animation panel
    const animationPanel = document.getElementById("animationPanel");
    animationPanel.style.display = "block";
    
    // Update frame indicator and slider
    const frameIndicator = document.getElementById("frameIndicator");
    if (frameIndicator) {
        const rounds = new Set(seedDesigns.map(d => d.round_number));
        frameIndicator.textContent = `Frame 0 / ${seedDesigns.length} (Seed: ${selectedSeed}, ${rounds.size} rounds)`;
    }
    
    // Update slider after animation is created
    setTimeout(() => {
        if (animationController) {
            animationController.updateSlider();
        }
    }, 100);
    
    // Create animation
    await animationController.createEvolutionAnimation(
        seedDesigns,
        async (pdbPath, design, pdbText) => {
            if (viewer && viewerReady) {
                // Use preloaded PDB text if available, otherwise fetch
                if (pdbText) {
                    await viewer.loadStructureFromText(pdbText, {
                        usePlddtColoring: true
                    });
                } else {
                    const pdbUrl = dataParser.getPdbUrl(pdbPath);
                    await viewer.loadStructure(pdbUrl, {
                        usePlddtColoring: true
                    });
                }
                
                // Update frame indicator
                const currentFrame = animationController.currentFrame + 1;
                const totalFrames = animationController.frames.length;
                const i_pTM = design.metrics.i_pTM ? design.metrics.i_pTM.toFixed(2) : "N/A";
                if (frameIndicator) {
                    frameIndicator.textContent = `Frame ${currentFrame} / ${totalFrames} - Seed: ${selectedSeed} - Round ${design.round_number} - ${design.design_name} (i_pTM: ${i_pTM})`;
                }
            }
        }
    );
    
    // Start playing
    animationController.play();
}

async function createEvolutionAnimation() {
    if (!animationController || allDesigns.length === 0) {
        alert("No designs available for animation");
        return;
    }
    
    // Get top 5 designs from each round
    const roundsMap = new Map();
    allDesigns.forEach(design => {
        const roundNum = design.round_number || 0;
        if (!roundsMap.has(roundNum)) {
            roundsMap.set(roundNum, []);
        }
        roundsMap.get(roundNum).push(design);
    });
    
    // Sort designs by i_pTM within each round and take top 5
    const topDesignsPerRound = [];
    const sortedRounds = Array.from(roundsMap.keys()).sort();
    
    sortedRounds.forEach(roundNum => {
        const designs = roundsMap.get(roundNum);
        designs.sort((a, b) => {
            const a_ptm = a.metrics.i_pTM || 0;
            const b_ptm = b.metrics.i_pTM || 0;
            return b_ptm - a_ptm;
        });
        // Take top 5 designs from this round
        const top5 = designs.slice(0, 5);
        topDesignsPerRound.push(...top5);
    });
    
    if (topDesignsPerRound.length === 0) {
        alert("No designs available for animation");
        return;
    }
    
    // Show animation panel
    const animationPanel = document.getElementById("animationPanel");
    animationPanel.style.display = "block";
    
    // Update frame indicator and slider
    const frameIndicator = document.getElementById("frameIndicator");
    if (frameIndicator) {
        frameIndicator.textContent = `Frame 0 / ${topDesignsPerRound.length} (${sortedRounds.length} rounds, top 5 per round)`;
    }
    
    // Update slider after animation is created
    setTimeout(() => {
        if (animationController) {
            animationController.updateSlider();
        }
    }, 100);
    
    // Create animation
    await animationController.createEvolutionAnimation(
        topDesignsPerRound,
        async (pdbPath, design, pdbText) => {
            if (viewer && viewerReady) {
                // Use preloaded PDB text if available, otherwise fetch
                if (pdbText) {
                    await viewer.loadStructureFromText(pdbText, {
                        usePlddtColoring: true
                    });
                } else {
                    const pdbUrl = dataParser.getPdbUrl(pdbPath);
                    await viewer.loadStructure(pdbUrl, {
                        usePlddtColoring: true
                    });
                }
                
                // Update frame indicator with round and design info
                const currentFrame = animationController.currentFrame + 1;
                const totalFrames = animationController.frames.length;
                const i_pTM = design.metrics.i_pTM ? design.metrics.i_pTM.toFixed(2) : "N/A";
                if (frameIndicator) {
                    frameIndicator.textContent = `Frame ${currentFrame} / ${totalFrames} - Round ${design.round_number} - ${design.design_name} (i_pTM: ${i_pTM})`;
                }
            }
        }
    );
    
    // Start playing
    animationController.play();
}

async function loadRun() {
    const runSelector = document.getElementById("runSelector");
    const runName = runSelector.value;
    
    if (!runName) {
        alert("Please select an evolution run");
        return;
    }
    
    // Show loading indicator
    const loadBtn = document.getElementById("loadRunBtn");
    const originalText = loadBtn.textContent;
    loadBtn.textContent = "Loading...";
    loadBtn.disabled = true;
    
    try {
        // 1. Get basic run data (now instant summary in backend)
        currentRunData = await dataParser.getRun(runName);
        if (!currentRunData || !currentRunData.rounds) {
            throw new Error("Invalid run data received");
        }
        
        // 2. Display rounds immediately (UI becomes responsive)
        displayRounds(currentRunData.rounds);
        
        // 2.5 Load notes for the run
        const notes = await dataParser.getRunNotes(runName);
        const notesTextarea = document.getElementById("runNotesTextarea");
        if (notesTextarea) {
            notesTextarea.value = notes;
        }
        
        // Auto-select first round
        if (currentRunData.rounds.length > 0) {
            const firstRound = currentRunData.rounds[0];
            const roundItems = document.querySelectorAll(".round-item");
            if (roundItems.length > 0) {
                roundItems[0].click();
            }
        }
        
        // Restore buttons state immediately after basic info is loaded
        loadBtn.textContent = originalText;
        loadBtn.disabled = false;
        
        // 3. Reset stats state
        currentStatsRoundIndex = 0;
        allRoundsStats.clear();
        allDesigns = [];
        allSeeds.clear();
        
        // 4. Load round designs (can take a bit longer for many designs)
        console.log("Loading designs from all rounds in background...");
        
        // We still want to load all designs for stats, but we do it more gracefully
        const loadPromises = currentRunData.rounds.map(async (round) => {
            try {
                const designs = await dataParser.getRoundDesigns(currentRunData.run_name, round.round_number);
                
                // Cache designs for stats
                const cacheKey = `${currentRunData.run_name}_round_${round.round_number}`;
                allRoundsStats.set(cacheKey, designs);
                
                // Process designs as they come in
                designs.forEach(design => {
                    allDesigns.push(design);
                    
                    // Extract seed ID
                    let seedId = null;
                    const patterns = [
                        /^([A-Z0-9]+)_/,
                        /^([A-Z]\d+[A-Z]\d+)/,
                        /seed[_-]?([A-Z0-9]+)/i,
                    ];
                    for (const pattern of patterns) {
                        const match = design.design_name.match(pattern);
                        if (match) {
                            seedId = match[1];
                            break;
                        }
                    }
                    if (seedId) {
                        if (!allSeeds.has(seedId)) {
                            allSeeds.set(seedId, []);
                        }
                        allSeeds.get(seedId).push(design);
                    }
                });
                
                // Update UI incrementally if needed (e.g. counts)
                updateRoundDisplayCounts();
                updateSeedSelector();
                
                return { round: round.round_number, designs, success: true };
            } catch (error) {
                console.error(`Error loading round ${round.round_number}:`, error);
                return { round: round.round_number, designs: [], success: false, error };
            }
        });
        
        // Wait for all rounds to finish loading to calculate global stats
        Promise.all(loadPromises).then(() => {
            console.log(`Finished loading all rounds. Total designs: ${allDesigns.length}`);
            
            // Calculate global ranges for consistent histogram scales
            if (allDesigns.length > 0) {
                roundStatistics.calculateGlobalRanges(allDesigns);
            }
            
            // Enable animation button if we have multiple rounds
            const animationBtn = document.getElementById("createAnimationBtn");
            if (currentRunData.rounds.length > 1 && allDesigns.length > 0) {
                animationBtn.style.display = "block";
            } else {
                animationBtn.style.display = "none";
            }
        });
        
    } catch (error) {
        console.error("Error loading run:", error);
        alert("Failed to load evolution run: " + error.message);
    } finally {
        loadBtn.textContent = originalText;
        loadBtn.disabled = false;
    }
}

function updateRoundDisplayCounts() {
    // Update round items to show actual loaded design counts
    const roundsList = document.getElementById("roundsList");
    const roundItems = roundsList.querySelectorAll(".round-item");
    
    roundItems.forEach(item => {
        const roundNum = parseInt(item.dataset.roundNum);
        if (!isNaN(roundNum)) {
            const cacheKey = `${currentRunData.run_name}_round_${roundNum}`;
            const roundDesigns = allRoundsStats.get(cacheKey);
            if (roundDesigns) {
                item.textContent = `Round ${roundNum} (${roundDesigns.length} designs loaded)`;
            }
        }
    });
}

function displayRounds(rounds) {
    const roundsList = document.getElementById("roundsList");
    roundsList.innerHTML = "";
    
    // Reset designs when loading a new run
    allDesigns = [];
    allSeeds.clear();
    
    if (!rounds || rounds.length === 0) {
        roundsList.innerHTML = "<div style='padding: 1rem; color: #999;'>No rounds found</div>";
        return;
    }
    
    console.log("Displaying rounds:", rounds);
    
    rounds.forEach(round => {
        const roundItem = document.createElement("div");
        roundItem.className = "round-item";
        const roundNum = round.round_number || round.roundNumber || "?";
        // Show loading state initially
        roundItem.textContent = `Round ${roundNum} (Loading...)`;
        roundItem.dataset.roundNum = roundNum;
        roundItem.addEventListener("click", () => {
            console.log("Round clicked:", round);
            // Remove active class from all items
            document.querySelectorAll(".round-item").forEach(item => {
                item.classList.remove("active");
            });
            roundItem.classList.add("active");
            
            loadRound(round);
        });
        roundsList.appendChild(roundItem);
    });
}

async function loadRound(round) {
    if (!currentRunData) {
        console.error("No current run data available");
        return;
    }
    
    console.log("Loading round:", round.round_number, "from run:", currentRunData.run_name);
    currentRoundData = round;
    
    // Show loading indicator
    const designsList = document.getElementById("designsList");
    designsList.innerHTML = "<div style='padding: 1rem; text-align: center; color: #666;'>Loading designs...</div>";
    
    try {
        // Load designs for this round
        const designs = await dataParser.getRoundDesigns(
            currentRunData.run_name,
            round.round_number
        );
        
        console.log(`Loaded ${designs.length} designs for round ${round.round_number}`);
        
        if (designs.length === 0) {
            designsList.innerHTML = "<div style='padding: 1rem; text-align: center; color: #999;'>No designs found for this round</div>";
            return;
        }
        
        // Designs are already loaded in allDesigns from loadRun()
        // Just filter for this round's designs for display
        const roundDesigns = allDesigns.filter(d => d.round_number === round.round_number);
        const designsToDisplay = roundDesigns.length > 0 ? roundDesigns : designs;
        
        // Display designs
        displayDesigns(designsToDisplay);
        
        // Auto-select first design
        if (designsToDisplay.length > 0) {
            const designItems = document.querySelectorAll(".design-item");
            if (designItems.length > 0) {
                designItems[0].click();
            }
        }
        
        // Don't show stats in sidebar anymore - only in stats tab
        
        // Animation button should already be visible if conditions are met
        // (set in loadRun())
    } catch (error) {
        console.error("Error loading round designs:", error);
        designsList.innerHTML = `<div style='padding: 1rem; text-align: center; color: #e74c3c;'>Error loading designs: ${error.message}</div>`;
    }
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const activeTab = document.getElementById(`${tabName}Tab`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    // If switching to statistics tab, load stats for current round
    if (tabName === 'statistics' && currentRunData && currentRunData.rounds.length > 0) {
        loadStatsForRound(currentStatsRoundIndex);
    }
    
    // If switching to scoring table tab, initialize if needed
    if (tabName === 'scoring' && currentRunData) {
        initializeScoringTable();
    }
}

function navigateRound(direction) {
    if (!currentRunData || !currentRunData.rounds.length) return;
    
    const newIndex = currentStatsRoundIndex + direction;
    if (newIndex >= 0 && newIndex < currentRunData.rounds.length) {
        currentStatsRoundIndex = newIndex;
        loadStatsForRound(currentStatsRoundIndex);
    }
}

async function loadStatsForRound(roundIndex) {
    if (!currentRunData || roundIndex < 0 || roundIndex >= currentRunData.rounds.length) {
        return;
    }
    
    const round = currentRunData.rounds[roundIndex];
    const roundNum = round.round_number;
    
    // Update round label
    document.getElementById("currentRoundLabel").textContent = `Round ${roundNum}`;
    
    // Update navigation buttons
    document.getElementById("prevRoundBtn").disabled = roundIndex === 0;
    document.getElementById("nextRoundBtn").disabled = roundIndex === currentRunData.rounds.length - 1;
    
    // Check cache first
    const cacheKey = `${currentRunData.run_name}_round_${roundNum}`;
    if (allRoundsStats.has(cacheKey)) {
        roundStatistics.displayStats(allRoundsStats.get(cacheKey), 'roundStats', true);
        return;
    }
    
    // Load designs for this round
    try {
        const designs = await dataParser.getRoundDesigns(
            currentRunData.run_name,
            roundNum
        );
        
        // Use designs from allDesigns if available (more complete)
        const roundDesigns = allDesigns.filter(d => d.round_number === roundNum);
        const designsToUse = roundDesigns.length > 0 ? roundDesigns : designs;
        
        // Cache the designs
        allRoundsStats.set(cacheKey, designsToUse);
        
        // Display statistics with global scale
        roundStatistics.displayStats(designsToUse, 'roundStats', true);
    } catch (error) {
        console.error("Error loading stats for round:", error);
        document.getElementById("roundStats").innerHTML = 
            `<div style='padding: 2rem; text-align: center; color: #e74c3c;'>Error loading statistics: ${error.message}</div>`;
    }
}

function displayDesigns(designs) {
    const designsList = document.getElementById("designsList");
    designsList.innerHTML = "";
    
    // Sort by i_pTM if available
    designs.sort((a, b) => {
        const a_ptm = a.metrics.i_pTM || 0;
        const b_ptm = b.metrics.i_pTM || 0;
        return b_ptm - a_ptm;
    });
    
    designs.forEach(design => {
        const designItem = document.createElement("div");
        designItem.className = "design-item";
        
        const ptm = design.metrics.i_pTM;
        const plddt = design.metrics.pLDDT || design.metrics.average_plddt;
        
        // Quality indicator
        let qualityClass = "quality-none";
        if (ptm !== undefined) {
            if (ptm >= 0.8) qualityClass = "quality-high";
            else if (ptm >= 0.6) qualityClass = "quality-med";
            else qualityClass = "quality-low";
        }
        
        const indicator = document.createElement("div");
        indicator.className = `quality-indicator ${qualityClass}`;
        
        const info = document.createElement("div");
        info.className = "design-info";
        info.textContent = `${design.design_name} (M${design.model})`;
        
        const score = document.createElement("div");
        score.className = "design-score";
        score.textContent = ptm !== undefined ? ptm.toFixed(2) : "N/A";
        
        designItem.appendChild(indicator);
        designItem.appendChild(info);
        designItem.appendChild(score);
        
        // Tooltip hint
        designItem.title = `i_pTM: ${ptm !== undefined ? ptm.toFixed(3) : 'N/A'}\npLDDT: ${plddt !== undefined ? plddt.toFixed(1) : 'N/A'}`;
        
        designItem.addEventListener("click", () => {
            // Remove active class from all items
            document.querySelectorAll(".design-item").forEach(item => {
                item.classList.remove("active");
            });
            designItem.classList.add("active");
            
            loadDesign(design);
        });
        
        designsList.appendChild(designItem);
    });

    // Reset search on round change
    const designSearch = document.getElementById("designSearch");
    if (designSearch) designSearch.value = "";
}

async function loadDesign(design) {
    if (!design) return;
    currentDesignData = design;
    
    // Display metrics
    metricsDisplay.displayMetrics(design.metrics);
    
    // Show/hide PDB version selector based on available versions
    const pdbVersionSelect = document.getElementById("pdbVersionSelect");
    const availableVersions = [];
    
    console.log("Design paths for", design.design_name, ":", {
        af2: design.af2_pdb_path,
        relaxed: design.relaxed_pdb_path,
        laproteina: design.laproteina_pdb_path
    });
    
    if (design.af2_pdb_path) availableVersions.push('af2');
    if (design.relaxed_pdb_path) availableVersions.push('relaxed');
    if (design.laproteina_pdb_path) availableVersions.push('laproteina');
    
    // Always show selector if we have at least one version available
    // This allows switching between LaProteina, AF2, and Relaxed
    if (pdbVersionSelect && availableVersions.length > 0) {
        // Show selector
        pdbVersionSelect.style.display = "inline-block";
        
        // Update options to show all available versions
        pdbVersionSelect.innerHTML = '';
        
        // Add AF2 Prediction
        if (design.af2_pdb_path) {
            const option = document.createElement('option');
            option.value = 'af2';
            option.textContent = 'AF2 Prediction';
            if (currentPdbVersion === 'af2') {
                option.selected = true;
            }
            pdbVersionSelect.appendChild(option);
        }
        
        // Add LaProteina
        if (design.laproteina_pdb_path) {
            const option = document.createElement('option');
            option.value = 'laproteina';
            option.textContent = 'LaProteina Prediction';
            if (currentPdbVersion === 'laproteina') {
                option.selected = true;
            }
            pdbVersionSelect.appendChild(option);
        }
        
        // Add Relaxed
        if (design.relaxed_pdb_path) {
            const option = document.createElement('option');
            option.value = 'relaxed';
            option.textContent = 'Relaxed';
            if (currentPdbVersion === 'relaxed') {
                option.selected = true;
            }
            pdbVersionSelect.appendChild(option);
        }
    } else {
        // No versions available - hide selector
        if (pdbVersionSelect) {
            pdbVersionSelect.style.display = "none";
        }
    }
    
    // Check if viewer is ready
    if (!viewer || !viewerReady) {
        console.error("Viewer is not ready. Waiting for initialization...");
        // Wait a bit and retry
        setTimeout(() => {
            if (viewer && viewerReady) {
                loadDesign(design);
            } else {
                alert("Viewer is not initialized. Please reload the page.");
            }
        }, 500);
        return;
    }
    
    // Preload all available versions
    const cacheKey = design.design_name;
    if (!preloadedPdbCache.has(cacheKey)) {
        console.log("Preloading PDB files for design:", design.design_name);
        const preloadPromises = [];
        const cacheEntry = {};
        
        if (design.af2_pdb_path) {
            preloadPromises.push(
                fetch(dataParser.getPdbUrl(design.af2_pdb_path))
                    .then(response => {
                        if (response.ok) {
                            return response.text().then(text => {
                                cacheEntry.af2 = text;
                                console.log("Preloaded AF2 version");
                            });
                        }
                    })
                    .catch(error => {
                        console.error("Error preloading AF2 version:", error);
                    })
            );
        }
        
        if (design.relaxed_pdb_path) {
            preloadPromises.push(
                fetch(dataParser.getPdbUrl(design.relaxed_pdb_path))
                    .then(response => {
                        if (response.ok) {
                            return response.text().then(text => {
                                cacheEntry.relaxed = text;
                                console.log("Preloaded relaxed version");
                            });
                        }
                    })
                    .catch(error => {
                        console.error("Error preloading relaxed version:", error);
                    })
            );
        }
        
        if (design.laproteina_pdb_path) {
            console.log("Preloading LaProteina version from:", design.laproteina_pdb_path);
            preloadPromises.push(
                fetch(dataParser.getPdbUrl(design.laproteina_pdb_path))
                    .then(response => {
                        if (response.ok) {
                            return response.text().then(text => {
                                cacheEntry.laproteina = text;
                                console.log("Preloaded LaProteina version");
                            });
                        } else {
                            console.warn("Failed to preload LaProteina version:", response.status, response.statusText);
                        }
                    })
                    .catch(error => {
                        console.error("Error preloading LaProteina version:", error);
                    })
            );
        } else {
            console.log("No LaProteina PDB path found for design:", design.design_name);
        }
        
        // Wait for all preloads to complete
        await Promise.all(preloadPromises);
        preloadedPdbCache.set(cacheKey, cacheEntry);
        console.log("Finished preloading PDB files");
    }
    
    // Load structure in viewer - respect currentPdbVersion, but fallback if version not available
    const cacheEntry = preloadedPdbCache.get(cacheKey);
    let pdbText = null;
    let selectedVersion = currentPdbVersion;
    
    // Check if the currently selected version is available for this design
    const versionAvailable = (version) => {
        if (version === 'af2') return cacheEntry && cacheEntry.af2;
        if (version === 'laproteina') return cacheEntry && cacheEntry.laproteina;
        if (version === 'relaxed') return cacheEntry && cacheEntry.relaxed;
        return false;
    };
    
    // Try to use currentPdbVersion, but fallback to AF2 if not available
    if (versionAvailable(currentPdbVersion)) {
        if (currentPdbVersion === 'af2') {
            pdbText = cacheEntry.af2;
        } else if (currentPdbVersion === 'laproteina') {
            pdbText = cacheEntry.laproteina;
        } else if (currentPdbVersion === 'relaxed') {
            pdbText = cacheEntry.relaxed;
        }
        // Update dropdown to match current selection
        if (pdbVersionSelect) {
            pdbVersionSelect.value = currentPdbVersion;
        }
    } else {
        // Fallback: try AF2 first, then LaProteina, then Relaxed
        if (cacheEntry && cacheEntry.af2) {
            pdbText = cacheEntry.af2;
            selectedVersion = 'af2';
            currentPdbVersion = 'af2'; // Update global state
        } else if (cacheEntry && cacheEntry.laproteina) {
            pdbText = cacheEntry.laproteina;
            selectedVersion = 'laproteina';
            currentPdbVersion = 'laproteina';
        } else if (cacheEntry && cacheEntry.relaxed) {
            pdbText = cacheEntry.relaxed;
            selectedVersion = 'relaxed';
            currentPdbVersion = 'relaxed';
        } else {
            // Fallback to URL loading if preload failed
            const pdbPath = design.af2_pdb_path || design.laproteina_pdb_path || design.relaxed_pdb_path;
            
            if (!pdbPath) {
                console.warn("No PDB path available for design:", design);
                showToast("No PDB file found for this design", "error");
                return;
            }
            
            console.log("Loading design from URL (preload failed):", design.design_name);
            try {
                const pdbUrl = dataParser.getPdbUrl(pdbPath);
                const response = await fetch(pdbUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch PDB: ${response.status}`);
                }
                pdbText = await response.text();
                // Determine version from path
                if (pdbPath.includes('laproteina') || pdbPath.includes('generated')) {
                    selectedVersion = 'laproteina';
                    currentPdbVersion = 'laproteina';
                } else if (pdbPath.includes('relaxed')) {
                    selectedVersion = 'relaxed';
                    currentPdbVersion = 'relaxed';
                } else {
                    selectedVersion = 'af2';
                    currentPdbVersion = 'af2';
                }
            } catch (error) {
                console.error("Error loading structure:", error);
                showToast("Failed to load structure: " + error.message, "error");
                return;
            }
        }
        
        // Update dropdown to reflect the actual loaded version
        if (pdbVersionSelect) {
            pdbVersionSelect.value = selectedVersion;
        }
    }
    
    console.log("Loading design:", design.design_name, `(${currentPdbVersion} version)`);
    
    try {
        const component = await viewer.loadStructureFromText(pdbText, {
            usePlddtColoring: true
        });
        
        if (!component) {
            throw new Error("Failed to load structure component");
        }
        
        console.log("Design loaded successfully");
    } catch (error) {
        console.error("Error loading structure:", error);
        showToast("Failed to load structure: " + error.message, "error");
    }
}

/**
 * Load a specific version (AF2, relaxed, or LaProteina) of a design from preloaded cache
 */
async function loadDesignVersion(design, version = 'af2') {
    if (!viewer || !viewerReady) {
        console.warn("Viewer not ready, cannot load design version");
        return;
    }
    
    currentDesignData = design;
    currentPdbVersion = version;
    
    const cacheKey = design.design_name;
    const cacheEntry = preloadedPdbCache.get(cacheKey);
    
    let pdbText = null;
    let pdbPath = null;
    
    // Determine which PDB to load
    if (version === 'relaxed') {
        pdbPath = design.relaxed_pdb_path;
        if (cacheEntry && cacheEntry.relaxed) {
            pdbText = cacheEntry.relaxed;
        }
    } else if (version === 'laproteina') {
        pdbPath = design.laproteina_pdb_path;
        if (cacheEntry && cacheEntry.laproteina) {
            pdbText = cacheEntry.laproteina;
        }
    } else {
        // Default to AF2
        pdbPath = design.af2_pdb_path;
        if (cacheEntry && cacheEntry.af2) {
            pdbText = cacheEntry.af2;
        }
    }
    
    // If not in cache, fetch it
    if (!pdbText && pdbPath) {
        try {
            const pdbUrl = dataParser.getPdbUrl(pdbPath);
            const response = await fetch(pdbUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch PDB: ${response.status}`);
            }
            pdbText = await response.text();
            
            // Cache it for future use
            if (!cacheEntry) {
                preloadedPdbCache.set(cacheKey, {});
            }
            const entry = preloadedPdbCache.get(cacheKey);
            if (version === 'relaxed') {
                entry.relaxed = pdbText;
            } else if (version === 'laproteina') {
                entry.laproteina = pdbText;
            } else {
                entry.af2 = pdbText;
            }
        } catch (error) {
            console.error(`Error fetching ${version} version:`, error);
            alert(`${version} version not available for this design`);
            return;
        }
    }
    
    if (!pdbText) {
        console.error(`${version} version not available for design:`, design.design_name);
        alert(`${version} version not available for this design`);
        return;
    }
    
    try {
        // Load from preloaded text - instant switching!
        await viewer.loadStructureFromText(pdbText, {
            usePlddtColoring: true
        });
        
        // Update dropdown selection
        const pdbVersionSelect = document.getElementById("pdbVersionSelect");
        if (pdbVersionSelect) {
            pdbVersionSelect.value = version;
        }
        
        console.log(`Loaded ${version} version of design:`, design.design_name, pdbText === cacheEntry?.[version] ? "(from cache)" : "(fetched)");
    } catch (error) {
        console.error(`Error loading ${version} version:`, error);
        showToast(`Failed to load ${version} version: ${error.message}`, "error");
    }
}

// Initialize animation controller after viewer is ready
setTimeout(() => {
    if (viewer) {
        animationController = new AnimationController(viewer);
    }
}, 1000);

/**
 * Update highlight button appearance based on state
 */
function updateHighlightButtonState() {
    const btn = document.getElementById("highlightBinderBtn");
    if (btn) {
        if (isBinderHighlighted) {
            btn.textContent = "✨ Unhighlight Binder";
            btn.style.backgroundColor = "#e74c3c";
        } else {
            btn.textContent = "✨ Highlight Binder";
            btn.style.backgroundColor = "";
        }
    }
}

function initializeScoringTable() {
    // Populate round filter dropdown
    const roundFilter = document.getElementById("scoringRoundFilter");
    if (roundFilter && currentRunData) {
        // Clear existing options except "All Rounds"
        roundFilter.innerHTML = '<option value="all">All Rounds</option>';
        
        currentRunData.rounds.forEach(round => {
            const option = document.createElement("option");
            option.value = round.round_number;
            option.textContent = `Round ${round.round_number}`;
            roundFilter.appendChild(option);
        });
    }
    
    // Load table if we have data cached, otherwise show placeholder
    if (allDesigns.length > 0) {
        scoringTableData = allDesigns;
        renderScoringTable();
    } else {
        // Show message to refresh
        document.getElementById("scoringTableContainer").innerHTML = 
            '<div class="scoring-table-loading">Click "Refresh" to load scoring data</div>';
    }
}

async function loadScoringTable() {
    if (!currentRunData) {
        return;
    }
    
    const container = document.getElementById("scoringTableContainer");
    container.innerHTML = '<div class="scoring-table-loading">Loading scoring data...</div>';
    
    try {
        // Load all designs from all rounds
        const loadPromises = currentRunData.rounds.map(async (round) => {
            try {
                const designs = await dataParser.getRoundDesigns(
                    currentRunData.run_name,
                    round.round_number
                );
                return designs;
            } catch (error) {
                console.error(`Error loading round ${round.round_number}:`, error);
                return [];
            }
        });
        
        const allRoundDesigns = await Promise.all(loadPromises);
        scoringTableData = allRoundDesigns.flat();
        
        // Also update allDesigns cache if needed
        if (allDesigns.length === 0) {
            allDesigns = scoringTableData;
        }
        
        renderScoringTable();
    } catch (error) {
        console.error("Error loading scoring table:", error);
        container.innerHTML = 
            `<div class="scoring-table-loading" style="color: #e74c3c;">Error loading scoring data: ${error.message}</div>`;
    }
}

/**
 * Save notes for the current run
 */
async function saveNotes() {
    if (!currentRunData) {
        showToast("No evolution run loaded", "error");
        return;
    }
    
    const notesTextarea = document.getElementById("runNotesTextarea");
    if (!notesTextarea) return;
    
    const notes = notesTextarea.value;
    const saveBtn = document.getElementById("saveNotesBtn");
    const originalText = saveBtn.textContent;
    
    saveBtn.textContent = "⌛ Saving...";
    saveBtn.disabled = true;
    
    try {
        const success = await dataParser.saveRunNotes(currentRunData.run_name, notes);
        if (success) {
            showToast("Notes saved successfully", "success");
        } else {
            showToast("Failed to save notes", "error");
        }
    } catch (error) {
        console.error("Error saving notes:", error);
        showToast("Error saving notes: " + error.message, "error");
    } finally {
        saveBtn.textContent = originalText;
        saveBtn.disabled = false;
    }
}

function renderScoringTable() {
    if (!scoringTableData || scoringTableData.length === 0) {
        document.getElementById("scoringTableContainer").innerHTML = 
            '<div class="scoring-table-empty">No scoring data available</div>';
        return;
    }
    
    // Filter data
    let filteredData = [...scoringTableData];
    
    // Filter by round
    if (scoringTableFilterRound !== 'all') {
        const roundNum = parseInt(scoringTableFilterRound);
        filteredData = filteredData.filter(d => d.round_number === roundNum);
    }
    
    // Sort data (default to i_pTM descending if available, otherwise no sort)
    if (!scoringTableSortColumn && filteredData.length > 0) {
        // Check if i_pTM exists in metrics
        const hasIpTM = filteredData.some(d => d.metrics?.i_pTM != null);
        if (hasIpTM) {
            scoringTableSortColumn = 'i_pTM';
            scoringTableSortDirection = 'desc';
        }
    }
    
    if (scoringTableSortColumn) {
        filteredData.sort((a, b) => {
            let aVal, bVal;
            
            if (scoringTableSortColumn === 'round_number') {
                aVal = a.round_number;
                bVal = b.round_number;
            } else if (scoringTableSortColumn === 'design_name') {
                aVal = a.design_name;
                bVal = b.design_name;
            } else if (scoringTableSortColumn === 'model') {
                aVal = a.model;
                bVal = b.model;
            } else if (scoringTableSortColumn === 'status') {
                aVal = a.status || '';
                bVal = b.status || '';
            } else {
                // Metric column
                aVal = a.metrics?.[scoringTableSortColumn] ?? null;
                bVal = b.metrics?.[scoringTableSortColumn] ?? null;
            }
            
            // Handle null/undefined
            if (aVal == null && bVal == null) return 0;
            if (aVal == null) return 1;
            if (bVal == null) return -1;
            
            // Compare
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return scoringTableSortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            } else {
                const aStr = String(aVal).toLowerCase();
                const bStr = String(bVal).toLowerCase();
                if (scoringTableSortDirection === 'asc') {
                    return aStr.localeCompare(bStr);
                } else {
                    return bStr.localeCompare(aStr);
                }
            }
        });
    }
    
    // Get all unique metric keys
    const metricKeys = new Set();
    filteredData.forEach(design => {
        if (design.metrics) {
            Object.keys(design.metrics).forEach(key => metricKeys.add(key));
        }
    });
    const sortedMetricKeys = Array.from(metricKeys).sort();
    
    // Build table
    const table = document.createElement('table');
    table.className = 'scoring-table';
    
    // Header row
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const headerColumns = [
        { key: 'round_number', label: 'Round', class: 'round-col' },
        { key: 'design_name', label: 'Design Name', class: 'design-col' },
        { key: 'model', label: 'Model', class: '' },
        { key: 'status', label: 'Status', class: '' },
        ...sortedMetricKeys.map(key => ({ key, label: key, class: 'numeric' }))
    ];
    
    headerColumns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.label;
        th.className = col.class + (col.key !== 'status' ? ' sortable' : '');
        th.dataset.column = col.key;
        
        if (scoringTableSortColumn === col.key) {
            th.classList.add(`sort-${scoringTableSortDirection}`);
        }
        
        if (col.key !== 'status') {
            th.addEventListener('click', () => {
                if (scoringTableSortColumn === col.key) {
                    scoringTableSortDirection = scoringTableSortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    scoringTableSortColumn = col.key;
                    scoringTableSortDirection = 'desc';
                }
                renderScoringTable();
            });
        }
        
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Body rows
    const tbody = document.createElement('tbody');
    filteredData.forEach((design, idx) => {
        const row = document.createElement('tr');
        row.dataset.designName = design.design_name;
        row.dataset.roundNumber = design.round_number;
        
        // Round
        const tdRound = document.createElement('td');
        tdRound.className = 'round-col';
        tdRound.textContent = design.round_number;
        row.appendChild(tdRound);
        
        // Design name
        const tdDesign = document.createElement('td');
        tdDesign.className = 'design-col';
        tdDesign.textContent = design.design_name;
        row.appendChild(tdDesign);
        
        // Model
        const tdModel = document.createElement('td');
        tdModel.textContent = design.model || '';
        row.appendChild(tdModel);
        
        // Status
        const tdStatus = document.createElement('td');
        tdStatus.textContent = design.status || '';
        row.appendChild(tdStatus);
        
        // Metrics
        sortedMetricKeys.forEach(key => {
            const td = document.createElement('td');
            td.className = 'numeric';
            const value = design.metrics?.[key];
            if (value != null) {
                if (typeof value === 'number') {
                    td.textContent = value.toFixed(3);
                } else {
                    td.textContent = String(value);
                }
            } else {
                td.textContent = '—';
            }
            row.appendChild(td);
        });
        
        // Click handler to load design in viewer
        row.addEventListener('click', () => {
            switchTab('viewer');
            loadDesign(design);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    
    // Update container
    const container = document.getElementById("scoringTableContainer");
    container.innerHTML = '';
    container.appendChild(table);
    
    // Add summary
    const summary = document.createElement('div');
    summary.style.cssText = 'padding: 0.5rem; color: #666; font-size: 0.85rem; border-top: 1px solid #ddd; margin-top: 0.5rem;';
    summary.textContent = `Showing ${filteredData.length} of ${scoringTableData.length} designs`;
    container.appendChild(summary);
}

