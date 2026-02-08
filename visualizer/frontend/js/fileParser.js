/**
 * Client-side file parser for evolution data.
 * Communicates with backend API.
 */

const API_BASE_URL = "http://127.0.0.1:5000/api";

class EvolutionDataParser {
    constructor() {
        this.currentRun = null;
        this.currentRound = null;
        this.currentDesign = null;
    }
    
    async listRuns() {
        try {
            const response = await fetch(`${API_BASE_URL}/runs`);
            const data = await response.json();
            return data.runs || [];
        } catch (error) {
            console.error("Error listing runs:", error);
            return [];
        }
    }

    async getRunNotes(runName) {
        try {
            const response = await fetch(`${API_BASE_URL}/runs/${encodeURIComponent(runName)}/notes`);
            const data = await response.json();
            return data.notes || "";
        } catch (error) {
            console.error("Error getting run notes:", error);
            return "";
        }
    }

    async saveRunNotes(runName, notes) {
        try {
            const response = await fetch(`${API_BASE_URL}/runs/${encodeURIComponent(runName)}/notes`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ notes })
            });
            const data = await response.json();
            return data.status === "success";
        } catch (error) {
            console.error("Error saving run notes:", error);
            return false;
        }
    }
    
    async getRun(runName) {
        try {
            const response = await fetch(`${API_BASE_URL}/runs/${encodeURIComponent(runName)}`);
            const data = await response.json();
            this.currentRun = data;
            return data;
        } catch (error) {
            console.error("Error getting run:", error);
            return null;
        }
    }
    
    async getRoundDesigns(runName, roundNum) {
        try {
            const url = `${API_BASE_URL}/runs/${encodeURIComponent(runName)}/rounds/${roundNum}/designs`;
            console.log("Fetching designs from:", url);
            
            const response = await fetch(url);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`HTTP ${response.status}: ${errorData.error || response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            return data.designs || [];
        } catch (error) {
            console.error("Error getting round designs:", error);
            throw error; // Re-throw to let caller handle it
        }
    }
    
    async getStructureInfo(filePath) {
        try {
            const encodedPath = encodeURIComponent(filePath);
            const response = await fetch(`${API_BASE_URL}/structure/${encodedPath}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error("Error getting structure info:", error);
            return null;
        }
    }
    
    getPdbUrl(filePath) {
        // Handle both absolute and relative paths
        let path = filePath;
        
        // If it's already a string Path object, convert it
        if (path && typeof path === 'object' && path.toString) {
            path = path.toString();
        }
        
        // Ensure path is a string
        path = String(path);
        
        // Ensure path uses forward slashes
        path = path.replace(/\\/g, '/');
        
        // Remove any leading slashes or dots
        path = path.replace(/^\.\.\//g, '');
        path = path.replace(/^\.\//g, '');
        path = path.replace(/^\//g, '');
        
        // Ensure evolution_output/ prefix is present if path doesn't start with it
        // and doesn't already include it
        if (!path.startsWith('evolution_output/') && !path.includes('/evolution_output/')) {
            // If path starts with evolution_ (a run name), add evolution_output/ prefix
            // Matches both classic "evolution_YYYYMMDD_HHMM/" and v2 "evolution_v2_YYYYMMDD_HHMM/"
            if (path.match(/^evolution_(v2_)?\d{8}_\d{4}\//)) {
                path = 'evolution_output/' + path;
            }
        }
        
        console.log("getPdbUrl - input:", filePath, "output:", path);
        
        // The backend expects paths relative to BindCraft root
        const encodedPath = encodeURIComponent(path);
        return `${API_BASE_URL}/pdb/${encodedPath}`;
    }
    
    async checkHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            const data = await response.json();
            return data.status === "ok";
        } catch (error) {
            console.error("Health check failed:", error);
            return false;
        }
    }
}

// Global parser instance
const dataParser = new EvolutionDataParser();

