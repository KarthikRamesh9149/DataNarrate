// NO IMPORTS - This is a dynamic window!
// All dependencies are provided globally by the app

const DataNarrateWindow = () => {
  const [serverOnline, setServerOnline] = React.useState(false);
  const [datasetInfo, setDatasetInfo] = React.useState(null);
  const [intent, setIntent] = React.useState("");
  const [profile, setProfile] = React.useState(null);
  const [cleaningLog, setCleaningLog] = React.useState([]);
  const [charts, setCharts] = React.useState([]);
  const [explanations, setExplanations] = React.useState([]);
  const [explanationMode, setExplanationMode] = React.useState("quick");
  const [allowSampleRows, setAllowSampleRows] = React.useState(false);
  const [loading, setLoading] = React.useState({
    ingest: false,
    profile: false,
    plan: false,
    clean: false,
    charts: false,
    explain: false,
  });
  const [error, setError] = React.useState(null);
  const [vizPlan, setVizPlan] = React.useState(null);
  const [activeTab, setActiveTab] = React.useState("setup");
  const [showProfile, setShowProfile] = React.useState(false);
  const [showCleaningLog, setShowCleaningLog] = React.useState(false);

  // Cleaning summary and final analysis states
  const [cleaningSummary, setCleaningSummary] = React.useState(null);
  const [finalSummary, setFinalSummary] = React.useState(null);
  const [loadingFinalSummary, setLoadingFinalSummary] = React.useState(false);

  // Analysis phase tracking: idle -> cleaning (user reviews) -> complete
  const [analysisPhase, setAnalysisPhase] = React.useState<"idle" | "cleaning" | "complete">("idle");

  // Target column inference state
  const [targetInference, setTargetInference] = React.useState(null);
  const [selectedTarget, setSelectedTarget] = React.useState(null);
  const [showTargetSelector, setShowTargetSelector] = React.useState(false);

  // Modeling state
  const [modelResults, setModelResults] = React.useState(null);
  const [modelExplanation, setModelExplanation] = React.useState(null);
  const [maxIter, setMaxIter] = React.useState(1000);
  const [loadingModel, setLoadingModel] = React.useState(false);

  // Server setup state
  const [availableVenvs, setAvailableVenvs] = React.useState([]);
  const [selectedVenv, setSelectedVenv] = React.useState('');
  const [serverPort, setServerPort] = React.useState(8891);
  const [serverRunning, setServerRunning] = React.useState(false);
  const [connecting, setConnecting] = React.useState(false);
  const [checkingDeps, setCheckingDeps] = React.useState(false);
  const [installingDeps, setInstallingDeps] = React.useState(false);
  const [installingPackage, setInstallingPackage] = React.useState('');
  const [depsStatus, setDepsStatus] = React.useState({});
  const [serverStatus, setServerStatus] = React.useState(null);

  const REQUIRED_PACKAGES = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'matplotlib', 'httpx', 'python-dotenv', 'openpyxl', 'pydantic', 'python-multipart', 'scikit-learn', 'scipy', 'joblib'];

  const getServerUrl = () => `http://127.0.0.1:${serverPort}`;
  const fileInputRef = React.useRef(null);

  const normalizePackageName = (name: string): string => name.toLowerCase().replace(/-/g, '_');

  const parsePackageInfo = (pkgStr: string): { name: string; version?: string } => {
    if (pkgStr.includes(' @ ')) {
      const name = pkgStr.split(' @ ')[0].trim();
      return { name, version: 'local' };
    }
    if (pkgStr.includes('==')) {
      const [name, version] = pkgStr.split('==');
      return { name: name.trim(), version: version?.trim() };
    }
    const parts = pkgStr.split(' ');
    return { name: parts[0].trim(), version: parts[1]?.trim() };
  };

  const checkDeps = async () => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer || !selectedVenv) return;

    setCheckingDeps(true);
    try {
      // Force refresh by passing refresh: true
      const vres = await ipcRenderer.invoke('python-list-venvs', { refresh: true });
      console.log('Venvs response:', vres);
      if (vres.success) {
        const v = (vres.venvs || []).find((x: any) => x.name === selectedVenv);
        console.log('Selected venv data:', v);
        if (v && Array.isArray(v.packages)) {
          console.log('Packages in venv:', v.packages.slice(0, 20)); // Log first 20
          const map: Record<string, { installed: boolean; version?: string }> = {};
          for (const pkg of REQUIRED_PACKAGES) {
            const normalizedPkg = normalizePackageName(pkg);
            const found = v.packages.find((p: string) => {
              const parsed = parsePackageInfo(p);
              return normalizePackageName(parsed.name) === normalizedPkg;
            });
            if (found) {
              const parsed = parsePackageInfo(found);
              map[pkg] = { installed: true, version: parsed.version };
            } else {
              map[pkg] = { installed: false };
            }
          }
          console.log('Deps status map:', map);
          setDepsStatus(map);
        }
      }
    } catch (e: any) {
      console.error('Error checking deps:', e);
    } finally {
      setCheckingDeps(false);
    }
  };

  // Poll server status
  React.useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${getServerUrl()}/status`);
        const data = await res.json();
        setServerOnline(data.success);
        setServerStatus(data);
        setServerRunning(true);
      } catch {
        setServerOnline(false);
        setServerRunning(false);
        setServerStatus(null);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [serverPort]);

  // Load available venvs
  React.useEffect(() => {
    const loadVenvs = async () => {
      const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
      if (!ipcRenderer) return;

      const result = await ipcRenderer.invoke('python-list-venvs');
      if (result.success && result.venvs.length > 0) {
        const names = result.venvs.map((v: any) => v.name);
        setAvailableVenvs(names);
        if (!selectedVenv) {
          setSelectedVenv(names[0]);
        }
      }
    };
    loadVenvs();
  }, []);

  // Auto-check deps when venv changes
  React.useEffect(() => {
    if (selectedVenv) {
      checkDeps();
    }
  }, [selectedVenv]);

  const handleError = (message) => {
    setError(message);
    setTimeout(() => setError(null), 5000);
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading((prev) => ({ ...prev, ingest: true }));
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("use_sample", "false");

      const res = await fetch(`${getServerUrl()}/ingest`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.success) {
        // Reset ALL state for fresh analysis workflow
        setDatasetInfo(data.data);
        setProfile(null);
        setCleaningLog([]);
        setCharts([]);
        setExplanations([]);
        setVizPlan(null);
        // Reset target and modeling state
        setTargetInference(null);
        setSelectedTarget(null);
        setShowTargetSelector(false);
        setModelResults(null);
        setModelExplanation(null);
        setFinalSummary(null);
        // Reset to data tab for fresh start
        setActiveTab("data");
      } else {
        handleError(data.error || "Failed to load dataset");
      }
    } catch (err) {
      handleError("Failed to connect to server");
    } finally {
      setLoading((prev) => ({ ...prev, ingest: false }));
    }
  };

  const handleUseSample = async () => {
    setLoading((prev) => ({ ...prev, ingest: true }));
    setError(null);

    try {
      const formData = new FormData();
      formData.append("use_sample", "true");

      const res = await fetch(`${getServerUrl()}/ingest`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.success) {
        // Reset ALL state for fresh analysis workflow
        setDatasetInfo(data.data);
        setProfile(null);
        setCleaningLog([]);
        setCharts([]);
        setExplanations([]);
        setVizPlan(null);
        // Reset target and modeling state
        setTargetInference(null);
        setSelectedTarget(null);
        setShowTargetSelector(false);
        setModelResults(null);
        setModelExplanation(null);
        setFinalSummary(null);
        // Reset to data tab for fresh start
        setActiveTab("data");
      } else {
        handleError(data.error || "Failed to load sample dataset");
      }
    } catch (err) {
      handleError("Failed to connect to server");
    } finally {
      setLoading((prev) => ({ ...prev, ingest: false }));
    }
  };

  // Phase 1: Profile the data and show cleaning options
  const runAnalysis = async () => {
    if (!datasetInfo || !intent.trim()) return;

    setError(null);
    setCleaningSummary(null);
    setFinalSummary(null);
    setCharts([]);
    setExplanations([]);
    setTargetInference(null);
    setShowTargetSelector(false);

    // Step 1: Profile the original data
    setLoading((prev) => ({ ...prev, profile: true }));
    try {
      const profileRes = await fetch(`${getServerUrl()}/profile`, { method: "POST" });
      const profileData = await profileRes.json();
      if (profileData.success) {
        setProfile(profileData.data);
      } else {
        throw new Error(profileData.error);
      }
    } catch (err) {
      handleError("Profiling failed: " + err.message);
      setLoading((prev) => ({ ...prev, profile: false }));
      return;
    }
    setLoading((prev) => ({ ...prev, profile: false }));

    // Step 1.5: Target inference (if intent implies prediction)
    try {
      const targetRes = await fetch(`${getServerUrl()}/infer-target`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent, allow_sample_rows: allowSampleRows }),
      });
      const targetData = await targetRes.json();
      if (targetData.success) {
        setTargetInference(targetData.data);
        if (targetData.data.chosen_target) {
          setSelectedTarget(targetData.data.chosen_target);
        }
        // Show selector if needed
        if (targetData.data.needs_user_confirmation && targetData.data.target_candidates?.length > 0) {
          setShowTargetSelector(true);
        }
      }
    } catch (err) {
      console.warn("Target inference failed (non-critical):", err);
    }

    // Switch to cleaning tab and show options
    setActiveTab("cleaning");
    setAnalysisPhase("cleaning");
  };

  // Set target column explicitly
  const setTargetColumn = async (column: string | null) => {
    try {
      await fetch(`${getServerUrl()}/set-target`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: column }),
      });
      setSelectedTarget(column);
      setShowTargetSelector(false);
    } catch (err) {
      console.error("Failed to set target:", err);
    }
  };

  // Phase 2: Apply cleaning and generate visualizations
  const continueAnalysis = async (applyRecommended: boolean = false) => {
    setError(null);

    // Step 2: Apply Cleaning
    setLoading((prev) => ({ ...prev, clean: true }));
    try {
      const cleanRes = await fetch(`${getServerUrl()}/clean/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply_recommended: applyRecommended }),
      });
      const cleanData = await cleanRes.json();
      if (cleanData.success) {
        setCleaningLog(cleanData.data.cleaning_log || []);
        setDatasetInfo((prev) => ({
          ...prev,
          n_rows: cleanData.data.n_rows,
          n_cols: cleanData.data.n_cols,
          columns: cleanData.data.columns,  // Update columns with new snake_case names
          preview: cleanData.data.preview,
        }));

        // Map selectedTarget to snake_case version if it was renamed
        const colMapping = cleanData.data.column_name_mapping || {};
        if (selectedTarget && colMapping[selectedTarget]) {
          setSelectedTarget(colMapping[selectedTarget]);
        }
      } else {
        throw new Error(cleanData.error);
      }
    } catch (err) {
      handleError("Cleaning failed: " + err.message);
      setLoading((prev) => ({ ...prev, clean: false }));
      return;
    }
    setLoading((prev) => ({ ...prev, clean: false }));

    // Step 2.5: Fetch Cleaning Summary
    try {
      const summaryRes = await fetch(`${getServerUrl()}/cleaning-summary`, { method: "POST" });
      const summaryData = await summaryRes.json();
      if (summaryData.success) {
        setCleaningSummary(summaryData.data);
      }
    } catch (err) {
      console.error("Failed to fetch cleaning summary:", err);
    }

    // Step 3: Re-profile the cleaned data (now has snake_case columns)
    setLoading((prev) => ({ ...prev, profile: true }));
    try {
      const profileRes = await fetch(`${getServerUrl()}/profile`, { method: "POST" });
      const profileData = await profileRes.json();
      if (profileData.success) {
        setProfile(profileData.data);
      }
    } catch (err) {
      console.error("Failed to re-profile cleaned data:", err);
    }
    setLoading((prev) => ({ ...prev, profile: false }));

    // Step 4: Generate Plan (now uses cleaned profile with snake_case columns)
    setLoading((prev) => ({ ...prev, plan: true }));
    try {
      const planRes = await fetch(`${getServerUrl()}/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent, allow_sample_rows: allowSampleRows }),
      });
      const planData = await planRes.json();
      if (planData.success) {
        setVizPlan(planData.data);
      } else {
        throw new Error(planData.error);
      }
    } catch (err) {
      handleError("Planning failed: " + err.message);
      setLoading((prev) => ({ ...prev, plan: false }));
      return;
    }
    setLoading((prev) => ({ ...prev, plan: false }));

    // Step 5: Generate Charts
    setLoading((prev) => ({ ...prev, charts: true }));
    try {
      const chartsRes = await fetch(`${getServerUrl()}/charts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ intent }),
      });
      const chartsData = await chartsRes.json();
      if (chartsData.success) {
        setCharts(chartsData.data.charts || []);
        setActiveTab("charts");
        setAnalysisPhase("complete");
      } else {
        throw new Error(chartsData.error);
      }
    } catch (err) {
      handleError("Chart generation failed: " + err.message);
      setLoading((prev) => ({ ...prev, charts: false }));
      return;
    }
    setLoading((prev) => ({ ...prev, charts: false }));

    // Step 6: Generate Explanations
    setLoading((prev) => ({ ...prev, explain: true }));
    try {
      const explainRes = await fetch(`${getServerUrl()}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: explanationMode,
          intent,
          allow_sample_rows: allowSampleRows,
        }),
      });
      const explainData = await explainRes.json();
      if (explainData.success) {
        setExplanations(explainData.data.explanations || []);
      } else {
        throw new Error(explainData.error);
      }
    } catch (err) {
      handleError("Explanation failed: " + err.message);
    } finally {
      setLoading((prev) => ({ ...prev, explain: false }));
    }

    // Note: Final Summary is now generated only after model training in the Modeling tab
  };

  const startServer = async () => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) {
      setError('Not running in Electron environment');
      return;
    }

    setConnecting(true);
    setError(null);

    if (!selectedVenv) {
      setError('Please select a Python virtual environment');
      setConnecting(false);
      return;
    }

    console.log('Starting server with venv:', selectedVenv);

    // Get the script path
    const scriptResult = await ipcRenderer.invoke('resolve-workflow-script', {
      workflowFolder: 'DataNarrate',
      scriptName: 'datanarrate_server.py'
    });

    console.log('Script resolution result:', scriptResult);

    if (!scriptResult.success) {
      setError(`Could not find server script: ${scriptResult.error}`);
      setConnecting(false);
      return;
    }

    console.log('Starting server with script path:', scriptResult.path);

    const result = await ipcRenderer.invoke('python-start-script-server', {
      venvName: selectedVenv,
      scriptPath: scriptResult.path,
      port: serverPort,
      serverName: 'datanarrate',
    });

    console.log('Server start result:', result);

    if (result.success) {
      console.log(`Server process started (PID: ${result.pid}), waiting for connection...`);
      // Poll for server connection
      let attempts = 0;
      const maxAttempts = 30;
      const pollInterval = setInterval(async () => {
        attempts++;
        try {
          const res = await fetch(`${getServerUrl()}/status`);
          if (res.ok) {
            const data = await res.json();
            console.log('Server connected!', data);
            setServerStatus(data);
            setServerRunning(true);
            setConnecting(false);
            clearInterval(pollInterval);
          }
        } catch (e) {
          console.log(`Attempt ${attempts}/${maxAttempts} - waiting for server...`);
          if (attempts >= maxAttempts) {
            clearInterval(pollInterval);
            setError('Server failed to start within timeout. Check console for details.');
            setConnecting(false);
          }
        }
      }, 1000);
    } else {
      setError(`Failed to start server: ${result.error}`);
      setConnecting(false);
    }
  };

  const stopServer = async () => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) return;

    const result = await ipcRenderer.invoke('python-stop-script-server', 'datanarrate');
    if (result.success) {
      setServerRunning(false);
      setServerStatus(null);
    } else {
      try {
        await fetch(`${getServerUrl()}/shutdown`, { method: 'POST' });
      } catch {
        // Server already stopped
      }
      setServerRunning(false);
      setServerStatus(null);
    }
  };

  const installMissingDeps = async () => {
    const ipcRenderer = (window as any).require?.('electron')?.ipcRenderer;
    if (!ipcRenderer) return;

    setInstallingDeps(true);

    for (const pkg of REQUIRED_PACKAGES) {
      if (!depsStatus[pkg]?.installed) {
        setInstallingPackage(pkg);
        const result = await ipcRenderer.invoke('python-install-package', {
          venvName: selectedVenv,
          package: pkg,
        });
        console.log(`Install ${pkg}:`, result);
      }
    }

    setInstallingPackage('');
    setInstallingDeps(false);
    await checkDeps();
  };

  const applyRecommendedCleaning = async () => {
    setLoading((prev) => ({ ...prev, clean: true }));
    try {
      const res = await fetch(`${getServerUrl()}/clean/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apply_recommended: true }),
      });
      const data = await res.json();
      if (data.success) {
        setCleaningLog(data.data.cleaning_log || []);
        setDatasetInfo((prev) => ({
          ...prev,
          n_rows: data.data.n_rows,
          n_cols: data.data.n_cols,
          preview: data.data.preview,
        }));
      } else {
        handleError(data.error);
      }
    } catch (err) {
      handleError("Recommended cleaning failed");
    } finally {
      setLoading((prev) => ({ ...prev, clean: false }));
    }
  };

  const regenerateExplanations = async () => {
    if (charts.length === 0) return;
    setLoading((prev) => ({ ...prev, explain: true }));
    try {
      const res = await fetch(`${getServerUrl()}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: explanationMode,
          intent,
          allow_sample_rows: allowSampleRows,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setExplanations(data.data.explanations || []);
      } else {
        handleError(data.error);
      }
    } catch (err) {
      handleError("Explanation failed");
    } finally {
      setLoading((prev) => ({ ...prev, explain: false }));
    }
  };

  const downloadCleaned = () => {
    window.open(`${getServerUrl()}/download/cleaned`, "_blank");
  };

  const downloadCharts = () => {
    window.open(`${getServerUrl()}/download/charts`, "_blank");
  };

  // Model training function
  const trainModel = async () => {
    if (!selectedTarget) return;

    setLoadingModel(true);
    setModelResults(null);
    setModelExplanation(null);

    try {
      const res = await fetch(`${getServerUrl()}/model/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target_column: selectedTarget,
          task_type: targetInference?.task_type || "classification",
          max_iter: maxIter,
          test_size: 0.2,
          random_seed: 42,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setModelResults(data.data);
        // Auto-generate model explanation
        try {
          const explainRes = await fetch(`${getServerUrl()}/model/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              mode: explanationMode,
              intent,
            }),
          });
          const explainData = await explainRes.json();
          if (explainData.success) {
            setModelExplanation(explainData.data);
          }
        } catch (err) {
          console.error("Model explanation failed:", err);
        }

        // Auto-generate final summary after model training
        setLoadingFinalSummary(true);
        try {
          const finalRes = await fetch(`${getServerUrl()}/final-summary`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ intent, allow_sample_rows: allowSampleRows }),
          });
          const finalData = await finalRes.json();
          if (finalData.success) {
            setFinalSummary(finalData.data.summary);
          }
        } catch (err) {
          console.error("Failed to generate final summary:", err);
        } finally {
          setLoadingFinalSummary(false);
        }
      } else {
        handleError("Model training failed: " + data.error);
      }
    } catch (err) {
      handleError("Model training failed: " + err.message);
    } finally {
      setLoadingModel(false);
    }
  };

  const isLoading = Object.values(loading).some((v) => v);
  const canRunAnalysis = datasetInfo && intent.trim() && !isLoading && serverOnline;

  return (
    <div className="flex flex-col h-full bg-gray-900 text-gray-100">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <div className="text-xl font-bold text-emerald-400">DataNarrate</div>
          <div className="text-xs text-gray-400">Intent-Aware Data Science Copilot</div>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${serverOnline ? "bg-emerald-400" : "bg-red-400"}`}
          />
          <span className="text-xs text-gray-400">
            {serverOnline ? "Server online" : "Server offline"}
          </span>
        </div>
      </div>


      {error && (
        <div className="px-4 py-2 bg-red-900/30 border-b border-red-700/50 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Data & Intent */}
        <div className="w-80 flex-shrink-0 flex flex-col border-r border-gray-700 bg-gray-850">
          {/* Dataset Selection */}
          <div className="p-4 border-b border-gray-700">
            <div className="text-sm font-medium text-gray-300 mb-3">Dataset</div>
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={!serverOnline || loading.ingest}
                className="flex-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm transition-colors"
              >
                Upload
              </button>
              <button
                onClick={handleUseSample}
                disabled={!serverOnline || loading.ingest}
                className="flex-1 px-3 py-2 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 rounded text-sm transition-colors"
              >
                Use Sample
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls,.json"
                onChange={handleUpload}
                className="hidden"
              />
            </div>

            {loading.ingest && (
              <div className="text-xs text-gray-400 animate-pulse">Loading dataset...</div>
            )}

            {datasetInfo && (
              <div className="bg-gray-800 rounded p-2 text-xs">
                <div className="font-medium text-emerald-400 truncate">{datasetInfo.dataset_name}</div>
                <div className="text-gray-400 mt-1">
                  {datasetInfo.n_rows?.toLocaleString()} rows × {datasetInfo.n_cols} columns
                </div>
              </div>
            )}
          </div>

          {/* Intent Input */}
          <div className="p-4 border-b border-gray-700 flex-1 flex flex-col">
            <div className="text-sm font-medium text-gray-300 mb-2">Project Intent</div>
            <textarea
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              placeholder="Describe your analysis goal, e.g., 'Analyze team performance patterns and identify factors contributing to match outcomes'"
              className="flex-1 bg-gray-800 border border-gray-700 rounded p-2 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Options */}
          <div className="p-4 border-b border-gray-700">
            <div className="text-sm font-medium text-gray-300 mb-3">Options</div>
            <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer mb-2">
              <input
                type="checkbox"
                checked={allowSampleRows}
                onChange={(e) => setAllowSampleRows(e.target.checked)}
                className="rounded border-gray-600 bg-gray-800 text-emerald-500 focus:ring-emerald-500"
              />
              Allow ≤5 sample rows to LLM
            </label>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span>Explanation:</span>
              <select
                value={explanationMode}
                onChange={(e) => setExplanationMode(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none focus:border-emerald-500"
              >
                <option value="quick">Quick</option>
                <option value="deep">Deep</option>
              </select>
            </div>
          </div>

          {/* Run Analysis Button */}
          <div className="p-4">
            <button
              onClick={runAnalysis}
              disabled={!canRunAnalysis}
              className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:opacity-50 rounded font-medium transition-colors"
            >
              {isLoading ? (
                <span className="animate-pulse">
                  {loading.profile && "Profiling..."}
                  {loading.plan && "Planning..."}
                  {loading.clean && "Cleaning..."}
                  {loading.charts && "Generating charts..."}
                  {loading.explain && "Explaining..."}
                </span>
              ) : (
                "Run Analysis"
              )}
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-gray-700 bg-gray-800">
            {["setup", "data", "profile", "cleaning", "charts", "modeling"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? "text-emerald-400 border-b-2 border-emerald-400"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                {tab === "charts" && charts.length > 0 && (
                  <span className="ml-1.5 px-1.5 py-0.5 bg-emerald-600 rounded text-xs">{charts.length}</span>
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-auto p-4">
            {/* Setup Tab */}
            {activeTab === "setup" && (
              <div className="space-y-4">
                {/* Server Connection Section */}
                <div className="bg-gray-800 rounded p-4 border border-gray-700">
                  <h4 className="m-0 mb-3 text-[13px] font-medium text-gray-300">Server Connection</h4>
                  <div className="flex items-center gap-2.5 mb-2 flex-wrap">
                    <span className="text-[13px] text-gray-400">Venv:</span>
                    <select
                      value={selectedVenv}
                      onChange={e => setSelectedVenv(e.target.value)}
                      className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-[13px] text-gray-100 focus:outline-none focus:border-emerald-500 w-[140px]"
                      disabled={serverRunning}
                    >
                      {availableVenvs.length === 0 ? (
                        <option value="">No venvs</option>
                      ) : (
                        availableVenvs.map(name => (
                          <option key={name} value={name}>{name}</option>
                        ))
                      )}
                    </select>
                    <span className="text-[13px] text-gray-400">Port:</span>
                    <input
                      type="number"
                      value={serverPort}
                      onChange={e => setServerPort(parseInt(e.target.value) || 8891)}
                      className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-[13px] text-gray-100 focus:outline-none focus:border-emerald-500 w-[70px]"
                      disabled={serverRunning}
                    />
                    {!serverRunning ? (
                      <button
                        onClick={startServer}
                        disabled={connecting || !selectedVenv}
                        className={`px-3 py-1 rounded text-[13px] font-medium transition-colors ${connecting ? 'bg-gray-600 opacity-50 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`}
                      >
                        {connecting ? 'Connecting...' : 'Start Server'}
                      </button>
                    ) : (
                      <button
                        onClick={stopServer}
                        className="px-3 py-1 bg-red-600 hover:bg-red-500 rounded text-[13px] font-medium text-white transition-colors"
                      >
                        Stop Server
                      </button>
                    )}
                  </div>
                  {serverStatus && (
                    <div className="text-[11px] text-gray-400 mt-2">
                      {serverStatus.cuda_available ? '✓ CUDA Available' : '✗ CPU Only'}
                    </div>
                  )}
                </div>

                {/* Python Packages Section */}
                <div className="bg-gray-800 rounded p-4 border border-gray-700">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="m-0 text-[13px] font-medium text-gray-300">
                      Python Packages {checkingDeps && <span className="text-gray-500 font-normal text-[11px]">(checking...)</span>}
                    </h4>
                    <div className="flex gap-2">
                      <button
                        onClick={checkDeps}
                        disabled={!selectedVenv || checkingDeps}
                        className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${!selectedVenv || checkingDeps ? 'bg-gray-600 opacity-50 cursor-not-allowed' : 'bg-gray-600 hover:bg-gray-500 text-white'}`}
                      >
                        Refresh
                      </button>
                      <button
                        onClick={installMissingDeps}
                        disabled={!selectedVenv || installingDeps || REQUIRED_PACKAGES.every(p => depsStatus[p]?.installed)}
                        className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${REQUIRED_PACKAGES.every(p => depsStatus[p]?.installed) ? 'bg-gray-600 text-gray-400 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`}
                      >
                        {installingDeps ? `Installing ${installingPackage}...` : 'Install All'}
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    {REQUIRED_PACKAGES.map(pkg => {
                      const status = depsStatus[pkg];
                      const isInstalled = status?.installed;
                      return (
                        <div
                          key={pkg}
                          className={`py-1.5 px-2.5 rounded text-[11px] flex items-center gap-1.5 ${
                            isInstalled ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/15 text-red-300'
                          }`}
                        >
                          <span>{isInstalled ? '✓' : '✗'}</span>
                          <span>{pkg}</span>
                          {status?.version && <span className="text-gray-500 text-[10px]">({status.version})</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Info Box */}
                <div className="bg-gray-800 rounded p-3 border border-gray-700">
                  <div className="text-[11px] text-gray-400">
                    <p className="m-0 mb-1">1. Select a Python virtual environment</p>
                    <p className="m-0 mb-1">2. Click "Install All" to download dependencies</p>
                    <p className="m-0">3. Click "Start Server" to launch the backend</p>
                  </div>
                </div>
              </div>
            )}

            {/* Data Tab */}
            {activeTab === "data" && (
              <div>
                {datasetInfo?.preview ? (
                  <div className="overflow-x-auto">
                    <div className="text-sm text-gray-400 mb-2">Preview (first 10 rows)</div>
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="bg-gray-800">
                          {datasetInfo.columns?.map((col, i) => (
                            <th key={i} className="border border-gray-700 px-2 py-1 text-left text-gray-300 font-medium">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {datasetInfo.preview.map((row, i) => (
                          <tr key={i} className="hover:bg-gray-800/50">
                            {datasetInfo.columns?.map((col, j) => (
                              <td key={j} className="border border-gray-700 px-2 py-1 text-gray-400 truncate max-w-32">
                                {row[col] ?? ""}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-48 text-gray-500">
                    Load a dataset to see preview
                  </div>
                )}
              </div>
            )}

            {/* Profile Tab */}
            {activeTab === "profile" && (
              <div>
                {profile ? (
                  <div className="space-y-4">
                    <div className="bg-gray-800 rounded p-4">
                      <div className="text-sm font-medium text-emerald-400 mb-2">Summary</div>
                      <div className="text-sm text-gray-300">{profile.summary_text}</div>
                    </div>

                    <div className="bg-gray-800 rounded p-4">
                      <div className="text-sm font-medium text-emerald-400 mb-2">Column Details</div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="text-left text-gray-400">
                              <th className="pb-2 pr-4">Column</th>
                              <th className="pb-2 pr-4">Type</th>
                              <th className="pb-2 pr-4">Missing</th>
                              <th className="pb-2 pr-4">Unique</th>
                              <th className="pb-2">Stats</th>
                            </tr>
                          </thead>
                          <tbody>
                            {profile.column_profiles?.map((col, i) => (
                              <tr key={i} className="border-t border-gray-700">
                                <td className="py-2 pr-4 text-gray-300">{col.name}</td>
                                <td className="py-2 pr-4 text-gray-400">{col.column_type || col.dtype}</td>
                                <td className="py-2 pr-4">
                                  <span className={col.missing_pct > 10 ? "text-red-400" : "text-gray-400"}>
                                    {col.missing_pct}%
                                  </span>
                                </td>
                                <td className="py-2 pr-4 text-gray-400">{col.unique_count}</td>
                                <td className="py-2 text-gray-400">
                                  {col.mean !== undefined && `μ=${col.mean?.toFixed(2)}`}
                                  {col.outlier_pct > 0 && ` | outliers=${col.outlier_pct}%`}
                                  {col.high_cardinality && " | high cardinality"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-48 text-gray-500">
                    Run analysis to generate profile
                  </div>
                )}
              </div>
            )}

            {/* Cleaning Tab */}
            {activeTab === "cleaning" && (
              <div className="space-y-6">
                {/* Phase 1: Show cleaning options before cleaning is applied */}
                {analysisPhase === "cleaning" && !cleaningSummary && (
                  <div className="space-y-6">
                    <div className="bg-gray-800 rounded-lg p-4">
                      <div className="text-lg font-medium text-emerald-400 mb-4">Data Cleaning Options</div>

                      {/* Dataset info */}
                      {profile && (
                        <div className="mb-4 p-3 bg-gray-900 rounded text-sm">
                          <div className="text-gray-400 mb-2">Current Dataset:</div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div><span className="text-gray-500">Rows:</span> <span className="text-gray-200">{profile.n_rows}</span></div>
                            <div><span className="text-gray-500">Columns:</span> <span className="text-gray-200">{profile.n_cols}</span></div>
                            <div><span className="text-gray-500">Missing Values:</span> <span className="text-amber-400">{profile.missing_total || 0}</span></div>
                            <div><span className="text-gray-500">Duplicates:</span> <span className="text-amber-400">{profile.duplicate_count || 0}</span></div>
                          </div>
                        </div>
                      )}

                      {/* Target Column Selector (for prediction tasks) */}
                      {targetInference && targetInference.task_type !== "eda" && (
                        <div className="mb-4 p-3 bg-indigo-900/30 border border-indigo-700/50 rounded">
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-sm font-medium text-indigo-400">
                              {targetInference.task_type === "classification" ? "Classification Target" :
                               targetInference.task_type === "regression" ? "Regression Target" :
                               "Prediction Target"}
                            </div>
                            {selectedTarget && !showTargetSelector && (
                              <button
                                onClick={() => setShowTargetSelector(true)}
                                className="text-xs text-indigo-400 hover:text-indigo-300"
                              >
                                Change
                              </button>
                            )}
                          </div>

                          {targetInference.assumption_note && (
                            <div className="text-xs text-gray-400 mb-2">{targetInference.assumption_note}</div>
                          )}

                          {/* Show auto-selected target */}
                          {selectedTarget && !showTargetSelector && (
                            <div className="flex items-center gap-2 text-sm">
                              <span className="text-gray-400">Auto-selected:</span>
                              <span className="text-indigo-300 font-mono">{selectedTarget}</span>
                            </div>
                          )}

                          {/* Show target selector dropdown */}
                          {(showTargetSelector || (!selectedTarget && targetInference.target_candidates?.length > 0)) && (
                            <div className="space-y-2">
                              <select
                                value={selectedTarget || ""}
                                onChange={(e) => setTargetColumn(e.target.value || null)}
                                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200"
                              >
                                <option value="">Select a target column...</option>
                                {targetInference.target_candidates?.map((c) => (
                                  <option key={c.column} value={c.column}>
                                    {c.column} {c.is_leakage_risk ? "(⚠️ potential leakage)" : ""} - {Math.round(c.confidence * 100)}% confidence
                                  </option>
                                ))}
                              </select>
                              {targetInference.target_candidates?.find(c => c.column === selectedTarget)?.warning && (
                                <div className="text-xs text-amber-400">
                                  ⚠️ {targetInference.target_candidates.find(c => c.column === selectedTarget)?.warning}
                                </div>
                              )}
                            </div>
                          )}

                          {/* No candidates found */}
                          {targetInference.target_candidates?.length === 0 && (
                            <div className="text-xs text-gray-500">
                              To train a predictive model, select a target column (e.g., full_time_result).
                            </div>
                          )}
                        </div>
                      )}

                      {/* Safe cleaning explanation */}
                      <div className="mb-4">
                        <div className="text-sm font-medium text-emerald-400 mb-2">Safe Cleaning (Always Applied):</div>
                        <ul className="text-xs text-gray-400 space-y-1 ml-4 list-disc">
                          <li>Normalize column names to snake_case (e.g., "Team Name" → "team_name")</li>
                          <li>Trim whitespace from text values</li>
                          <li>Standardize missing value tokens (N/A, null, None, etc. → NaN)</li>
                          <li>Remove exact duplicate rows</li>
                        </ul>
                      </div>

                      {/* Recommended cleaning explanation */}
                      <div className="mb-4">
                        <div className="text-sm font-medium text-amber-400 mb-2">Recommended Cleaning (Optional):</div>
                        <ul className="text-xs text-gray-400 space-y-1 ml-4 list-disc">
                          <li>Fill missing numeric values with column median</li>
                          <li>Fill missing categorical values with mode (most common)</li>
                          <li>Clip outliers to 1st-99th percentile range</li>
                          <li>Group rare categories (&lt;1% frequency) as "Other"</li>
                        </ul>
                      </div>

                      {/* Info about cleaning logic */}
                      <div className="mb-6 p-3 bg-blue-900/30 border border-blue-700/50 rounded text-xs text-blue-300">
                        <strong>Note:</strong> Cleaning uses rule-based logic (not LLM) for predictable, fast results.
                      </div>

                      {/* Single cleaning button */}
                      <div className="mt-4">
                        <button
                          onClick={() => continueAnalysis(false)}
                          disabled={loading.clean || loading.charts}
                          className="w-full px-4 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg font-medium transition-colors text-lg"
                        >
                          {loading.clean ? "Cleaning..." : loading.charts ? "Generating Charts..." : "Apply Cleaning & Generate Charts"}
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Phase 2: Show cleaning results after cleaning is applied */}
                {cleaningSummary && (
                  <div className="space-y-4">
                    {/* Before/After Comparison */}
                    {cleaningSummary.before && cleaningSummary.after && (
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-sm font-medium text-emerald-400 mb-3">Before / After Comparison</div>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-gray-900 rounded p-3">
                            <div className="text-xs text-gray-500 mb-2">BEFORE CLEANING</div>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Rows:</span>
                                <span className="text-gray-200">{cleaningSummary.before.rows}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Columns:</span>
                                <span className="text-gray-200">{cleaningSummary.before.columns}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Missing Values:</span>
                                <span className="text-red-400">{cleaningSummary.before.missing_total}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Duplicates:</span>
                                <span className="text-amber-400">{cleaningSummary.before.duplicates}</span>
                              </div>
                            </div>
                          </div>
                          <div className="bg-gray-900 rounded p-3">
                            <div className="text-xs text-gray-500 mb-2">AFTER CLEANING</div>
                            <div className="space-y-1 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Rows:</span>
                                <span className="text-gray-200">{cleaningSummary.after.rows}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Columns:</span>
                                <span className="text-gray-200">{cleaningSummary.after.columns}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Missing Values:</span>
                                <span className={cleaningSummary.after.missing_total <= cleaningSummary.before.missing_total ? "text-emerald-400" : "text-amber-400"}>
                                  {cleaningSummary.after.missing_total}
                                  {cleaningSummary.after.missing_total > cleaningSummary.before.missing_total && (
                                    <span className="text-xs text-gray-500 ml-1">
                                      (+{cleaningSummary.after.missing_total - cleaningSummary.before.missing_total} identified)
                                    </span>
                                  )}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Duplicates:</span>
                                <span className="text-emerald-400">{cleaningSummary.after.duplicates}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                        {/* Explanation note when missing values increase */}
                        {cleaningSummary.after.missing_total > cleaningSummary.before.missing_total && (
                          <div className="mt-3 p-2 bg-blue-900/30 border border-blue-700/50 rounded text-xs text-blue-300">
                            <strong>Note:</strong> Missing values increased because empty strings, dashes, and other placeholder values
                            were identified and converted to proper missing values (NaN). This is expected - these values were always
                            missing, just not counted before.
                          </div>
                        )}
                      </div>
                    )}

                    {/* Columns Renamed */}
                    {cleaningSummary.columns_renamed && cleaningSummary.columns_renamed.length > 0 && (
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-sm font-medium text-amber-400 mb-3">
                          Columns Renamed ({cleaningSummary.columns_renamed_count} columns)
                        </div>
                        <div className="max-h-48 overflow-y-auto space-y-1">
                          {cleaningSummary.columns_renamed.map((rename, i) => (
                            <div key={i} className="text-xs flex items-center gap-2">
                              <span className="text-gray-400 font-mono truncate max-w-[40%]">{rename.original}</span>
                              <span className="text-gray-600">→</span>
                              <span className="text-emerald-400 font-mono truncate max-w-[40%]">{rename.cleaned}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Download Cleaned Dataset Button */}
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <button
                        onClick={downloadCleaned}
                        className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download Cleaned Dataset (CSV)
                      </button>
                    </div>
                  </div>
                )}

                {/* Cleaning Actions Log */}
                {cleaningLog.length > 0 && (
                  <div className="space-y-4">
                    <div className="text-sm font-medium text-emerald-400">Cleaning Actions Log</div>
                    <div className="space-y-2">
                      {cleaningLog.map((entry, i) => (
                        <div
                          key={i}
                          className={`bg-gray-800 rounded p-3 border-l-2 ${
                            entry.action.includes("complete")
                              ? "border-emerald-500"
                              : entry.action.includes("recommended")
                              ? "border-amber-500"
                              : "border-gray-600"
                          }`}
                        >
                          <div className="text-xs font-medium text-gray-300">{entry.action}</div>
                          <div className="text-xs text-gray-400 mt-1">{entry.details}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Initial state - no analysis run yet */}
                {analysisPhase === "idle" && !cleaningLog.length && (
                  <div className="flex items-center justify-center h-48 text-gray-500">
                    Click "Run Analysis" to start the data cleaning process
                  </div>
                )}
              </div>
            )}

            {/* Charts Tab */}
            {activeTab === "charts" && (
              <div>
                {charts.length > 0 ? (
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <div className="text-sm font-medium text-emerald-400">
                        Visualizations ({charts.length} charts)
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={regenerateExplanations}
                          disabled={loading.explain}
                          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded text-sm transition-colors"
                        >
                          {loading.explain ? "Generating..." : "Regenerate Explanations"}
                        </button>
                        <button
                          onClick={downloadCharts}
                          className="px-3 py-1 bg-emerald-700 hover:bg-emerald-600 rounded text-sm transition-colors"
                        >
                          Download ZIP
                        </button>
                        <button
                          onClick={downloadCleaned}
                          className="px-3 py-1 bg-blue-700 hover:bg-blue-600 rounded text-sm transition-colors"
                        >
                          Download CSV
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                      {charts.map((chart, i) => {
                        const explanation = explanations.find((e) => e.chart_index === chart.index);
                        return (
                          <div key={i} className="bg-gray-800 rounded overflow-hidden">
                            <div className="p-2 border-b border-gray-700">
                              <div className="text-sm font-medium text-gray-300">
                                {chart.index}. {chart.type?.replace(/_/g, " ")}
                              </div>
                            </div>
                            <div className="p-2">
                              <img
                                src={`${getServerUrl()}/chart/${chart.filename}`}
                                alt={chart.type}
                                className="w-full rounded bg-white"
                              />
                            </div>
                            {explanation && (
                              <div className="p-3 border-t border-gray-700 bg-gray-850">
                                <div className="text-xs text-gray-300 leading-relaxed">
                                  {explanation.explanation}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-48 text-gray-500">
                    Run analysis to generate charts
                  </div>
                )}
              </div>
            )}

            {/* Modeling Tab */}
            {activeTab === "modeling" && (
              <div className="space-y-6">
                {/* Header */}
                <div className="flex justify-between items-center">
                  <div className="text-lg font-medium text-emerald-400">Machine Learning Model</div>
                </div>

                {/* Target Selection (if prediction task but no target confirmed) */}
                {targetInference && targetInference.task_type !== "eda" && !selectedTarget && (
                  <div className="bg-amber-900/30 border border-amber-700/50 rounded-lg p-4">
                    <div className="text-amber-400 font-medium mb-2">Target Column Required</div>
                    <div className="text-sm text-gray-400 mb-3">
                      To train a predictive model, select a target column.
                    </div>
                    <select
                      value={selectedTarget || ""}
                      onChange={(e) => setTargetColumn(e.target.value || null)}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200"
                    >
                      <option value="">Select a target column...</option>
                      {targetInference.target_candidates?.map((c) => (
                        <option key={c.column} value={c.column}>
                          {c.column} {c.is_leakage_risk ? "(⚠️ potential leakage)" : ""} - {Math.round(c.confidence * 100)}% confidence
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Model Configuration */}
                {selectedTarget && (
                  <div className="bg-gray-800 rounded-lg p-4">
                    <div className="text-sm font-medium text-emerald-400 mb-3">Model Configuration</div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Target Column</div>
                        <div className="text-sm text-indigo-300 font-mono">{selectedTarget}</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-1">Task Type</div>
                        <div className="text-sm text-gray-200">
                          {targetInference?.task_type === "classification" ? "Classification" :
                           targetInference?.task_type === "regression" ? "Regression" :
                           "Auto-detect"}
                        </div>
                      </div>
                    </div>

                    <div className="mb-4">
                      <label className="block text-xs text-gray-500 mb-1">Max Iterations</label>
                      <input
                        type="number"
                        min={100}
                        max={5000}
                        value={maxIter}
                        onChange={(e) => setMaxIter(Math.min(5000, Math.max(100, parseInt(e.target.value) || 1000)))}
                        className="w-32 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm"
                      />
                      <span className="text-xs text-gray-500 ml-2">(100 - 5000)</span>
                    </div>

                    <button
                      onClick={trainModel}
                      disabled={loadingModel || !selectedTarget}
                      className="w-full px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg font-medium transition-colors"
                    >
                      {loadingModel ? "Training Model..." : "Train Model"}
                    </button>
                  </div>
                )}

                {/* Model Results */}
                {modelResults && (
                  <div className="space-y-4">
                    {/* Metrics Card */}
                    <div className="bg-gray-800 rounded-lg p-4">
                      <div className="text-sm font-medium text-emerald-400 mb-3">Model Performance</div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {modelResults.task_type === "classification" ? (
                          <>
                            <div className="bg-gray-900 rounded p-3 text-center">
                              <div className="text-2xl font-bold text-emerald-400">
                                {(modelResults.metrics.accuracy * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs text-gray-500">Accuracy</div>
                            </div>
                            <div className="bg-gray-900 rounded p-3 text-center">
                              <div className="text-2xl font-bold text-blue-400">
                                {(modelResults.metrics.macro_f1 * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs text-gray-500">Macro F1</div>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="bg-gray-900 rounded p-3 text-center">
                              <div className="text-2xl font-bold text-emerald-400">
                                {modelResults.metrics.r2.toFixed(3)}
                              </div>
                              <div className="text-xs text-gray-500">R²</div>
                            </div>
                            <div className="bg-gray-900 rounded p-3 text-center">
                              <div className="text-2xl font-bold text-blue-400">
                                {modelResults.metrics.rmse.toFixed(2)}
                              </div>
                              <div className="text-xs text-gray-500">RMSE</div>
                            </div>
                          </>
                        )}
                        <div className="bg-gray-900 rounded p-3 text-center">
                          <div className="text-lg font-bold text-gray-300">{modelResults.model_summary.train_rows}</div>
                          <div className="text-xs text-gray-500">Train Rows</div>
                        </div>
                        <div className="bg-gray-900 rounded p-3 text-center">
                          <div className="text-lg font-bold text-gray-300">{modelResults.model_summary.test_rows}</div>
                          <div className="text-xs text-gray-500">Test Rows</div>
                        </div>
                      </div>
                    </div>

                    {/* Artifacts (Confusion Matrix / Residuals) */}
                    {modelResults.artifacts && (
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-sm font-medium text-emerald-400 mb-3">
                          {modelResults.task_type === "classification" ? "Confusion Matrix" : "Residuals Plot"}
                        </div>
                        <img
                          src={`${getServerUrl()}/model/artifact/${modelResults.artifacts.filename}`}
                          alt="Model Artifact"
                          className="w-full max-w-lg mx-auto rounded bg-white"
                        />
                      </div>
                    )}

                    {/* Feature Importance */}
                    {modelResults.model_summary.feature_importance?.length > 0 && (
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="text-sm font-medium text-emerald-400 mb-3">
                          Top Features (Model Reliance)
                        </div>
                        <div className="space-y-2">
                          {modelResults.model_summary.feature_importance.slice(0, 10).map((f, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <span className="text-xs text-gray-500 w-4">{i + 1}.</span>
                              <span className="text-sm text-gray-300 flex-1 truncate">{f.feature}</span>
                              <span className={`text-sm ${f.coefficient >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                {f.coefficient >= 0 ? '+' : ''}{f.coefficient.toFixed(3)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Modeling Summary / Final Answer */}
                <div className="mt-6 bg-gradient-to-r from-indigo-900/30 to-emerald-900/30 rounded-lg border border-indigo-700/50">
                  <div className="p-4 border-b border-indigo-700/30">
                    <div className="flex justify-between items-center">
                      <div className="text-sm font-medium text-indigo-400">
                        📊 Modeling Summary / Final Answer
                      </div>
                      <button
                        onClick={async () => {
                          setLoadingFinalSummary(true);
                          try {
                            const res = await fetch(`${getServerUrl()}/final-summary`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ intent, allow_sample_rows: allowSampleRows }),
                            });
                            const data = await res.json();
                            if (data.success) {
                              setFinalSummary(data.data.summary);
                            }
                          } catch (err) {
                            console.error("Failed to regenerate summary:", err);
                          } finally {
                            setLoadingFinalSummary(false);
                          }
                        }}
                        disabled={loadingFinalSummary}
                        className="px-3 py-1 bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 rounded text-xs transition-colors"
                      >
                        {loadingFinalSummary ? "Generating..." : "Regenerate Summary"}
                      </button>
                    </div>
                    {intent && (
                      <div className="mt-2 text-xs text-gray-400">
                        Question: <span className="text-gray-300 italic">"{intent}"</span>
                      </div>
                    )}
                  </div>
                  <div className="p-4">
                    {loadingFinalSummary ? (
                      <div className="flex items-center justify-center py-8">
                        <div className="text-gray-400 animate-pulse">Analyzing data and generating summary...</div>
                      </div>
                    ) : finalSummary ? (
                      <div className="prose prose-invert prose-sm max-w-none">
                        <div className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">
                          {finalSummary}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500 text-sm">
                        {modelResults
                          ? "Model trained! Click 'Regenerate Summary' for insights."
                          : "Train a model or run analysis to see the final summary here."}
                      </div>
                    )}
                  </div>
                </div>

                {/* EDA-only message */}
                {targetInference?.task_type === "eda" && (
                  <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4 text-blue-300 text-sm">
                    <strong>Exploratory Analysis Mode:</strong> Your intent suggests exploratory data analysis (EDA)
                    rather than prediction. No model training is required. Review the Charts tab for insights.
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Export for ContextUI
window.DataNarrateWindow = DataNarrateWindow;
