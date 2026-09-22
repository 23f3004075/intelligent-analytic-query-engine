/**
 * IAQE Web Dashboard — Vue 3 Application Logic
 */

const { createApp, ref, computed, onMounted } = Vue;

createApp({
  setup() {
    // ── Reactive State ──
    const queryText = ref("");
    const selectedTrack = ref("auto");
    const loading = ref(false);
    const activePipelineStep = ref("Initializing query...");
    const result = ref(null);
    const error = ref(null);
    const samples = ref([]);
    const serverInfo = ref(null);
    const serverReady = ref(false);
    const showCode = ref(true);
    const codeCopied = ref(false);
    const userFeedback = ref(null);
    const feedbackSuccessMsg = ref(null);

    let stepInterval = null;

    // ── Computed Properties ──
    const modelName = computed(() => {
      return serverInfo.value?.model || "";
    });

    const resultColumns = computed(() => {
      if (!result.value || !result.value.result || result.value.result.length === 0) {
        return [];
      }
      return Object.keys(result.value.result[0]);
    });

    // ── Lifecycle Hooks ──
    onMounted(() => {
      fetchServerInfo();
      fetchSamples();
    });

    // ── API Interactions ──
    async function fetchServerInfo() {
      try {
        const res = await fetch("/api/info");
        if (res.ok) {
          serverInfo.value = await res.json();
          serverReady.value = true;
        }
      } catch (err) {
        console.warn("Could not fetch server info:", err);
      }
    }

    async function fetchSamples() {
      try {
        const res = await fetch("/api/samples");
        if (res.ok) {
          samples.value = await res.json();
        }
      } catch (err) {
        console.warn("Could not fetch sample queries:", err);
      }
    }

    function loadSample(q) {
      queryText.value = q;
      error.value = null;
    }

    function clearQuery() {
      queryText.value = "";
      result.value = null;
      error.value = null;
      userFeedback.value = null;
      feedbackSuccessMsg.value = null;
    }

    function startPipelineAnimation() {
      const steps = [
        "1. Normalizing & checking domain vocabulary...",
        "2. Building schema digest & glossary...",
        "3. Classifying query intent with LLM...",
        "4. Retrieving relevant few-shot feedback...",
        "5. Generating SQL & Pandas logic...",
        "6. Executing code in DuckDB sandbox...",
        "7. Validating result against intent...",
        "8. Computing multi-signal confidence score...",
        "9. Generating 3-part business explanation...",
        "10. Persisting execution telemetry...",
      ];
      let stepIdx = 0;
      activePipelineStep.value = steps[0];
      stepInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        activePipelineStep.value = steps[stepIdx];
      }, 1400);
    }

    function stopPipelineAnimation() {
      if (stepInterval) {
        clearInterval(stepInterval);
        stepInterval = null;
      }
    }

    async function executeCurrentQuery() {
      const q = queryText.value.trim();
      if (!q || loading.value) return;

      loading.value = true;
      error.value = null;
      userFeedback.value = null;
      feedbackSuccessMsg.value = null;
      startPipelineAnimation();

      try {
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: q,
            track: selectedTrack.value,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned ${res.status}`);
        }

        const data = await res.json();
        result.value = data;
      } catch (err) {
        error.value = err.message || "An unexpected error occurred during execution.";
      } finally {
        loading.value = false;
        stopPipelineAnimation();
      }
    }

    async function submitUserFeedback(wasCorrect) {
      if (!result.value) return;
      userFeedback.value = wasCorrect;

      const entryId = result.value.metadata?.entry_id;

      try {
        const res = await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            entry_id: entryId !== undefined ? entryId : null,
            was_correct: wasCorrect,
            note: wasCorrect ? "Marked accurate via Web Dashboard" : "Marked inaccurate via Web Dashboard",
          }),
        });

        if (res.ok) {
          feedbackSuccessMsg.value = wasCorrect
            ? "Accurate feedback recorded. Boost will be applied to similar queries."
            : "Feedback recorded. Guardrail will prevent repeating this error.";
          setTimeout(() => {
            feedbackSuccessMsg.value = null;
          }, 4000);
        }
      } catch (err) {
        console.warn("Feedback submission error:", err);
      }
    }

    function exportToCSV() {
      if (!result.value || !result.value.result || result.value.result.length === 0) return;

      const rows = result.value.result;
      const cols = Object.keys(rows[0]);

      const csvContent = [
        cols.join(","),
        ...rows.map(row =>
          cols.map(c => {
            const val = row[c] === null || row[c] === undefined ? "" : String(row[c]);
            return `"${val.replace(/"/g, '""')}"`;
          }).join(",")
        ),
      ].join("\n");

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `iaqe_result_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }

    function copyCode() {
      if (!result.value || !result.value.generated_logic) return;
      navigator.clipboard.writeText(result.value.generated_logic).then(() => {
        codeCopied.value = true;
        setTimeout(() => {
          codeCopied.value = false;
        }, 2000);
      });
    }

    // ── Helper Formatting Functions ──
    function getConfidenceClass(score) {
      if (score === null || score === undefined) return "confidence-medium";
      if (score >= 0.85) return "confidence-high";
      if (score >= 0.65) return "confidence-medium";
      if (score >= 0.40) return "confidence-low";
      return "confidence-unreliable";
    }

    function getConfidenceLabel(score) {
      if (score === null || score === undefined) return "UNKNOWN";
      if (score >= 0.85) return "HIGH CONFIDENCE";
      if (score >= 0.65) return "MEDIUM CONFIDENCE";
      if (score >= 0.40) return "LOW CONFIDENCE";
      return "UNRELIABLE";
    }

    function formatCellValue(val) {
      if (val === null || val === undefined) return "—";
      if (typeof val === "number") {
        // If float with decimals
        if (!Number.isInteger(val)) {
          return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        return val.toLocaleString();
      }
      return String(val);
    }

    return {
      queryText,
      selectedTrack,
      loading,
      activePipelineStep,
      result,
      error,
      samples,
      serverInfo,
      serverReady,
      showCode,
      codeCopied,
      userFeedback,
      feedbackSuccessMsg,
      modelName,
      resultColumns,
      loadSample,
      clearQuery,
      executeCurrentQuery,
      submitUserFeedback,
      exportToCSV,
      copyCode,
      getConfidenceClass,
      getConfidenceLabel,
      formatCellValue,
    };
  },
}).mount("#app");
