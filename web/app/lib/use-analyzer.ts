import { useState, useCallback } from "react";
import type { AnalyzerState, AnalyzeResponse } from "./types";

export function useAnalyzer() {
  const [state, setState] = useState<AnalyzerState>({
    status: "idle",
    data: null,
    error: null,
    fileName: null,
  });

  const analyze = useCallback(async (file: File) => {
    setState({ status: "loading", data: null, error: null, fileName: file.name });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/analyze", { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data: AnalyzeResponse = await res.json();
      setState({ status: "ready", data, error: null, fileName: file.name });
    } catch (e) {
      setState({
        status: "error",
        data: null,
        error: e instanceof Error ? e.message : String(e),
        fileName: file.name,
      });
    }
  }, []);

  const reset = useCallback(() => {
    setState({ status: "idle", data: null, error: null, fileName: null });
  }, []);

  return { ...state, analyze, reset };
}
