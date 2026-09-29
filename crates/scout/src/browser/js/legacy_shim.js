// Compatibility shim for pages written against the legacy WebMCP API
// (`navigator.modelContext.registerTool/unregisterTool/provideContext/clearContext`).
// Maps it onto `document.modelContext.registerTool(tool, { signal })`, where
// unregistration happens through an AbortController. Injected before any page
// script runs (Page.addScriptToEvaluateOnNewDocument). No-op when the page
// already has `navigator.modelContext` or when WebMCP is unavailable.
(() => {
  const mc = document.modelContext;
  if (!mc || navigator.modelContext) return;
  const ctrls = new Map();
  const reg = (t) => {
    ctrls.get(t.name)?.abort();
    const c = new AbortController(); ctrls.set(t.name, c);
    const schema = t.inputSchema ?? { type: "object", properties: {} };
    return mc.registerTool({ ...t, inputSchema: schema }, { signal: c.signal });
  };
  Object.defineProperty(navigator, "modelContext", { configurable: true, value: {
    registerTool: reg,
    unregisterTool: (name) => { ctrls.get(name)?.abort(); ctrls.delete(name); },
    provideContext: ({ tools = [] } = {}) => { for (const c of ctrls.values()) c.abort(); ctrls.clear(); return Promise.all(tools.map(reg)); },
    clearContext: () => { for (const c of ctrls.values()) c.abort(); ctrls.clear(); },
  }});
})();
