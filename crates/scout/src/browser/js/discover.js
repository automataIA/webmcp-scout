// Called with Runtime.callFunctionOn. Returns the tools registered in this
// document. `inputSchema` is a JSON string on Chrome 154 while the draft spec
// exposes an object: normalize to a string so Rust parses one shape.
async function () {
  const mc = document.modelContext;
  if (!mc || typeof mc.getTools !== "function") {
    return { supported: false, url: location.href, origin: location.origin, tools: [] };
  }
  const tools = await mc.getTools();
  return {
    supported: true,
    url: location.href,
    origin: location.origin,
    tools: tools.map((t) => ({
      name: t.name,
      title: t.title || null,
      description: t.description || "",
      inputSchema:
        t.inputSchema == null
          ? null
          : typeof t.inputSchema === "string"
            ? t.inputSchema
            : JSON.stringify(t.inputSchema),
      origin: t.origin || location.origin,
    })),
  };
}
