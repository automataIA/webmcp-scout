// Called with Runtime.callFunctionOn(name, args, encoding). Arguments arrive as
// CDP call arguments, never interpolated into source. Tool errors are returned
// as data so they are distinguishable from CDP/navigation failures.
async function (name, args, encoding) {
  const mc = document.modelContext;
  if (!mc || typeof mc.executeTool !== "function") {
    return { status: "unsupported", url: location.href };
  }
  const tools = await mc.getTools();
  const tool = tools.find(
    (t) => t.name === name && (t.origin || location.origin) === location.origin,
  );
  if (!tool) return { status: "not_found", url: location.href };
  const input = encoding === "json_string" ? JSON.stringify(args) : args;
  try {
    const out = await mc.executeTool(tool, input);
    const result = out === undefined || out === null ? null : typeof out === "string" ? out : JSON.stringify(out);
    return { status: "ok", result, url: location.href };
  } catch (e) {
    return { status: "error", error: String((e && e.message) || e), url: location.href };
  }
}
