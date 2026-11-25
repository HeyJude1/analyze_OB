// kg-build 的 workflow 扇出示例。
// 用法：把本文件内容贴给 workflow 工具的 script；args 传：
//   { skillDir: "<abs>/skills/kg-build", files: [{op, path}], outDir: "<abs>/work", schemaHint: "7 实体/12 关系" }
// 注意：workflow 脚本没有文件系统权限，读源码/写文件都由子代理完成。

const files = args.files || [];
if (!files.length) throw new Error('args.files 为空');

const analyzePrompt = (f) =>
  `你是 kg-build 的 analyzer 子代理。\n` +
  `先读 ${args.skillDir}/agents/analyzer.md 与 ${args.skillDir}/references/schema.md，严格照做。\n` +
  `算子: ${f.op}\n源文件: ${f.path}\n` +
  `输出目录: ${args.outDir}/analysis/${f.op}/\n` +
  `约束: 每条策略必须带 line_start/line_end；条件必须用闭集谓词；禁止写性能数字。\n` +
  `只返回 JSON: {ok, op, file, out_path, strategies, patterns, errors[]}`;

const analyzeSchema = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    op: { type: 'string' },
    file: { type: 'string' },
    out_path: { type: 'string' },
    strategies: { type: 'integer' },
    patterns: { type: 'integer' },
    errors: { type: 'array', items: { type: 'string' } }
  },
  required: ['ok', 'op', 'file', 'out_path'],
  additionalProperties: true
};

phase('analyze');
const analyzed = await parallel(files.map((f) => () =>
  agent(analyzePrompt(f), {
    label: `${f.op}/${f.path.split(/[\\/]/).pop()}`,
    phase: 'analyze',
    schema: analyzeSchema
  })
));

const okFiles = analyzed.filter((r) => r && r.ok);
const failed = analyzed.filter((r) => !r || !r.ok);

log(`analyzer: ${okFiles.length}/${files.length} 成功`);

// 后续阶段（extract / resolve / build / verify / export）建议由主代理串行派发：
// 它们需要读全局产物、跑门、决定是否回灌；不适合再扇出。
return {
  analyzed: okFiles,
  failed,
  next: 'extractor'
};
