const fs = require("fs");
const path = require("path");

async function* walk(dir, ext) {
  for await (const d of await fs.promises.opendir(dir)) {
    const entry = path.join(dir, d.name);
    if (d.isDirectory()) {
      yield* walk(entry, ext);
    } else if (d.isFile() && (!ext || ext.test(path.extname(d.name)))) {
      yield entry;
    }
  }
}

// Then, use it with a simple async for loop
async function main() {
  const configs = [];
  for await (const p of walk("config_templates", /\.ya?ml$/)) {
    configs.push(p);
  }
  const samples = [];
  for await (const p of walk("sample_data")) {
    samples.push(p);
  }
  const sample_data_previews = [];
  for await (const p of walk("sample_data_previews")) {
    sample_data_previews.push(p);
  }
  /* 
  Files whose first comment is `# deprecated: This configuration will be deprecated soon. <additional message>` are deprecated and filtered from the manifest
  */
  const templates = configs.filter((template) => {
    const str = fs.readFileSync(template, "utf8");
    return !str.startsWith("# deprecated");
  });

  console.log(JSON.stringify({ templates, samples, sample_data_previews }));
}

main();
