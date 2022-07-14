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
  const templates = [];
  for await (const p of walk("config_templates", /\.ya?ml$/)) {
    templates.push(p);
  }
  const samples = [];
  for await (const p of walk("sample_data")) {
    samples.push(p);
  }
  console.log(JSON.stringify({ templates, samples }));
}

main();
