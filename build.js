const fs = require("fs");
const path = require("path");

async function* walk(dir) {
  for await (const d of await fs.promises.opendir(dir)) {
    const entry = path.join(dir, d.name);
    if (d.isDirectory()) yield* walk(entry);
    else if (d.isFile()) yield entry;
  }
}

// Then, use it with a simple async for loop
async function main() {
  const files = [];
  for await (const p of walk("config_templates")) {
    if (/\.ya?ml$/.test(p)) files.push(p);
  }

  console.log(JSON.stringify({ files }));
}

main();
