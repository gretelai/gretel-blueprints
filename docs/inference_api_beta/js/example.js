/*
REQUIREMENTS
1. Requires Node v18 or later
2. Must set GRETEL_API_KEY env variable before running. Get your key from https://console.gretel.ai/users/me/key
3. Maximum of 50 rows. Use the standard /models batch API for more.

EXAMPLES

1. Using default settings: 

GRETEL_API_KEY={your key} node example.js --prompt="generate a users table"

2. Using advanced settings:

GRETEL_API_KEY={your key} node example.js --prompt="generate a users table" --num_rows=40 \
--model_id=gretelai/tabular-v0c --temperature=0.9 --top_k=20 --top_p=0.6

3. Get list of available inference models:

GRETEL_API_KEY={your key} node example.js --getModels
*/

import { getModels, createStructuredData } from "./streaming_tabllm.js";

const main = async () => {
  // Check for API key
  if (!process.env.GRETEL_API_KEY) {
    console.log("GRETEL_API_KEY environment variable not set");
    return;
  }

  // Read CLI arguments
  const args = {};
  for (const arg of process.argv) {
    if (arg.indexOf("--") === 0) {
      const parts = arg.substring(2).split("=");
      if (parts.length === 2) {
        args[parts[0]] = parts[1];
      } else if (parts.length === 1) {
        args[parts[0]] = true;
      }
    }
  }

  if (args.getModels) {
    const models = getModels().then((models) => {
      console.log(models);
    });
  } else {
    // Check for prompt
    if (!args.prompt) {
      console.log("Prompt must be specified with --prompt=");
      return;
    }

    let params = {
      temperature: args.temperature,
      top_k: args.top_k,
      top_p: args.top_p,
    };

    let result = [];
    const rowCallback = (row) => {
      result = result.concat(row.table_data);
      console.table(result);
    };

    createStructuredData(
      args.prompt,
      rowCallback,
      args.num_rows,
      args.model_id,
      params
    );
  }
};

main();