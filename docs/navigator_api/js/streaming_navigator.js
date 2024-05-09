const MAX_ROWS_PER_STREAM = 50;
const REQUEST_TIMEOUT_SEC = 60;
const STREAM_SLEEP_TIME = 0.5;
const POLLING_DELAY = 500; // ms
// Use reasonable default params if not set
const DEFAULT_HYPERPARAMS = {
  temperature: 0.7,
  top_k: 40,
  top_p: 0.9,
};

const BASE_URL = "https://api-dev.gretel.cloud/";
// const BASE_URL = "https://api.gretel.cloud/";

///////////////////////////////////////////////
// Generates structured data given a prompt //
/////////////////////////////////////////////
/**
 *
 * @param {string} prompt -- text describing what you want the model to generate data about.
 * @param {function} rowCallback -- function that receives the generated data. this will be called several times, depending on how much data is requested
 * @param {number, optional} num_rows -- number of rows to generate. Defaults to 10.
 * @param {string, optional} model_id -- string id of the model (from getModels) to use to generate data. If none provided, the latest default model will be used.
 * @param {object, optional} params -- hyperparameter settings. Includes: temperature, top_k, top_p.
 */
export const createStructuredData = async (
  prompt,
  rowCallback,
  num_rows = 10,
  model_id, // = "gretelai/tabular-v0",
  params
) => {
  const { stream_id } = await createStream({
    model_id,
    num_rows,
    prompt,
    params,
  });

  // Read from stream.
  let iterator = "";
  let resultCount = 0;
  while (true) {
    let iterateResult;
    try {
      iterateResult = await readFromStream({ stream_id, num_rows });
    } catch (err) {
      console.log(`Error iterating on stream: ${err}`);
      break;
    }

    iterator = iterateResult.next_iterator;

    for (const data of iterateResult.data) {
      if (data.data_type === "TabularResponse") {
        rowCallback(data.data);
        resultCount++;
      } else {
        console.log(data.data);
      }
    }
    if (
      iterateResult.stream_state.status === "closed" ||
      resultCount >= num_rows
    ) {
      break;
    }
    await new Promise((r) => setTimeout(r, POLLING_DELAY));
  }
};

const createStream = async ({ model_id, num_rows, prompt, params }) => {
  // Create stream
  const streamResponse = await _callAPI("v1/inference/tabular/stream", {
    model_id,
    num_rows,
    prompt,
    params: { ...DEFAULT_HYPERPARAMS, ...params },
  });

  const streamResult = await streamResponse.json();

  // if stream creation errored, surface to user and stop process.
  if (streamResponse.status !== 200) {
    console.log("Error creating stream", JSON.stringify(streamResult));
    return;
  }

  console.log("created stream:", streamResult); // logs stream id
  return streamResult;
};
const readFromStream = async ({ stream_id, num_rows }) => {
  const iterateResponse = await _callAPI(
    "v1/inference/tabular/stream/iterate",
    {
      stream_id,
      iterator,
      count: num_rows,
    }
  );

  const iterateResult = await iterateResponse.json();

  if (iterateResponse.status !== 200) {
    throw new Error(JSON.stringify(iterateResult));
  } else {
    /**
     * next_iterator, data, stream_state
     */
    return iterateResult;
  }
};

/////////////////////////////////////////////
// Returns a list of available models IDs //
///////////////////////////////////////////
export const getModels = async () => {
  const response = await _callAPI("v1/inference/models", null, "GET");
  if (response.status !== 200) {
    return `Unable to fetch models: ${response.stausText}`;
  }
  return response.json();
};

///////////////
// Internal //
//////////////
const _callAPI = (url, body, method = "POST") => {
  const req = {
    method,
    headers: {
      "Content-Type": "application/json",
      Authorization: process.env.GRETEL_API_KEY,
    },
  };

  if (method === "POST" && body) {
    req.body = JSON.stringify(body);
  }

  return fetch(BASE_URL + url, req);
};
