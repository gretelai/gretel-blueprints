// -- constants and defaults --
const MAX_ROWS_PER_STREAM = 50;
const POLLING_DELAY = 500; // ms
// Use reasonable default params if not set
const DEFAULT_HYPERPARAMS = {
  temperature: 0.7,
  top_k: 40,
  top_p: 0.9,
};

const BASE_URL = "https://api.gretel.cloud/";

///////////////////////////////////////////////
// Generates structured data given a prompt //
/////////////////////////////////////////////
/**
 *
 * @param {string} prompt -- text describing what you want the model to generate data about.
 * @param {function} rowCallback -- function that receives the generated data. this will be called several times, depending on how much data is requested
 * @param {number, optional} num_rows -- number of rows to generate. Defaults to 10.
 * @param {string, optional} model_id -- string id of the model (from getModels) to use to generate data. If none provided, a default model will be used.
 * @param {object, optional} params -- hyperparameter settings. Includes: temperature, top_k, top_p.
 */
export const createStructuredData = async (
  prompt,
  rowCallback,
  num_rows = 10,
  model_id = "gretelai/tabular-v0",
  params
) => {
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
    return streamResult;
  };

  const readFromStream = async ({ stream_id, num_rows, iterator }) => {
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
      throw new Error(iterateResult);
    } else {
      return iterateResult;
    }
  };

  const getDataFromStream = async ({ stream_id, num_rows, rowHandler }) => {
    let iterator = "";
    let resultCount = 0;

    while (true) {
      let iterateResult;
      try {
        iterateResult = await readFromStream({ stream_id, num_rows, iterator });
      } catch (err) {
        console.log(`Error iterating on stream: ${JSON.stringify(err)}`);
        break;
      }

      iterator = iterateResult.next_iterator;

      for (const data of iterateResult.data) {
        if (data.data_type === "TabularResponse") {
          rowHandler(data.data);
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

  // Combines functions above to creates a single stream and then fetch data from that stream.
  // Currently, each stream can return a maximum of 50 rows.
  const createStreamAndMakeData = async ({
    model_id,
    num_rows,
    prompt,
    table_headers,
    table_data,
    params,
    rowHandler,
  }) => {
    const { stream_id } = await createStream({
      model_id,
      num_rows,
      prompt,
      table_headers,
      table_data,
      params,
    });

    return await getDataFromStream({ stream_id, num_rows, rowHandler });
  };

  /**
   *
   * Since each stream allows a MAX of 50 rows, create multiple streams as needed,
   * passing the results of the previous stream into the next stream so we don't duplicate data.
   *
   */
  const recursiveStreaming = async ({
    numRows,
    prompt,
    table_headers,
    table_data,
  }) => {
    let numRowsLeft = numRows;
    /**
     * gather generated data from current streaming to pass to
     * the next stream.
     */
    let results = [];

    const newRowHandler = (row) => {
      numRowsLeft--;
      // add to internal tracker
      results.push(row);
      // call external callback with current data
      rowCallback(row);
    };

    const numRowsForStream =
      numRows > MAX_ROWS_PER_STREAM ? MAX_ROWS_PER_STREAM : numRows;

    return createStreamAndMakeData({
      model_id,
      num_rows: numRowsForStream,
      prompt: prompt,
      params,
      table_data,
      table_headers,
      rowHandler: newRowHandler,
    }).then(() => {
      if (numRowsLeft > 0 && results.length) {
        const table_headers = results[0].table_headers;
        const tableHeadersString = table_headers.join(", ");
        const table_data = results.map((row) => row.table_data).flat();
        const lastRows = table_data.slice(Math.max(table_data.length - 3, 0));
        try {
          return recursiveStreaming({
            numRows: numRowsLeft,
            prompt: `Generate more data like the following table with the columns: ${tableHeadersString}`,
            table_headers,
            table_data: lastRows,
          });
        } catch (err) {
          console.log("Error in recursive streaming", err);
        }
      }
    });
  };

  return await recursiveStreaming({ numRows: num_rows, prompt });
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
