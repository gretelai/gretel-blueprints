///////////////////////////////////////////////
// Generates structured data given a prompt //
/////////////////////////////////////////////
export const createStructuredData = async (
  prompt,
  rowCallback,
  num_rows = 10,
  model_id = "gretelai/tabular-v0",
  params
) => {
  // Use reasonable default params if not set
  params = {
    temperature: params.temperature || 0.7,
    top_k: params.top_k || 40,
    top_p: params.top_p || 0.9,
  };

  // Create stream
  const streamResponse = await _callAPI("v1/inference/tabular/stream", {
    model_id: model_id,
    num_rows: num_rows,
    prompt: prompt,
    params: params,
  });
  const streamResult = await streamResponse.json();
  console.log(streamResult);

  // Read from stream.
  let iterator = "";
  let resultCount = 0;
  while (true) {
    const iterateResponse = await _callAPI(
      "v1/inference/tabular/stream/iterate",
      {
        stream_id: streamResult.stream_id,
        iterator: iterator,
        count: num_rows,
      }
    );
    const iterateResult = await iterateResponse.json();

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
    await new Promise((r) => setTimeout(r, 500));
  }
};

/////////////////////////////////////////////
// Returns a list of available models IDs //
///////////////////////////////////////////
export const getModels = async () => {
  const response = await _callAPI("v1/inference/models", null, "GET");
  return response.json();
};

///////////////
// Internal //
//////////////
const _callAPI = (url, body, method = "POST") => {
  const BASE_URL = "https://api.gretel.cloud/";
  const req = {
    method: method,
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
